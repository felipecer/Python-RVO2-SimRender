from typing import Dict, Tuple
import rvo2_rl
import argparse
from pydantic import BaseModel, ValidationError
import math
import sys
import yaml
import numpy as np
from rendering.pygame_renderer import PyGameRenderer
from rendering.text_renderer import TextRenderer
from simulator.engines.base import SimulationEngine, SimulationState
from simulator.models.observer import SimulationSubject
from simulator.models.simulation import Simulation
from simulator.models.messages import (
    RayCastingUpdateMessage,
    SimulationInitializedMessage,
    AgentPositionsUpdateMessage,
    ObstaclesProcessedMessage,
    GoalsProcessedMessage
)
from simulator.models.simulation_configuration.simulation_events import GoalReachedEvent
from simulator.models.simulation_configuration.registry import global_registry
import traceback
import pprint


class RVO2SimulatorWrapper(SimulationEngine, SimulationSubject):
    def __init__(self, world_config: BaseModel, simulation_id: str, seed: int = None):
        SimulationEngine.__init__(self, seed=seed)
        SimulationSubject.__init__(self)
        """
        Initializes the RVO2 simulator with the world configuration and an optional renderer.

        Args:
            world_config (BaseModel): The world configuration in Pydantic format.
            simulation_id (str): The ID of the current simulation.
        """
        self.world_config = world_config  # Stores the provided world configuration
        self.simulation_id = simulation_id  # Stores the simulation ID
        self.sim = None  # Instance of the RVO2 simulator, will be initialized later
        self.agent_goals = {}  # Dictionary to store agent goals
        # self.steps_buffer = []  # Buffer to store data for each simulation step
        self.obstacles = []
        self.agent_initial_positions = []
        self._manual_velocity_updates = []
        self.intersect_list = None
        self.initial_distance_from_goal_array = []

    def calculate_preferred_velocity(self, agent_position, goal_position, max_speed):
        vector_to_goal = (
            goal_position[0] - agent_position[0],
            goal_position[1] - agent_position[1]
        )
        distance = math.sqrt(vector_to_goal[0] ** 2 + vector_to_goal[1] ** 2)

        if distance > 0:
            return (
                vector_to_goal[0] / distance * max_speed,
                vector_to_goal[1] / distance * max_speed
            )
        else:
            return (0, 0)

    def set_agent_defaults(self, agent_idx, agent_defaults):
        self.sim.setAgentNeighborDist(agent_idx, agent_defaults.neighbor_dist)
        self.sim.setAgentMaxNeighbors(agent_idx, agent_defaults.max_neighbors)
        self.sim.setAgentTimeHorizon(agent_idx, agent_defaults.time_horizon)
        self.sim.setAgentTimeHorizonObst(
            agent_idx, agent_defaults.time_horizon_obst)
        self.sim.setAgentRadius(agent_idx, agent_defaults.radius)
        self.sim.setAgentMaxSpeed(agent_idx, agent_defaults.max_speed)
        self.sim.setAgentVelocity(agent_idx, agent_defaults.velocity)

    def initialize_simulation(self):
        """
        Method to initialize the RVO2 simulation with the provided configuration.
        This method will convert Pydantic objects into RVO2-compatible objects.
        """
        config = self.world_config  # Access the world configuration.
        self.sim = rvo2_rl.RVOSimulator(
            config.time_step,
            config.agent_defaults.neighbor_dist,
            config.agent_defaults.max_neighbors,
            config.agent_defaults.time_horizon,
            config.agent_defaults.time_horizon_obst,
            config.agent_defaults.radius,
            config.agent_defaults.max_speed
        )

        # Add agents and save their goals
        # Initialize the global agent_id counter
        global_agent_id = 0
        agent_behaviours = {}
        # pprint.pprint(config.agents, indent=4)
        # Iterate over each group of agents
        for agent_group in config.agents:
            positions = agent_group.pattern.generate_positions()

            # Generate goal positions for this group if they exist
            goals = agent_group.goals.pattern.generate_positions() if agent_group.goals else None

            # Iterate over the generated agent positions
            for local_agent_index, position in enumerate(positions):
                agent_defaults = agent_group.agent_defaults or config.agent_defaults

                # Add the agent to the rvo2 simulation and get its global ID
                agent_id = self.sim.add_agent(
                    rvo2_rl.Vector2(*position),
                    agent_defaults.neighbor_dist,
                    agent_defaults.max_neighbors,
                    agent_defaults.time_horizon,
                    agent_defaults.time_horizon_obst,
                    agent_defaults.radius,
                    agent_defaults.max_speed,
                    rvo2_rl.Vector2(*agent_defaults.velocity)
                )

                # Set the agent's preferred velocity
                self.sim.set_agent_pref_velocity(
                    agent_id, rvo2_rl.Vector2(*agent_defaults.velocity))
                self.agent_initial_positions.append(position)
                # If goals are defined for the agent group
                if goals:
                    # Assign the correct goal to the agent using the local index
                    self.agent_goals[agent_id] = goals[local_agent_index]
                    self.notify_observers(GoalsProcessedMessage(
                        step=-1, goals=self.agent_goals))

                if agent_group.assigned_behaviors:
                    final_behavior_name = agent_group.assigned_behaviors[local_agent_index]
                    agent_behaviours[agent_id] = final_behavior_name
                    self.update_agent_with_behavior_params(
                        agent_id, final_behavior_name)
                else:
                    final_behavior_name = None
                # Increment the global agent ID for the next agent
                global_agent_id += 1

        goals_v2 = [rvo2_rl.Vector2(goal[0], goal[1])
                    for _, goal in self.agent_goals.items()]
        self.sim.set_goals_list(goals_v2)
        # Add obstacles to the simulation
        if config.obstacles:
            obstacle_shapes = []
            for obstacle_shape in config.obstacles:
                shape = obstacle_shape.generate_shape()
                self.sim.add_obstacle(shape)
                obstacle_shapes.append(shape)
            self.sim.process_obstacle()
            self.notify_observers(ObstaclesProcessedMessage(
                step=-1, obstacles=obstacle_shapes))

        agent_initialization_data = [
            {
                "agent_id": agent_id,
                "radius": self.sim.get_agent_radius(agent_id),
                "max_speed": self.sim.get_agent_max_speed(agent_id),
                "neighbor_dist": self.sim.get_agent_neighbor_dist(agent_id),
                "max_neighbors": self.sim.get_agent_max_neighbors(agent_id),
                "time_horizon": self.sim.get_agent_time_horizon(agent_id),
                "time_horizon_obst": self.sim.get_agent_time_horizon_obst(agent_id),
                "goal": self.agent_goals[agent_id],
                # Agent behavior
                "behaviour": agent_behaviours.get(agent_id)
            }
            for agent_id in range(self.sim.get_num_agents())
        ]

        # Send initialization information to observers
        self.notify_observers(SimulationInitializedMessage(
            step=-1, agent_initialization_data=agent_initialization_data))
        self._setup_obstacle_vertex_array()
        self.initial_distance_from_goal_array = [self.distance_from_goal(
            agent_id) for agent_id in range(self.sim.get_num_agents())]

    def update_agent_with_behavior_params(self, agent_id: int, behavior_name: str):
        if behavior_name:
            behavior = global_registry.get('behaviour', behavior_name)
            agent_defaults = behavior.get_agent_params()
            self.sim.set_agent_max_speed(agent_id, agent_defaults.max_speed)
            self.sim.set_agent_radius(agent_id, agent_defaults.radius)
            self.sim.set_agent_time_horizon(
                agent_id, agent_defaults.time_horizon)
            self.sim.set_agent_time_horizon_obst(
                agent_id, agent_defaults.time_horizon_obst)
            self.sim.set_agent_max_neighbors(
                agent_id, agent_defaults.max_neighbors)
            self.sim.set_agent_neighbor_dist(
                agent_id, agent_defaults.neighbor_dist)
            self.sim.set_agent_velocity(
                agent_id, rvo2_rl.Vector2(*agent_defaults.velocity))
            # print(f"Agent {agent_id} updated with {behavior_name} parameters")

    def _setup_obstacle_vertex_array(self):
        simulator = self.sim
        """
        Precomputes all obstacle segments and stores them in a NumPy array for efficient queries.
        
        Parameters:
        - simulator: Instance of RVOSimulator (Python bindings).

        Returns:
        - NumPy array of shape (N, 2, 2) where N is the number of segments.
        Each segment is stored as [[Ax, Ay], [Bx, By]].
        """
        num_obstacles = simulator.get_num_obstacle_vertices()
        static_segments = np.zeros(
            (num_obstacles, 2, 2), dtype=np.float32)  # Shape (N, 2, 2)

        for i in range(num_obstacles):
            A = np.array(simulator.get_obstacle_vertex(
                i), dtype=np.float32)  # First vertex
            B = np.array(simulator.get_obstacle_vertex(
                simulator.get_next_obstacle_vertex_no(i)), dtype=np.float32)  # Next vertex
            static_segments[i] = [A, B]  # Store segment

        self._obstacle_segment_np_array = static_segments
        # print(self._obstacle_segment_np_array)

    def get_obstacle_vertex_array(self):
        return self._obstacle_segment_np_array

    def vector_360_ray_intersections(self, agent_id):
        """
        A simple, non-vectorized approach to raycasting in 360 degrees,
        testing all static_segments, ignoring neighbor indices.
        No max detection radius – all distances are valid.

        Returns a list of 360 intersection points [x, y] or [None, None].
        """

        simulator = self.sim
        # shape (N, 2, 2) for all obstacles
        static_segments = self.get_obstacle_vertex_array()
        agent_position = np.array(
            (simulator.get_agent_position(agent_id).x(), simulator.get_agent_position(agent_id).y()), dtype=np.float32)
        # Generate 360 directions
        angles = np.radians(np.arange(360), dtype=np.float32)
        directions = np.column_stack(
            (np.cos(angles), np.sin(angles))).astype(np.float32)
        final_result = []
        for i in range(360):
            direction = directions[i]
            # track smallest valid t across all segments
            best_t = float('inf')
            best_point = [9999, 9999]

            # For each obstacle segment in ALL segments
            for seg in static_segments:
                A = seg[0].astype(np.float32)
                B = seg[1].astype(np.float32)
                seg_dir = A - B  # direction of the segment

                # Build the system to solve for [t, u]:
                #   A + u * seg_dir = O + t * direction
                # => (A - O) = [direction, seg_dir] * [t, u]^T
                M = np.array([
                    [direction[0], seg_dir[0]],
                    [direction[1], seg_dir[1]]
                ], dtype=np.float32)

                rhs = A - agent_position  # (2,)

                det = np.linalg.det(M)
                if abs(det) < 1e-6:
                    # Parallel or nearly parallel
                    continue

                # Solve
                try:
                    sol = np.linalg.solve(M, rhs)  # returns [t, u]
                    t, u = sol[0], sol[1]
                except np.linalg.LinAlgError:
                    continue  # numerical fail

                # Must be in front of the ray (t >= 0) and on the segment (0 <= u <= 1)
                if t >= 0 and 0 <= u <= 1:
                    # No maximum radius filter – any t is valid
                    if t < best_t:
                        best_t = t
                        # Intersection = O + t * direction
                        x = agent_position[0] + direction[0] * t
                        y = agent_position[1] + direction[1] * t
                        best_point = [float(x), float(y)]

            final_result.append(best_point)

        self.intersect_list = final_result
        return final_result

    def reset(self):
        """Resets the simulation to its initial state."""
        self.current_step = 0
        self._state = SimulationState.SETUP

        # Reset agent positions to initial positions
        for agent_id, initial_position in enumerate(self.agent_initial_positions):
            self.sim.set_agent_position(
                agent_id, rvo2_rl.Vector2(initial_position[0], initial_position[1]))

        # Reset any other relevant variables (such as goals)
        # Optional: If you need to reset goals, you can do it here
        for agent_id in self.agent_goals:
            # If you need to update goals you can do it here, otherwise just reinitialize the simulation
            self.set_goal(agent_id, self.agent_goals[agent_id])

    def step(self):
        """
        Executes the simulation for a specified number of steps.

        Args:
            steps (int): Number of steps the simulation should execute.
        """
        self.update_agent_velocities()
        self.sim.do_step()

        # Detect if any agent has reached its goal
        for agent_id in range(self.sim.get_num_agents()):
            if self.is_goal_reached(agent_id):
                event = GoalReachedEvent(
                    agent_id=agent_id,
                    goal_position=self.agent_goals[agent_id],
                    current_position=(self.sim.get_agent_position(
                        agent_id).x(), self.sim.get_agent_position(agent_id).y()),
                    step=self.current_step
                )
                self.handle_event(event.alias, event)

        # Collect more data from each agent
        agent_data = [
            (
                agent_id,
                # Current position (x, y)
                *[self.sim.get_agent_position(agent_id).x(),
                  self.sim.get_agent_position(agent_id).y()],
                self.sim.get_agent_velocity(agent_id),  # Current velocity
                self.sim.get_agent_pref_velocity(
                    agent_id),  # Preferred velocity
                math.dist([self.sim.get_agent_position(agent_id).x(),
                           self.sim.get_agent_position(agent_id).y()],
                          self.agent_goals[agent_id])  # Distance to goal
            )
            for agent_id in range(self.sim.get_num_agents())
        ]

        # Send the message with additional data
        self.notify_observers(AgentPositionsUpdateMessage(
            step=self.current_step, agent_positions=agent_data))
        if self.intersect_list != None:
            self.notify_observers(RayCastingUpdateMessage(
                step=self.current_step, intersections=self.intersect_list))
        # self.store_step(self.current_step)

    def run_simulation(self, steps: int):
        """
        Executes the simulation for a specified number of steps.

        Args:
            steps (int): Number of steps the simulation should execute.
        """
        for step in range(steps):
            self.update_agent_velocities()
            self.sim.doStep()

            # Detect if any agent has reached its goal
            for agent_id in range(self.sim.get_num_agents()):
                if self.is_goal_reached(agent_id):
                    event = GoalReachedEvent(
                        agent_id=agent_id,
                        goal_position=self.agent_goals[agent_id],
                        current_position=self.sim.get_agent_position(agent_id),
                        step=step
                    )
                    self.handle_event(event.alias, event)

            agent_positions = [(agent_id, *self.sim.get_agent_position(agent_id))
                               for agent_id in range(self.sim.get_num_agents())]
            print(
                f"Sending AgentPositionsUpdateMessage for step {self.current_step}")
            self.notify_observers(AgentPositionsUpdateMessage(
                step=self.current_step, agent_positions=agent_positions))
            # self.store_step(step)

    def update_agent_velocity(self, agent_id: int, velocity: Tuple[float, float]):
        """
        Registers a manual velocity update for a specific agent.
        """
        self._manual_velocity_updates.append((agent_id, velocity))

    def get_agent_min_speed(self, agent_id: int) -> float:
        """
        Returns the minimum speed of the agent with the specified ID.
        """
        return 0.0

    def get_agent_max_speed(self, agent_id: int) -> float:
        """
        Returns the maximum speed of the agent with the specified ID.
        """
        return self.sim.get_agent_max_speed(agent_id)

    def get_velocity_min_euclid_dist(self, agent_id: int) -> Tuple[float, float]:
        """
        Returns the velocity that minimizes the Euclidean distance to the goal, clipped to the agent's maximum speed.
        """
        agent_position = self.sim.get_agent_position(agent_id)
        goal_position = self.agent_goals[agent_id]
        max_speed = self.sim.get_agent_max_speed(agent_id)

        vector_to_goal = (
            goal_position[0] - agent_position.x(),
            goal_position[1] - agent_position.y()
        )
        distance = math.sqrt(vector_to_goal[0] ** 2 + vector_to_goal[1] ** 2)

        if distance > 0:
            preferred_velocity = (
                vector_to_goal[0] / distance * max_speed,
                vector_to_goal[1] / distance * max_speed
            )
        else:
            preferred_velocity = (0, 0)

        return preferred_velocity

    def update_agent_velocities(self):
        """
        Updates the preferred velocities of agents in the simulation, considering manual updates.
        """
        self.sim.set_preferred_velocities()
        # Apply manual updates first
        manual_update_ids = set(agent_id for agent_id,
                                _ in self._manual_velocity_updates)
        # print("Manual updates:", self._manual_velocity_updates)
        for agent_id, velocity in self._manual_velocity_updates:
            self.sim.set_agent_pref_velocity(
                agent_id, rvo2_rl.Vector2(*velocity))
        # Clear the queue after applying updates
        self._manual_velocity_updates.clear()

    def clear_buffer(self):
        self.steps_buffer = []

    def get_agent_max_num_neighbors(self, agent_id):
        return self.sim.get_agent_max_neighbors(agent_id)

    def get_neighbors_data(self, agent_id):
        neighbor_count = self.sim.get_agent_num_agent_neighbors(agent_id)
        neighbors_data = []
        for i in range(neighbor_count):
            neighbor_agent_id = self.sim.get_agent_agent_neighbor(agent_id, i)
            neighbor_position = self.sim.get_agent_position(neighbor_agent_id)
            neighbor_velocity = self.sim.get_agent_velocity(neighbor_agent_id)
            neighbor_pref_velocity = self.sim.get_agent_pref_velocity(
                neighbor_agent_id)
            neighbors_data.append(neighbor_position.x())
            neighbors_data.append(neighbor_position.y())
            neighbors_data.append(neighbor_velocity.x())
            neighbors_data.append(neighbor_velocity.y())
            neighbors_data.append(neighbor_pref_velocity.x())
            neighbors_data.append(neighbor_pref_velocity.y())
        return neighbors_data

    def get_agent_position(self, agent_id) -> Tuple[float, float]:
        """Returns the current position of the agent."""
        return self.sim.get_agent_position(agent_id)

    def get_agent_positions(self) -> Dict[int, Tuple[float, float]]:
        """
        Returns the current positions of all agents in the simulation.

        Returns:
            Dict[int, Tuple[float, float]]: A dictionary where keys are agent IDs and values are positions (x, y).
        """
        agent_positions = {}
        for agent_id in range(self.sim.get_num_agents()):
            position = self.sim.get_agent_position(agent_id)
            agent_positions[agent_id] = position
        return agent_positions

    def set_goal(self, agent_id: int, goal: Tuple[float, float]) -> None:
        """Adds or updates the goal of the agent given its ID."""
        self.agent_goals[agent_id] = goal

    def get_goal(self, agent_id: int) -> Tuple[float, float]:
        """
        Returns the current goal of an agent given its ID.

        Args:
            agent_id (int): The ID of the agent.

        Returns:
            Tuple[float, float]: The position of the agent's goal.
        """
        return self.agent_goals.get(agent_id)

    def is_goal_reached(self, agent_id: int) -> bool:
        """
        Checks if an agent has reached its goal.

        Args:
            agent_id (int): The ID of the agent.

        Returns:
            bool: True if the agent has reached its goal, False otherwise.
        """
        distance = self.distance_from_goal(agent_id)
        # Consider the goal reached if the distance is less than or equal to a threshold
        return distance <= 0.50

    def distance_from_goal(self, agent_id):
        current_position = self.sim.get_agent_position(agent_id)
        goal_position = self.get_goal(agent_id)
        if not goal_position:
            return False

        distance = math.sqrt(
            (current_position.x() - goal_position[0]) ** 2 +
            (current_position.y() - goal_position[1]) ** 2
        )
        # Consider the goal reached if the distance is less than or equal to a threshold
        return distance


def main():
    parser = argparse.ArgumentParser(
        description='Simulador de Navegación de Agentes')
    parser.add_argument('world_file', type=str,
                        help='Archivo YAML de configuración del mundo')
    parser.add_argument('--renderer', type=str, choices=['pygame', 'text'], default='pygame',
                        help='El tipo de renderer a usar: pygame o text (por defecto: pygame)')
    args = parser.parse_args()

    world_file = args.world_file
    try:
        with open(world_file, 'r') as stream:
            data = yaml.safe_load(stream)
            world_config = Simulation(**data['simulation'])

    except FileNotFoundError:
        print(f"File {world_file} not found.")
        sys.exit(1)
    except yaml.YAMLError as exc:
        print(f"Error reading YAML file: {exc}")
        sys.exit(1)
    except ValidationError as exc:
        print(f"Validation error: {exc}")
        sys.exit(1)

    window_width = int((world_config.map_settings.x_max -
                       world_config.map_settings.x_min) * world_config.map_settings.cell_size)
    window_height = int((world_config.map_settings.y_max -
                        world_config.map_settings.y_min) * world_config.map_settings.cell_size)

    if args.renderer == 'pygame':
        renderer = PyGameRenderer(
            window_width,
            window_height,
            obstacles=[], goals={}, cell_size=int(world_config.map_settings.cell_size)
        )
        renderer.setup()
    else:
        renderer = TextRenderer()
        renderer.setup()

    rvo2_simulator = RVO2SimulatorWrapper(world_config, "test_simulation")
    rvo2_simulator.register_observer(renderer)

    for dynamic_config in world_config.dynamics:
        rvo2_simulator.register_dynamic(dynamic_config)

    rvo2_simulator.run_pipeline(5000)  # Se asume 5000 pasos como ejemplo
    rvo2_simulator.save_simulation_runs()


if __name__ == "__main__":
    main()
