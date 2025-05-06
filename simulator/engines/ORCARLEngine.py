from typing import Dict, Tuple
import rvo2_rl
from pydantic import BaseModel
import math
from simulator.engines.base import SimulationEngine, SimulationState
from simulator.models.simulation_configuration.registry import global_registry
from rvo2_rl.rl import RVO2RLWrapper, ObsMode
from rvo2_rl.rvo2 import Vector2


class ORCARLEngine(SimulationEngine):
    def __init__(self, world_config: BaseModel, simulation_id: str, seed: int = None):
        SimulationEngine.__init__(self, seed=seed)
        """
        Initializes the RVO2 simulator with the world configuration and an optional renderer.

        Args:
            world_config (BaseModel): The world configuration in Pydantic format.
            simulation_id (str): The ID of the current simulation.
        """
        self.world_config = world_config  # Stores the provided world configuration
        self.simulation_id = simulation_id  # Stores the simulation ID
        self.sim = None  # Instance of the RVO2 simulator, will be initialized later
        self.wrapper = None
        self._manual_velocity_updates = []
        self.goal_reached_threshhold = 0.02
        self.normalized = True
        self.agent_goals = {}

    def set_agent_defaults(self, agent_idx, agent_defaults):
        self.sim.set_agent_neighbor_dist(
            agent_idx, agent_defaults.neighbor_dist)
        self.sim.set_agent_max_neighbors(
            agent_idx, agent_defaults.max_neighbors)
        self.sim.set_agent_time_horizon(agent_idx, agent_defaults.time_horizon)
        self.sim.set_agent_time_horizon_obst(
            agent_idx, agent_defaults.time_horizon_obst)
        self.sim.set_agent_radius(agent_idx, agent_defaults.radius)
        self.sim.set_agent_max_speed(agent_idx, agent_defaults.max_speed)
        self.sim.set_agent_velocity(agent_idx, Vector2(
            agent_defaults.velocity[0], agent_defaults.velocity[1]))

    def initialize_simulation(self):
        """
        Method to initialize the RVO2 simulation with the provided configuration.
        This method will convert Pydantic objects into RVO2-compatible objects.
        """
        config = self.world_config  # Access the world configuration.
        self.wrapper = RVO2RLWrapper(
            time_step=config.time_step,
            neighbor_dist=config.agent_defaults.neighbor_dist,
            max_neighbors=config.agent_defaults.max_neighbors,
            time_horizon=config.agent_defaults.time_horizon,
            time_horizon_obst=config.agent_defaults.time_horizon_obst,
            radius=config.agent_defaults.radius,
            max_speed=config.agent_defaults.max_speed,
            mode=ObsMode.Cartesian,
            use_obs_mask=False,
            use_lidar=False
        )
        self.sim = self.wrapper.get_simulator()        

        # Add agents and save their goals
        # Initialize the global agent_id counter
        global_agent_id = 0
        agent_behaviours = {}
        # Iterate over each group of agents
        for agent_group in config.agents:
            positions = agent_group.pattern.generate_positions()
            # Generate goal positions for this group if they exist
            goals = agent_group.goals.pattern.generate_positions() if agent_group.goals else None
            # Iterate over the generated agent positions
            for local_agent_index, position in enumerate(positions):
                agent_defaults = agent_group.agent_defaults or config.agent_defaults

                # Add the agent to the rvo2 simulation and get its global ID
                agent_id = self.wrapper.add_agent(
                    Vector2(*position),
                    agent_defaults.neighbor_dist,
                    agent_defaults.max_neighbors,
                    agent_defaults.time_horizon,
                    agent_defaults.time_horizon_obst,
                    agent_defaults.radius,
                    agent_defaults.max_speed,
                    Vector2(*agent_defaults.velocity)
                )

                # Set the agent's preferred velocity
                self.wrapper.get_simulator().set_agent_pref_velocity(
                    agent_id, Vector2(*agent_defaults.velocity))
                if goals:
                    # Assign the correct goal to the agent using the local index
                    self.agent_goals[agent_id] = goals[local_agent_index]

                if agent_group.assigned_behaviors:
                    final_behavior_name = agent_group.assigned_behaviors[local_agent_index]
                    agent_behaviours[agent_id] = final_behavior_name
                    self.update_agent_with_behavior_params(
                        agent_id, final_behavior_name)
                    self.wrapper.set_agent_behavior(
                        agent_id, final_behavior_name)
                else:
                    final_behavior_name = None
                # Increment the global agent ID for the next agent
                global_agent_id += 1

        goals_v2 = [Vector2(goal[0], goal[1])
                    for _, goal in self.agent_goals.items()]
        self.wrapper.set_goals(goals_v2)
        # Add obstacles to the simulation
        if config.obstacles:
            for obstacle_shape in config.obstacles:
                vertices = obstacle_shape.generate_shape()
                shape = [Vector2(vertex[0], vertex[1])
                         for vertex in vertices]
                self.sim.add_obstacle(shape)
            self.sim.process_obstacles()

        self.wrapper.initialize()
        
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
        self.agent_initialization_data = agent_initialization_data

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
                agent_id, Vector2(*agent_defaults.velocity))
            # print(f"Agent {agent_id} updated with {behavior_name} parameters")

    def get_obs(self, agent_id):
        return self.wrapper.get_observation(agent_id)

    def get_obs_limits(self):
        return self.wrapper.get_observation_limits()

    def get_agent_for_vis(self):
        return self.wrapper.get_agent_data_for_vis()

    def reset(self):
        """Resets the simulation to its initial state."""
        print("RESET")
        
        self._state = SimulationState.SETUP

        self.wrapper.reset_position_and_goals_to_init()
        self.wrapper.reset_step_count()       

        self.update_agent_velocities()

    def get_step_count(self):
        return self.wrapper.get_step_count()
    
    def step(self):
        """
        Executes the simulation for a specified number of steps.

        Args:
            steps (int): Number of steps the simulation should execute.
        """
        self.update_agent_velocities()        
        self.wrapper.do_step()

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
        return self.wrapper.get_simulator().get_agent_max_speed(agent_id)

    def get_collision_free_velocity(self, agent_id: int) -> Tuple[float, float]:
        return self.wrapper.get_simulator().get_agent_velocity(agent_id)

    def update_agent_velocities(self):
        """
        Updates the preferred velocities of agents in the simulation, considering manual updates.
        """
        self.wrapper.set_preferred_velocities()
        # print("Manual updates:", self._manual_velocity_updates)
        for agent_id, velocity in self._manual_velocity_updates:
            # print("manual update. id: ", agent_id, " velocity: ", velocity)
            self.wrapper.set_preferred_velocity(
                agent_id, Vector2(*velocity))
        # Clear the queue after applying updates
        self._manual_velocity_updates.clear()

    def get_agent_max_num_neighbors(self, agent_id):
        return self.wrapper.get_simulator().get_agent_max_neighbors(agent_id)

    def get_agent_position(self, agent_id) -> Tuple[float, float]:
        """Returns the current position of the agent."""
        return self.wrapper.get_simulator().get_agent_position(agent_id)

    def get_agent_positions(self) -> Dict[int, Tuple[float, float]]:
        pass

    def set_goal(self, agent_id: int, goal: Tuple[float, float]) -> None:
        """Adds or updates the goal of the agent given its ID."""
        self.wrapper.set_goal(agent_id, goal)

    def get_goal(self, agent_id: int) -> Tuple[float, float]:
        """
        Returns the current goal of an agent given its ID.

        Args:
            agent_id (int): The ID of the agent.

        Returns:
            Tuple[float, float]: The position of the agent's goal.
        """
        return self.wrapper.get_goal(agent_id)

    def is_goal_reached(self, agent_id: int) -> bool:
        """
        Checks if an agent has reached its goal.

        Args:
            agent_id (int): The ID of the agent.

        Returns:
            bool: True if the agent has reached its goal, False otherwise.
        """        
        distance = self.wrapper.get_distance_to_goal(agent_id, self.normalized)        
        # Consider the goal reached if the distance is less than or equal to a threshold
        return distance <= self.goal_reached_threshhold
    
    def get_distance_to_goal(self, agent_id, normalized):
        return self.wrapper.get_distance_to_goal(agent_id, normalized)
        

    def get_all_distances_from_goals(self):
        return self.wrapper.get_all_distances_to_goals()
