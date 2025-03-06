#!/usr/bin/env python
import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces, logger

from rendering.pygame_renderer import PyGameRenderer
from simulator.engines.RVO2SimulatorWrapper import RVO2SimulatorWrapper
from simulator.engines.base import SimulationState
from simulator.models.simulation import Simulation as SimulationModel


class RVOMiacCircle(gym.Env):
    metadata = {'render.modes': ['ansi', 'rgb']}

    def __init__(self, config_file=None, render_mode="rgb", seed=None):
        super(RVOMiacCircle, self).__init__()
        # Load YAML configuration
        with open(config_file, 'r') as stream:
            config_data = yaml.safe_load(stream)

        # Initialize the simulator with RVO2SimulatorWrapper, passing the seed
        world_config = SimulationModel(**config_data['simulation'])   
        print(world_config.agents)     
        dynamics = world_config.dynamics
        self.sim = RVO2SimulatorWrapper(world_config, "test_simulation", seed=seed)
        for dynamic_config in dynamics:
            self.sim.register_dynamic(dynamic_config)

        # Set render mode
        self.render_mode = render_mode
        if render_mode == "rgb":
            window_width = int((
                self.sim.world_config.map_settings.x_max - self.sim.world_config.map_settings.x_min) * self.sim.world_config.map_settings.cell_size)
            window_height = int((
                self.sim.world_config.map_settings.y_max - self.sim.world_config.map_settings.y_min) * self.sim.world_config.map_settings.cell_size)

            renderer = PyGameRenderer(
                window_width,
                window_height,
                obstacles=[],
                goals={},
                cell_size=int(self.sim.world_config.map_settings.cell_size)
            )
            renderer.setup()
            self.sim.register_observer(renderer)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(92,), dtype=np.float32)
        self.sim.initialize_simulation()
        self.sim.set_agent_defaults(0, world_config.agent_defaults)

    def _get_obs(self):
        """Gets the observation for agent 0 with padding for neighbor data."""
        # Get agent position and goal
        pos = self.sim.get_agent_position(0)
        goal = self.sim.get_goal(0)

        # Get the maximum number of allowed neighbors and the current neighbor data
        max_neigh = self.sim.get_agent_max_num_neighbors(0)
        agent_neighbors = self.sim.get_neighbors_data(0)  # Assume it returns a list of floats

        # Expected number of elements for neighbor data
        expected_neighbors_data_length = max_neigh * 6

        # If there are fewer neighbor data than expected, pad with -9999
        if len(agent_neighbors) < expected_neighbors_data_length:
            padding_length = expected_neighbors_data_length - len(agent_neighbors)
            agent_neighbors.extend([-9999] * padding_length)
        else:
            # If there are more data (may not be necessary), truncate
            agent_neighbors = agent_neighbors[:expected_neighbors_data_length]

        # Calculate the initial observation (difference between position and goal)
        observations = [goal[0] - pos[0], goal[1] - pos[1]]

        # Concatenate the initial observations with the neighbor data (already padded)
        observations.extend(agent_neighbors)
        # Convert the observation to a numpy array and return it
        return np.array(observations, dtype=np.float32)

    def step(self, action):
        """Performs a step in the simulation with the action as deviation."""
        velocity = self.sim.get_velocity_min_euclid_dist(0)
        # Interpret the action as a deviation
        deviation = np.array(action)
        velocity = velocity + deviation
        # Get the maximum and minimum allowed magnitude
        min_magnitude = self.sim.get_agent_min_speed(0)
        max_magnitude = self.sim.get_agent_max_speed(0)

        # Calculate the current magnitude of the velocity
        velocity_magnitude = np.linalg.norm(velocity)

        # If the magnitude is out of bounds, adjust
        if velocity_magnitude < min_magnitude:
            clipped_velocity = (velocity / velocity_magnitude) * min_magnitude
        elif velocity_magnitude > max_magnitude:
            clipped_velocity = (velocity / velocity_magnitude) * max_magnitude
        else:
            clipped_velocity = velocity  # Keep unchanged if within range
        self.sim.update_agent_velocity(0, tuple(clipped_velocity))
        self.sim.update_agent_velocities()
        self.sim.execute_simulation_step()
        self.sim.current_step += 1

        # Get new observation
        observations = self._get_obs()

        # Calculate reward
        reward = self.calculate_reward(0)

        # Check if the episode has ended
        done = self.is_done(0)
        truncated = self.sim.get_state() == SimulationState.STOPPED
        info = self._get_info()

        # Render if necessary
        if self.render_mode:
            self.render()

        return observations, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment and the simulation."""
        if seed is not None:
            self.sim.reset_rng_with_seed(seed)  # Update the seed if provided
        self.sim.reset()
        return self._get_obs(), self._get_info()

    def render(self):
        """Renders the current state of the simulation."""
        if self.render_mode is None:
            logger.warn("You are calling render method without specifying any render mode.")
        else:
            pass
            # self.sim.render()  # Delegated to the simulator

    def calculate_reward(self, agent_id):
        """Calculates the reward for agent 0."""
        reward = -10
        if self.is_done(agent_id):
            reward += 10000
        return reward

    def is_done(self, agent_id):
        """Determines if the agent has reached its goal."""
        return self.sim.is_goal_reached(agent_id)

    def _get_info(self):
        """Returns additional information about the environment."""
        return {}


if __name__ == "__main__":
    env = RVOMiacCircle('./simulator/worlds/miac/circle/circle_level_1.yaml', render_mode='rgb')
    observations = env.reset()
    done = False
    i = 0
    while not done:
        action = env.action_space.sample()  # Take random actions
        observations, reward, done, truncated, info = env.step(action)

        # Monitor the agent's position
        agent_position = env.sim.get_agent_position(0)
        if done or truncated:
            print(f"Episode done: {done}, truncated: {truncated}")
            break
        i += 1
