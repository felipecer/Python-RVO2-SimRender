#!/usr/bin/env python
import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces, logger

from rendering.pygame_renderer import PyGameRenderer
from simulator.engines.RVO2SimulatorWrapper import RVO2SimulatorWrapper
from simulator.engines.base import SimulationState
from simulator.models.simulation import Simulation as SimulationModel


class RVOSimulationEnv(gym.Env):
    metadata = {'render.modes': ['ansi', 'rgb']}

    def __init__(self, config_file=None, render_mode="rgb", seed=None):
        super(RVOSimulationEnv, self).__init__()
        # Load YAML configuration
        with open(config_file, 'r') as stream:
            config_data = yaml.safe_load(stream)
        
        # pprint(config_data, indent=6)
        # Initialize the simulator with RVO2SimulatorWrapper, passing the seed
        world_config = SimulationModel(**config_data['simulation'])
        dynamics = world_config.dynamics
        self.sim = RVO2SimulatorWrapper(world_config, "test_simulation", seed=seed)
        for dynamic_config in dynamics:
            self.sim.register_dynamic(dynamic_config)

        # Set up renderers if rendering is required
        self.render_mode = render_mode
        if render_mode == "rgb":
            window_width = int((self.sim.world_config.map_settings.x_max - self.sim.world_config.map_settings.x_min) * self.sim.world_config.map_settings.cell_size)
            window_height = int((self.sim.world_config.map_settings.y_max - self.sim.world_config.map_settings.y_min) * self.sim.world_config.map_settings.cell_size)

            renderer = PyGameRenderer(
                window_width,
                window_height,
                obstacles=[], 
                goals={}, 
                cell_size=int(self.sim.world_config.map_settings.cell_size)
            )
            renderer.setup()
            self.sim.register_observer(renderer)
            # print(f"Registered observers: {self.sim._observers}")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.sim.initialize_simulation()
        

    def _get_obs(self):
        """Gets the observation for agent 0."""
        pos = self.sim.get_agent_position(0)
        goal = self.sim.get_goal(0)
        observations = [goal[0] - pos[0], goal[1] - pos[1]]
        return np.array(observations, dtype=np.float32)

    def step(self, action):
        """Takes a step in the simulation with the provided action."""
        action = tuple(action)
        # print(f"Action: {action}")
        self.sim.update_agent_velocity(0, action)
        self.sim.update_agent_velocities()
        self.sim.execute_simulation_step()
        self.sim.current_step += 1
        observations = self._get_obs()
        reward = self.calculate_reward(0)
        done = self.is_done(0)
        # truncated = self.sim.current_step >= self.sim.world_config.time_limit  # Use the time_limit from world_config
        truncated = self.sim.get_state() == SimulationState.STOPPED
        # print(f"Step {self.sim.current_step}: Done: {done}, Truncated: {truncated}")
        info = self._get_info()

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
    env = RVOSimulationEnv('./simulator/worlds/simple_v2.yaml', render_mode='rgb')
    observations = env.reset()
    done = False
    i = 0
    while not done:
        action = env.action_space.sample()  # Take random actions
        # print(f"Action: {action}")
        observations, reward, done, truncated, info = env.step(action)

        # Monitor the agent's position
        agent_position = env.sim.get_agent_position(0)
        # print(f"Step {i}: Agent position: {agent_position}")
        if done or truncated:
            print(f"Episode done: {done}, truncated: {truncated}")
            break
        # print(f"Step {i} reward: {reward}")
        i += 1
