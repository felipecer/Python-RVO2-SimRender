from datetime import datetime
import os
import numpy as np
from gymnasium import logger
from simulator.engines.base import SimulationState
from rl_environments.single_agent.miac.miac_base import RVOBaseEnv


class RVOMiacPerp2(RVOBaseEnv):
    """
    Version 2 of the RVOMiacPerp2 environment, using RVOBaseEnv.
    Inherits simulator setup, rendering, seeding, and default step logic.
    """

    def __init__(self, config_file=None, render_mode="rgb_array", seed=None, step_mode='min_dist'):
        super().__init__(config_file=config_file,
                         render_mode=render_mode, seed=seed, step_mode=step_mode)

    def _get_obs(self):
        """
        Gets the observation for the agent (ID=0), with padding for neighbor data.
        By default, we expect 2 values for (goal-pos), plus neighbor data.
        Optimized to work directly with numpy arrays and minimize copying.
        """
        pos = self.sim.get_agent_position(0)
        goal = self.sim.get_goal(0)
        neighbor_data = self.sim.get_neighbors_data2(0)  # already numpy array
        ray_casting = self.sim.get_lidar_reading(0)  # already numpy array

        # Store ray_casting for visualization purposes
        self.ray_casting = ray_casting.tolist()
        self.sim.intersect_list = self.ray_casting

        # Create the goal offset array
        goal_offset = np.array(
            [self.sim.current_step, goal[0] - pos.x(), goal[1] - pos.y()], dtype=np.float32)

        # Concatenate all parts of the observation
        return np.concatenate([goal_offset, ray_casting.ravel(), neighbor_data])

    def calculate_reward(self, agent_id=0):
        """
        Simple reward: -10 each step, +10000 if goal reached.
        """
        reward = -10
        if self.is_done(agent_id):
            reward += 10000
        return reward

    def is_done(self, agent_id=0):
        """
        Returns True if agent's goal is reached.
        """
        return self.sim.is_goal_reached(agent_id)

    def _get_info(self):
        """
        Returns extra environment info if needed.
        """
        return {}


if __name__ == "__main__":
    # from gymnasium.wrappers import RecordVideo
    env = RVOMiacPerp2(
        config_file='./simulator/worlds/miac/perp2/perp2_level_1.yaml',
        render_mode='rgb_array',
        seed=42,
        step_mode='min_dist'
    )
    # Extract filename without extension from config_file path
    config_filename = os.path.basename(env.config_file)
    config_name = os.path.splitext(config_filename)[
        0]  # Gets 'two_paths_level_0'

    # Generate a unique name_prefix with filename and datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name_prefix = f"{config_name}_{timestamp}"
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Random actions
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            logger.info(f"Episode done: {done}, truncated: {truncated}")
            break
