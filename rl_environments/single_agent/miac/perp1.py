from datetime import datetime
import os
import numpy as np
from gymnasium import logger
from simulator.engines.base import SimulationState
from rl_environments.single_agent.miac.miac_base import RVOBaseEnv


class RVOMiacPerp1(RVOBaseEnv):
    """
    Version 2 of the RVOMiacPerp1 environment, using RVOBaseEnv.
    Inherits the base simulator setup, rendering, seeding, and step logic.
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

        # Create the goal offset array
        goal_offset = np.array(
            [self.sim.current_step/256.0, goal[0] - pos.x(), goal[1] - pos.y()], dtype=np.float32)

        # Concatenate all parts of the observation
        return np.concatenate([goal_offset, neighbor_data[0], neighbor_data[1]])

    def calculate_reward(self, agent_id=0):
        """
        Example reward function: -10 per step, +10000 on reaching the goal.
        """
        reward = -10
        if self.is_done(agent_id):
            reward += 10000
        return reward

    def is_done(self, agent_id=0):
        """
        Returns True if the agent has reached its goal.
        """
        return self.sim.is_goal_reached(agent_id)

    def _get_info(self):
        """
        Returns extra info, if needed.
        """
        return {}


if __name__ == "__main__":
    from gymnasium.wrappers import RecordVideo
    env = RVOMiacPerp1(
        config_file='./simulator/worlds/miac/perp1/perp1_level_3.yaml',
        render_mode='rgb_array',
        seed=42,
        step_mode='min_dist'
    )
    # Extract filename without extension from config_file path
    config_filename = os.path.basename(
        env.config_file)  # Gets 'two_paths_level_0.yaml'
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
