import numpy as np
from gymnasium import logger
from simulator.engines.base import SimulationState
from rl_environments.single_agent.miac.miac_base import RVOBaseEnv

class RVOMiacPerp2V2(RVOBaseEnv):
    """
    Version 2 of the RVOMiacPerp2 environment, using RVOBaseEnv.
    Inherits simulator setup, rendering, seeding, and default step logic.
    """

    def __init__(self, config_file=None, render_mode="rgb", seed=None, step_mode='min_dist'):
        super().__init__(config_file=config_file, render_mode=render_mode, seed=seed, step_mode=step_mode)

    def _get_obs(self):
        """
        Observations for agent 0, including neighbor data padding.
        """
        pos = self.sim.get_agent_position(0)
        goal = self.sim.get_goal(0)
        max_neigh = self.sim.get_agent_max_num_neighbors(0)
        neighbors = self.sim.get_neighbors_data(0)
        expected_len = max_neigh * 6

        # Pad or truncate neighbor data
        if len(neighbors) < expected_len:
            neighbors.extend([-9999]*(expected_len - len(neighbors)))
        else:
            neighbors = neighbors[:expected_len]

        obs = [goal[0] - pos[0], goal[1] - pos[1]]
        obs.extend(neighbors)
        return np.array(obs, dtype=np.float32)

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
    env = RVOMiacPerp2V2(
        config_file='./simulator/worlds/miac/perp2/perp2_level_2.yaml',
        render_mode='rgb',
        seed=42,
        step_mode='min_dist'
    )
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Random actions
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            logger.info(f"Episode done: {done}, truncated: {truncated}")
            break