import numpy as np
from gymnasium import logger
from simulator.engines.base import SimulationState
from rl_environments.single_agent.miac.miac_base import RVOBaseEnv

class RVOMiacCircle(RVOBaseEnv):
    """
    Version 2 of the RVOMiacCircle environment, using RVOBaseEnv.
    Inherits the base simulator setup, rendering, seeding, and step logic.
    """

    def __init__(self, config_file=None, render_mode="rgb", seed=None, step_mode='min_dist'):
        super().__init__(config_file=config_file, render_mode=render_mode, seed=seed, step_mode=step_mode)
        self.sim.set_agent_defaults(0, self.world_config.agent_defaults)

    def _get_obs(self):
        """
        Gets the observation for agent 0, including neighbor data padding.
        """
        pos = self.sim.get_agent_position(0)
        goal = self.sim.get_goal(0)

        max_neigh = self.sim.get_agent_max_num_neighbors(0)
        agent_neighbors = self.sim.get_neighbors_data(0)
        expected_len = max_neigh * 6

        # Pad or truncate neighbor data
        if len(agent_neighbors) < expected_len:
            agent_neighbors.extend([-9999] * (expected_len - len(agent_neighbors)))
        else:
            agent_neighbors = agent_neighbors[:expected_len]

        # Observation: (goal - position) + neighbor data
        obs = [goal[0] - pos[0], goal[1] - pos[1]]
        obs.extend(agent_neighbors)
        return np.array(obs, dtype=np.float32)

    def calculate_reward(self, agent_id=0):
        """
        Example reward function: -10 per step, +10000 on goal reach.
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
        Returns extra info if needed.
        """
        return {}

if __name__ == "__main__":
    env = RVOMiacCircle(
        config_file='./simulator/worlds/miac/circle/circle_level_2.yaml',
        render_mode='rgb',
        seed=42,
        step_mode='min_dist'
    )
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            logger.info(f"Episode done: {done}, truncated: {truncated}")
            break
