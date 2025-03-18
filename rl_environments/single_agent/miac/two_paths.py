import numpy as np
from gymnasium import logger
from simulator.engines.base import SimulationState
from rl_environments.single_agent.miac.miac_base import RVOBaseEnv

class RVOMiacTwoPaths(RVOBaseEnv):
    """
    Refactored version of RVOMiacTwoPaths that inherits from RVOBaseEnv.
    Uses the base class's step() logic with either 'naive' or 'min_dist' mode.
    """

    def __init__(self, config_file=None, render_mode="rgb", seed=None, step_mode='min_dist'):
        """
        :param config_file: Path to the YAML config file
        :param render_mode: 'rgb', 'ansi', or None
        :param seed: Optional seed
        :param step_mode: 'naive' or 'min_dist'
        """
        super().__init__(config_file=config_file, render_mode=render_mode, seed=seed, step_mode=step_mode)

    def _get_obs(self):
        """
        Gets the observation for the agent (ID=0), with padding for neighbor data.
        By default, we expect 2 values for (goal-pos), plus neighbor data.
        """
        pos = self.sim.get_agent_position(0)
        goal = self.sim.get_goal(0)
        max_neigh = self.sim.get_agent_max_num_neighbors(0)
        neighbor_data = self.sim.get_neighbors_data(0)  # Should return a list of floats
        ray_casting = self.sim.vector_360_ray_intersections(0)
        # ray_casting = self.sim.debug_360_ray_intersections_loop(0)
        self.ray_casting = ray_casting
        # print(self.ray_casting)
        
        flattened = [coord
             for row in ray_casting
             for coord in row]
               
        # Each neighbor might provide 6 values (distance, direction, etc.)
        expected_length = max_neigh * 6
        if len(neighbor_data) < expected_length:
            neighbor_data.extend([-9999]*(expected_length - len(neighbor_data)))
        else:
            neighbor_data = neighbor_data[:expected_length]

        # Observations begin with goal offset, then neighbor info
        observations = [goal[0] - pos[0], goal[1] - pos[1]]
        observations.extend(flattened)
        observations.extend(neighbor_data)

        return np.array(observations, dtype=np.float32)

    def calculate_reward(self, agent_id):
        """
        Example reward function: small negative reward per step, large reward on reaching goal.
        """
        reward = -10
        if self.is_done(agent_id):
            reward += 10000
        return reward

    def is_done(self, agent_id):
        """Check if the agent has reached its goal."""
        return self.sim.is_goal_reached(agent_id)

    def _get_info(self):
        """Customize if additional info is needed."""
        return {}

if __name__ == "__main__":
    env = RVOMiacTwoPaths(
        config_file='./simulator/worlds/miac/two_paths/two_paths_level_0.yaml',
        render_mode='rgb',
        seed=42,
        step_mode='min_dist'
    )
    obs, info = env.reset()
    done = False
    i = 0
    while not done:
        action = env.action_space.sample()  # Random actions
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            logger.info(f"Episode done: {done}, truncated: {truncated}")
            break
        i += 1
