from datetime import datetime
import os
import numpy as np
from gymnasium import logger
from simulator.engines.base import SimulationState
from rl_environments.single_agent.miac.miac_base import RVOBaseEnv

class RVOMiacCircle(RVOBaseEnv):
    """
    Version 2 of the RVOMiacCircle environment, using RVOBaseEnv.
    Inherits the base simulator setup, rendering, seeding, and step logic.
    """

    def __init__(self, config_file=None, render_mode="rgb_array", seed=None, step_mode='min_dist'):
        super().__init__(config_file=config_file, render_mode=render_mode, seed=seed, step_mode=step_mode)
        self.sim.set_agent_defaults(0, self.world_config.agent_defaults)

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
        # ray_casting = [(99999, 99999)] * 360
        # print(type(ray_casting))
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
    from gymnasium.wrappers import RecordVideo
    env = RVOMiacCircle(
        config_file='./simulator/worlds/miac/circle/circle_level_3.yaml',
        render_mode='rgb_array',
        seed=42,
        step_mode='min_dist'
    )
    # Extract filename without extension from config_file path
    config_filename = os.path.basename(env.config_file)  # Gets 'two_paths_level_0.yaml'
    config_name = os.path.splitext(config_filename)[0]  # Gets 'two_paths_level_0'
    
    # Generate a unique name_prefix with filename and datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name_prefix = f"{config_name}_{timestamp}"

    env = RecordVideo(
        env, 
        video_folder="videos/",
        name_prefix=name_prefix,
    )
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            logger.info(f"Episode done: {done}, truncated: {truncated}")
            break
