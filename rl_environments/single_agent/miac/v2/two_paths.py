from datetime import datetime
import os
import numpy as np
from gymnasium import logger
from rl_environments.single_agent.miac.v2.miac_base2 import RVOBaseEnv2
from rvo2_rl.rl import ObsMode


class RVOMiacTwoPathsV2(RVOBaseEnv2):
    """
    Refactored version of RVOMiacTwoPaths that inherits from RVOBaseEnv.
    Uses the base class's step() logic with either 'naive' or 'min_dist' mode.
    """

    def __init__(self, config_file=None, render_mode="rgb_array", seed=None, step_mode='min_dist', use_lidar=False, use_obs_mask=False, mode=ObsMode.Cartesian):
        super().__init__(config_file=config_file,
                         render_mode=render_mode, seed=seed, step_mode=step_mode, use_lidar=use_lidar, use_obs_mask=use_obs_mask, mode=mode)

    def _get_obs(self):
        """
        Gets the observation for the agent (ID=0), with padding for neighbor data.
        By default, we expect 2 values for (goal-pos), plus neighbor data.
        Optimized to work directly with numpy arrays and minimize copying.
        """
        observation = self.engine.get_obs(0)
        return observation

    def is_done(self, agent_id=0):
        """
        Determines if the agent has reached its goal.
        """
        return self.engine.is_goal_reached(agent_id)

    def _get_info(self):
        return {
            "success": int(self.is_done(self.intelligent_agent_id))
        }


if __name__ == "__main__":
    from gymnasium.wrappers import RecordVideo
    env = RVOMiacTwoPathsV2(
        config_file='./simulator/worlds/miac/two_paths/two_paths_level_2.yaml',
        render_mode='rgb_array',
        seed=42,
        step_mode='min_dist',
        use_lidar=True,
        use_obs_mask=True,
        mode=ObsMode.Polar
    )
    # Extract filename without extension from config_file path
    config_filename = os.path.basename(
        env.config_file)  # Gets 'two_paths_level_0.yaml'
    config_name = os.path.splitext(config_filename)[
        0]  # Gets 'two_paths_level_0'

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
    i = 0
    while not done:
        action = env.action_space.sample()  # Random actions
        obs, reward, done, truncated, info = env.step(action)
        # env.render()
        if done or truncated:
            # logger.info(f"Episode done: {done}, truncated: {truncated}")
            break
        i += 1
    env.close()
