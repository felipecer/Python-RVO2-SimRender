from pprint import pprint
import gymnasium as gym
import yaml
import numpy as np
from gymnasium import spaces, logger
from simulator.engines.ORCARLEngine import ORCARLEngine
from simulator.engines.base import SimulationState
from simulator.models.messages import AgentPositionsUpdateMessage, GoalsProcessedMessage, ObstaclesProcessedMessage, RayCastingUpdateMessage, SimulationInitializedMessage
from simulator.models.observer import SimulationSubject
from simulator.models.simulation import Simulation as SimulationModel
from rendering.pygame_renderer import PyGameRenderer
from rendering.text_renderer import TextRenderer
from rvo2_rl.rl import ObsMode


class RVOBaseEnv2(gym.Env, SimulationSubject):
    """
    Base environment for MIAC single-agent scenarios.
    Centralizes:
      - Config loading
      - Seeding
      - Rendering
      - Consistent signatures (step, reset, render, etc.)
      - Default 'naive' / 'min_dist' step behavior
    """

    metadata = {'render.modes': ['ansi', 'rgb_array', 'human']}

    def __init__(self, config_file=None, render_mode="human", seed=None, step_mode='min_dist', use_lidar=False, use_obs_mask=False, mode=ObsMode.Cartesian):
        """
        :param config_file: path to YAML config (optional)
        :param render_mode: 'rgb', 'ansi', or None
        :param seed: optional random seed
        :param step_mode: 'naive' or 'min_dist'; defines how step interprets the action
        """
        gym.Env().__init__()
        SimulationSubject.__init__(self)
        self.config_file = config_file
        self.render_mode = render_mode
        self.seed_val = seed
        self.step_mode = step_mode  # either 'naive' or 'min_dist'
        self.engine = None
        self.use_lidar = use_lidar

        # Load config if provided
        if config_file:
            self._load_config(config_file)

        # Set up simulator if config was loaded
        self.max_step_count = self.world_config.max_steps
        if hasattr(self, 'world_config'):
            self._init_simulator(use_lidar=use_lidar,
                                 use_obs_mask=use_obs_mask, mode=mode)
        
        self.action_space = spaces.Box(
            low=np.array([-np.pi, -2.0], dtype=np.float32),
            high=np.array([np.pi, 2.0], dtype=np.float32),
            dtype=np.float32
        )
        agent_bounds = self.engine.get_obs_limits()
        # print("get obs limits")
        # pprint(agent_bounds, indent= 8)
        self.observation_space = spaces.Box(
            low=agent_bounds["low"],
            high=agent_bounds["high"]
        )

    def _load_config(self, config_file):
        """Load YAML configuration and store it in self.world_config."""
        with open(config_file, 'r') as stream:
            config_data = yaml.safe_load(stream)
        self.world_config = SimulationModel(**config_data['simulation'])

    def _init_simulator(self, use_lidar=False, use_obs_mask=False, mode=ObsMode.Cartesian):
        """Initialize the RVO2 simulator and register dynamics."""
        # print("init0")
        self.engine = ORCARLEngine(max_steps=self.max_step_count, world_config=self.world_config, simulation_id="test_simulation", seed=self.seed_val)
        # print("init1")
        for dynamic_config in self.world_config.dynamics:
            self.engine.register_dynamic(dynamic_config)
        # print("init2")
        self._init_renderers()
        # print("init3")
        self.engine.initialize_simulation(use_lidar=use_lidar, use_obs_mask=use_obs_mask, mode=mode)
        # print("init4")
        # pprint(self.engine.agent_initialization_data, indent=10)
        if self.render_mode != None:
            self.notify_observers(SimulationInitializedMessage(
                step=-1, agent_initialization_data=self.engine.agent_initialization_data))
            self.notify_observers(GoalsProcessedMessage(
                        step=-1, goals=self.engine.agent_goals))
            self.notify_observers(ObstaclesProcessedMessage(
                step=-1, obstacles=self.engine.obstacle_shapes))
            
        # print("init5")

    def _init_renderers(self):
        """
        We choose which renderer to register based on self.render_mode.
        - 'human'      -> Normal PyGameRenderer with a visible window.
        - 'rgb_array'  -> RecordablePyGameRenderer for silent capturing.
        - 'ansi'       -> TextRenderer in console.
        - None         -> No rendering at all.
        """
        if self.render_mode is None:
            self.renderer = None
            return

        wcfg = self.engine.world_config.map_settings
        window_width = int((wcfg.x_max - wcfg.x_min) * wcfg.cell_size)
        window_height = int((wcfg.y_max - wcfg.y_min) * wcfg.cell_size)
        show_goals = wcfg.show_goals
        intelligent_agent_id = self.engine.world_config.intelligent_agent_id
        if self.render_mode == "human":
            self.renderer = PyGameRenderer(
                width=window_width,
                height=window_height,
                cell_size=int(wcfg.cell_size),
                obstacles=[],
                goals={},
                show_goals = show_goals,
                intelligent_agent_id = intelligent_agent_id
            )
            self.renderer.setup()
            self.register_observer(self.renderer)

        elif self.render_mode == "rgb_array":
            # IMPORTANT: import the recordable renderer
            from rendering.recordable_pygame_renderer import RecordablePyGameRenderer

            self.renderer = RecordablePyGameRenderer(
                width=window_width,
                height=window_height,
                cell_size=int(wcfg.cell_size),
                obstacles=[],
                goals={},
                record_all=False
            )
            self.renderer.setup()
            self.register_observer(self.renderer)

        elif self.render_mode == "ansi":
            self.renderer = TextRenderer()
            self.renderer.setup()
            self.register_observer(self.renderer)

        else:
            # No known render mode
            self.renderer = None

    def step(self, action):
        if self.engine is None:
            raise RuntimeError(
                "Simulator not initialized. Please check your config_file.")
        # step_count = self.engine.get_step_count()

        # 1. Determine base velocity
        self.engine.set_pref_vel_all_agents()
        if self.step_mode == 'min_dist':
            pref_vel_ag_0 = self.engine.get_agent_pref_velocity(0)
            base_vel = np.array([pref_vel_ag_0.x(), pref_vel_ag_0.y()])
        elif self.step_mode == 'naive':
            base_vel = np.zeros(2, dtype=np.float32)
        else:
            raise ValueError("Unknown step_mode: {}".format(self.step_mode))

        # 2. Interpret action as (delta_angle, delta_magnitude)
        delta_angle, delta_mag = action
        # 3. Convert base_vel to polar form
        base_theta = np.arctan2(base_vel[1], base_vel[0])
        # 4. Compute deviation vector in (x, y)
        angle = base_theta + delta_angle
        dev_vector = delta_mag * \
            np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        # 5. Add deviation to base_vel
        new_vel = base_vel + dev_vector

        # 6. Clip velocity magnitude
        min_magnitude = self.engine.get_agent_min_speed(0)
        max_magnitude = self.engine.get_agent_max_speed(0)
        magnitude = np.linalg.norm(new_vel)

        if magnitude < min_magnitude:
            if magnitude > 1e-9:
                clipped = (new_vel / magnitude) * min_magnitude
            else:
                clipped = np.zeros_like(new_vel)
        elif magnitude > max_magnitude:
            clipped = (new_vel / magnitude) * max_magnitude
        else:
            clipped = new_vel
        current_pos = self.engine.get_agent_position(0)
        # print("----------------------------------------------------")
        # print("step count: ", step_count, " clipped vel: ",
        #       clipped, " position: ", current_pos)
        # 7. Update simulator
        # if step_count % 12 == 0:
        #     print("step: ", step_count, "pref_vel: (", pref_vel_ag_0.x(), ", ",
        #           pref_vel_ag_0.y(), "); nn vel: (", clipped, ")")
        self.engine.store_pref_vel_man_update(0, tuple(clipped))

        self.engine.execute_simulation_step()
        self.engine.current_step += 1
        # 8. Collect results
        obs = self._get_obs()

        done = self.is_done(0)

        truncated = (self.engine.get_state() == SimulationState.STOPPED)
        reward = self.calculate_reward(0, done, truncated)

        info = self._get_info()
        return obs, reward, done, truncated, info

    def render(self):
        """
        - If 'human': the PyGame window is drawn automatically by observer updates,
          but typically you'd call this method to handle any final flips or returns of None.
        - If 'rgb_array': return the last frame from the recordable renderer.
        - If 'ansi': optionally return a string or None.
        """
        if self.renderer is None:
            logger.warn("Render called but no renderer is initialized.")
            return None

        if self.render_mode == "human":
            # Typically do nothing except possibly handle PyGame events,
            agent_data = self.engine.collect_agents_batch_data()            
            step = self.engine.get_step_count()
            if self.use_lidar:
                lidar_readings= self.engine.get_lidar_readings(0)                
                self.notify_observers(RayCastingUpdateMessage(
                    step=step, intersections=lidar_readings))        
            self.notify_observers(AgentPositionsUpdateMessage(
                step=step, agent_positions=agent_data))
            
            return None

        elif self.render_mode == "rgb_array":
            agent_data = self.engine.collect_agents_batch_data()
            step = self.engine.get_step_count()
            self.notify_observers(AgentPositionsUpdateMessage(
                step=step, agent_positions=agent_data))
            return self.renderer.get_rgb_array()

        elif self.render_mode == "ansi":
            return None

        return None

    def reset(self, seed=None, options=None):
        """Resets the environment and the simulation."""
        if self.engine is None:
            raise RuntimeError(
                "Simulator is not initialized. Make sure config_file is valid.")

        # if seed is not None:
        #     self.engine.reset_rng_with_seed(seed)
        # else:
        self.engine.reset()
        return self._get_obs(), self._get_info()

    def calculate_reward(self, agent_id, done, truncated):
        """Override in child classes (default raises error)."""
        reward = -20
        step_count = self.engine.get_step_count()
        max_steps = self.max_step_count
        steps_saved = max_steps - step_count

        if done:
            # flat bonus for winning
            winning_bonus = 5120

            # triangular bonus: 1 + 2 + … + steps_saved = steps_saved*(steps_saved+1)/2
            step_bonus = steps_saved * (steps_saved + 1) / 2

            reward += winning_bonus + (step_bonus * 20)
            return reward

        if truncated and not done:
            penalty = (max_steps / step_count) * 1000
            dist_to_goal = self.engine.get_distance_to_goal(agent_id, True)
            bonus = (1-dist_to_goal if dist_to_goal >
                     1 else dist_to_goal) * 5120
            reward += bonus - penalty

        return reward

    def is_done(self, agent_id):
        """Override in child classes (default raises error)."""
        raise NotImplementedError(
            "Please implement is_done() in the child class.")

    def _get_obs(self):
        """Override in child classes if you need actual observations."""
        # Default returns empty observation.
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_info(self):
        """Override in child classes if you need custom info."""
        return {}
