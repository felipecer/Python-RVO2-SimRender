import gymnasium as gym
import yaml
import numpy as np
from gymnasium import spaces, logger
from simulator.engines.RVO2SimulatorWrapper import RVO2SimulatorWrapper
from simulator.engines.base import SimulationState
from simulator.models.simulation import Simulation as SimulationModel
from rendering.pygame_renderer import PyGameRenderer
from rendering.text_renderer import TextRenderer


class RVOBaseEnv(gym.Env):
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

    def __init__(self, config_file=None, render_mode="human", seed=None, step_mode='min_dist', includes_lidar=False):
        """
        :param config_file: path to YAML config (optional)
        :param render_mode: 'rgb', 'ansi', or None
        :param seed: optional random seed
        :param step_mode: 'naive' or 'min_dist'; defines how step interprets the action
        """
        super().__init__()
        self.config_file = config_file
        self.render_mode = render_mode
        self.seed_val = seed
        self.step_mode = step_mode  # either 'naive' or 'min_dist'
        self.sim = None

        # Load config if provided
        if config_file:
            self._load_config(config_file)

        # Set up simulator if config was loaded
        if hasattr(self, 'world_config'):
            self._init_simulator()

        # Initialize default spaces (child classes may override these)
        # Example: assume 2D action, 92D observation

            # Let's assume angle delta ∈ [-π, π], magnitude delta ∈ [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=np.array([-np.pi, -1.0], dtype=np.float32),
            high=np.array([np.pi, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        if includes_lidar:
            obs_d = 3 + 360 + 360 + 15 * 6 + 15
            self.observation_space = spaces.Box(
                low=np.array(
                    [0] +                               # step
                    [-1000, -1000] +                         # agent position
                    [0.0] * 360 +                      # ray distances
                    [0.0] * 360 +                      # ray mask
                    # neighbor data
                    [-1000.0, -1000.0, 0.0, -np.pi, 0.0, -np.pi] * 15 +
                    [0.0] * 15                         # neighbor mask
                    , dtype=np.float32
                ),
                high=np.array(
                    [1] +                # agent position
                    [1000, 1000] +                         # agent orientation
                    [1.0] * 360 +                     # ray distances
                    [1.0] * 360 +                     # ray mask
                    [1000.0, 1000.0, 1.0, np.pi, 1.0, np.pi] * 15 +  # neighbor data
                    [1.0] * 15                        # neighbor mask
                    , dtype=np.float32
                ),
                dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=np.array(
                    [0] +                               # step
                    [-1000, -1000] +                         # agent position
                    # neighbor data
                    [-1000.0, -1000.0, 0.0, -np.pi, 0.0, -np.pi] * 15 +
                    [0.0] * 15                         # neighbor mask
                    , dtype=np.float32
                ),
                high=np.array(
                    [1] +                # agent position
                    [1000, 1000] +                         # agent orientation
                    [1000.0, 1000.0, 1.0, np.pi, 1.0, np.pi] * 15 +  # neighbor data
                    [1.0] * 15                        # neighbor mask
                    , dtype=np.float32
                ),
                dtype=np.float32
            )

    def _load_config(self, config_file):
        """Load YAML configuration and store it in self.world_config."""
        with open(config_file, 'r') as stream:
            config_data = yaml.safe_load(stream)
        self.world_config = SimulationModel(**config_data['simulation'])

    def _init_simulator(self):
        """Initialize the RVO2 simulator and register dynamics."""
        self.sim = RVO2SimulatorWrapper(
            self.world_config, "test_simulation", seed=self.seed_val)
        for dynamic_config in self.world_config.dynamics:
            self.sim.register_dynamic(dynamic_config)
        self._init_renderers()
        self.sim.initialize_simulation()

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

        wcfg = self.sim.world_config.map_settings
        window_width = int((wcfg.x_max - wcfg.x_min) * wcfg.cell_size)
        window_height = int((wcfg.y_max - wcfg.y_min) * wcfg.cell_size)

        if self.render_mode == "human":
            self.renderer = PyGameRenderer(
                width=window_width,
                height=window_height,
                cell_size=int(wcfg.cell_size),
                obstacles=[],
                goals={}
            )
            self.renderer.setup()
            self.sim.register_observer(self.renderer)

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
            self.sim.register_observer(self.renderer)

        elif self.render_mode == "ansi":
            self.renderer = TextRenderer()
            self.renderer.setup()
            self.sim.register_observer(self.renderer)

        else:
            # No known render mode
            self.renderer = None

    def step(self, action):
        if self.sim is None:
            raise RuntimeError(
                "Simulator not initialized. Please check your config_file.")

        # 1. Determine base velocity
        if self.step_mode == 'min_dist':
            coll_free_vel = self.sim.get_collision_free_velocity(0)
            base_vel = np.array([coll_free_vel.x(), coll_free_vel.y()])
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
        min_magnitude = self.sim.get_agent_min_speed(0)
        max_magnitude = self.sim.get_agent_max_speed(0)
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

        # 4. Update simulator
        self.sim.update_agent_velocity(0, tuple(clipped))
        # self.sim.update_agent_velocities()
        self.sim.execute_simulation_step()
        self.sim.current_step += 1

        # 5. Collect results
        obs = self._get_obs()
        reward = self.calculate_reward(0)
        done = self.is_done(0)
        truncated = (self.sim.get_state() == SimulationState.STOPPED)
        if truncated and not done:
            reward += (1 - (self.sim.distance_from_goal(0) /
                       self.sim.initial_distance_from_goal_array[0])) * 2560
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
            # because the observer pattern already updates the window each simulation step.
            return None

        elif self.render_mode == "rgb_array":
            return self.renderer.get_rgb_array()

        elif self.render_mode == "ansi":
            return None

        return None

    def reset(self, seed=None, options=None):
        """Resets the environment and the simulation."""
        if self.sim is None:
            raise RuntimeError(
                "Simulator is not initialized. Make sure config_file is valid.")

        if seed is not None:
            self.sim.reset_rng_with_seed(seed)
        self.sim.reset()
        return self._get_obs(), self._get_info()

    # def render(self):
    #     """Delegates rendering to the simulator's observers (if any)."""
    #     if not self.render_mode:
    #         logger.warn("Render called without a valid render_mode.")
    #     # If a renderer is registered, it handles drawing at each step.

    def calculate_reward(self, agent_id):
        """Override in child classes (default raises error)."""
        raise NotImplementedError(
            "Please implement calculate_reward() in the child class.")

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
