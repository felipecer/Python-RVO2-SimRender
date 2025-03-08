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

    metadata = {'render.modes': ['ansi', 'rgb']}

    def __init__(self, config_file=None, render_mode="rgb", seed=None, step_mode='naive'):
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
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(92,), dtype=np.float32)

    def _load_config(self, config_file):
        """Load YAML configuration and store it in self.world_config."""
        with open(config_file, 'r') as stream:
            config_data = yaml.safe_load(stream)
        self.world_config = SimulationModel(**config_data['simulation'])

    def _init_simulator(self):
        """Initialize the RVO2 simulator and register dynamics."""
        self.sim = RVO2SimulatorWrapper(self.world_config, "test_simulation", seed=self.seed_val)
        for dynamic_config in self.world_config.dynamics:
            self.sim.register_dynamic(dynamic_config)
        self._init_renderers()
        self.sim.initialize_simulation()

    def _init_renderers(self):
        """Set up renderers based on render_mode."""
        if self.render_mode == "rgb":
            wcfg = self.sim.world_config.map_settings
            window_width = int((wcfg.x_max - wcfg.x_min) * wcfg.cell_size)
            window_height = int((wcfg.y_max - wcfg.y_min) * wcfg.cell_size)

            renderer = PyGameRenderer(
                width=window_width,
                height=window_height,
                obstacles=[],
                goals={},
                cell_size=int(wcfg.cell_size)
            )
            renderer.setup()
            self.sim.register_observer(renderer)

        elif self.render_mode == 'ansi':
            renderer = TextRenderer()
            renderer.setup()
            self.sim.register_observer(renderer)

        else:
            pass  # No rendering

    def step(self, action):
        """
        Default step method with two built-in modes:
          - 'naive': interpret action as the velocity directly
          - 'min_dist': interpret action as a deviation from the min-euclid-dist velocity
        Child classes can override this if needed.
        """
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized. Make sure config_file is valid.")

        # 1. Determine base velocity according to step_mode
        if self.step_mode == 'min_dist':
            base_vel = np.array(self.sim.get_velocity_min_euclid_dist(0))
        elif self.step_mode == 'naive':
            # 'naive' means the agent doesn't automatically move toward the goal;
            # the action itself is the entire motion vector
            base_vel = np.zeros(2, dtype=np.float32)
        else:
            raise ValueError(f"Unknown step_mode '{self.step_mode}'. Choose 'naive' or 'min_dist'.")

        # 2. Add the action as a deviation (for 'min_dist') or as the entire velocity (for 'naive')
        new_vel = base_vel + np.array(action)

        # 3. Clip velocity to min/max
        min_magnitude = self.sim.get_agent_min_speed(0)
        max_magnitude = self.sim.get_agent_max_speed(0)
        magnitude = np.linalg.norm(new_vel)

        if magnitude < min_magnitude:
            clipped = (new_vel / magnitude) * min_magnitude if magnitude > 1e-9 else np.zeros_like(new_vel)
        elif magnitude > max_magnitude:
            clipped = (new_vel / magnitude) * max_magnitude
        else:
            clipped = new_vel

        # 4. Update simulator
        self.sim.update_agent_velocity(0, tuple(clipped))
        self.sim.update_agent_velocities()
        self.sim.execute_simulation_step()
        self.sim.current_step += 1

        # 5. Collect results
        obs = self._get_obs()
        reward = self.calculate_reward(0)
        done = self.is_done(0)
        truncated = self.sim.get_state() == SimulationState.STOPPED
        info = self._get_info()

        # 6. Optionally render
        if self.render_mode:
            self.render()

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment and the simulation."""
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized. Make sure config_file is valid.")

        if seed is not None:
            self.sim.reset_rng_with_seed(seed)
        self.sim.reset()
        return None, {}

    def render(self):
        """Delegates rendering to the simulator's observers (if any)."""
        if not self.render_mode:
            logger.warn("Render called without a valid render_mode.")
        # If a renderer is registered, it handles drawing at each step.

    def calculate_reward(self, agent_id):
        """Override in child classes (default raises error)."""
        raise NotImplementedError("Please implement calculate_reward() in the child class.")

    def is_done(self, agent_id):
        """Override in child classes (default raises error)."""
        raise NotImplementedError("Please implement is_done() in the child class.")

    def _get_obs(self):
        """Override in child classes if you need actual observations."""
        # Default returns empty observation.
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_info(self):
        """Override in child classes if you need custom info."""
        return {}