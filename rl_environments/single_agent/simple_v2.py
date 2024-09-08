#!/usr/bin/env python
import gymnasium as gym
from gymnasium import spaces, logger
import numpy as np
import yaml
from rendering.pygame_renderer import PyGameRenderer
from simulator.engines.RVO2SimulatorWrapper import RVO2SimulatorWrapper
from simulator.engines.base import SimulationState
from simulator.models.simulation import Simulation as SimulationModel

class RVOSimulationEnv2(gym.Env):
    metadata = {'render.modes': ['ansi', 'rgb']}

    def __init__(self, config_file=None, render_mode="rgb", seed=None):
        super(RVOSimulationEnv2, self).__init__()
        # Cargar configuración YAML
        with open(config_file, 'r') as stream:
            config_data = yaml.safe_load(stream)
        
        # print(type(config_data))
        # Inicializar el simulador con RVO2SimulatorWrapper, pasando el seed
        self.sim = RVO2SimulatorWrapper(SimulationModel(**config_data['simulation']), "test_simulation", seed=seed)

        # Establecer renderers si se requiere renderización
        self.render_mode = render_mode
        if render_mode == "rgb":
            window_width = int((self.sim.world_config.map_settings.x_max - self.sim.world_config.map_settings.x_min) * self.sim.world_config.map_settings.cell_size)
            window_height = int((self.sim.world_config.map_settings.y_max - self.sim.world_config.map_settings.y_min) * self.sim.world_config.map_settings.cell_size)

            renderer = PyGameRenderer(
                window_width,
                window_height,
                obstacles=[], 
                goals={}, 
                cell_size=int(self.sim.world_config.map_settings.cell_size)
            )
            renderer.setup()
            self.sim.register_observer(renderer)
            # print(f"Observadores registrados: {self.sim._observers}")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.sim.initialize_simulation()

    def _get_obs(self):
        """Obtiene la observación para el agente 0."""
        pos = self.sim.get_agent_position(0)
        goal = self.sim.get_goal(0)
        observations = [goal[0] - pos[0], goal[1] - pos[1]]
        return np.array(observations, dtype=np.float32)

    def step(self, action):
        """Realiza un paso en la simulación con la acción proporcionada."""
        action = tuple(action)
        # print(f"Action: {action}")
        self.sim.update_agent_velocity(0, action)
        self.sim.update_agent_velocities()
        self.sim.execute_simulation_step()
        self.sim.current_step += 1
        observations = self._get_obs()
        reward = self.calculate_reward(0)
        done = self.is_done(0)
        # truncated = self.sim.current_step >= self.sim.world_config.time_limit  # Usar el time_limit del world_config
        truncated = self.sim.get_state() == SimulationState.STOPPED
        info = self._get_info()

        if self.render_mode:
            self.render()

        return observations, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """Resetea el entorno y la simulación."""
        if seed is not None:
            self.sim.reset_rng_with_seed(seed)  # Actualizar el seed si se proporciona
        self.sim.reset()
        return self._get_obs(), self._get_info()

    def render(self):
        """Renderiza el estado actual de la simulación."""
        if self.render_mode is None:
            logger.warn("You are calling render method without specifying any render mode.")
        else:
            pass
            # self.sim.render()  # Delegado al simulador

    def calculate_reward(self, agent_id):
        """Calcula la recompensa para el agente 0."""
        reward = -10
        if self.is_done(agent_id):
            reward += 10000
        return reward

    def is_done(self, agent_id):
        """Determina si el agente ha alcanzado su meta."""
        return self.sim.is_goal_reached(agent_id)

    def _get_info(self):
        """Retorna información adicional sobre el entorno."""
        return {}

if __name__ == "__main__":
    env = RVOSimulationEnv2('./simulator/worlds/simple_v2.yaml', render_mode='rgb')
    observations = env.reset()
    done = False
    i = 0
    while not done:
        action = env.action_space.sample()  # Take random actions
        # print(f"Action: {action}")
        observations, reward, done, truncated, info = env.step(action)

        # Monitorear la posición del agente
        agent_position = env.sim.get_agent_position(0)
        # print(f"Step {i}: Agent position: {agent_position}")
        if done or truncated:
            # print(f"Episode done: {done}, truncated: {truncated}")
            break
        # print(f"Step {i} reward: {reward}")
        i += 1
