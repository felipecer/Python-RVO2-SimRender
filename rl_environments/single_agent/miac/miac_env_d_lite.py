#!/usr/bin/env python
import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces, logger

from rendering.pygame_renderer import PyGameRenderer
from simulator.engines.RVO2SimulatorWrapper import RVO2SimulatorWrapper
from simulator.engines.base import SimulationState
from simulator.models.simulation import Simulation as SimulationModel


class RVOSimulationEnvMIAC(gym.Env):
    metadata = {'render.modes': ['ansi', 'rgb']}

    def __init__(self, config_file=None, render_mode="rgb", seed=None):
        super(RVOSimulationEnvMIAC, self).__init__()
        # Cargar configuración YAML
        with open(config_file, 'r') as stream:
            config_data = yaml.safe_load(stream)

        # pprint(config_data, indent=6)
        # Inicializar el simulador con RVO2SimulatorWrapper, pasando el seed
        world_config = SimulationModel(**config_data['simulation'])   
        # world_config.agents[0].agent_defaults = world_config.agent_defaults
        print(world_config.agents)     
        dynamics = world_config.dynamics
        self.sim = RVO2SimulatorWrapper(world_config, "test_simulation", seed=seed)
        for dynamic_config in dynamics:
            self.sim.register_dynamic(dynamic_config)

        # Establecer renderers si se requiere renderización
        self.render_mode = render_mode
        if render_mode == "rgb":
            window_width = int((
                                           self.sim.world_config.map_settings.x_max - self.sim.world_config.map_settings.x_min) * self.sim.world_config.map_settings.cell_size)
            window_height = int((
                                            self.sim.world_config.map_settings.y_max - self.sim.world_config.map_settings.y_min) * self.sim.world_config.map_settings.cell_size)

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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(92,), dtype=np.float32)
        self.sim.initialize_simulation()
        self.sim.set_agent_defaults(0, world_config.agent_defaults)

    def _get_obs(self):
        """Obtiene la observación para el agente 0 con padding en los datos de vecinos."""
        # Obtener posición y objetivo del agente
        pos = self.sim.get_agent_position(0)
        goal = self.sim.get_goal(0)

        # Obtener el máximo número de vecinos permitidos y los datos actuales de vecinos
        max_neigh = self.sim.get_agent_max_num_neighbors(0)
        agent_neighbors = self.sim.get_neighbors_data(0)  # Supongamos que devuelve una lista de floats

        # Número esperado de elementos para los datos de vecinos
        expected_neighbors_data_length = max_neigh * 6

        # Si hay menos datos de vecinos que el esperado, hacer padding con -9999
        if len(agent_neighbors) < expected_neighbors_data_length:
            padding_length = expected_neighbors_data_length - len(agent_neighbors)
            agent_neighbors.extend([-9999] * padding_length)
        else:
            # En caso de que haya más datos (puede que no sea necesario), truncamos
            agent_neighbors = agent_neighbors[:expected_neighbors_data_length]

        # Calcular la observación inicial (diferencia entre posición y objetivo)
        observations = [goal[0] - pos[0], goal[1] - pos[1]]

        # Concatenar las observaciones iniciales con los datos de vecinos (ya con padding)
        observations.extend(agent_neighbors)
        # Convertir la observación en un array numpy y retornarla
        return np.array(observations, dtype=np.float32)

        # return observations, reward, done, truncated, info
    def step(self, action):
        """Realiza un paso en la simulación con la acción como desviación."""
        velocity = self.sim.get_velocity_min_euclid_dist(0)
        # Interpretar la acción como una desviación
        deviation = np.array(action)
        velocity = velocity + deviation
        # Obtener la magnitud máxima y mínima permitida
        min_magnitude = self.sim.get_agent_min_speed(0)
        max_magnitude = self.sim.get_agent_max_speed(0)

        # Calcular la magnitud actual de la velocidad
        velocity_magnitude = np.linalg.norm(velocity)

        # Si la magnitud está fuera de los límites, ajustar
        if velocity_magnitude < min_magnitude:
            clipped_velocity = (velocity / velocity_magnitude) * min_magnitude
        elif velocity_magnitude > max_magnitude:
            clipped_velocity = (velocity / velocity_magnitude) * max_magnitude
        else:
            clipped_velocity = velocity  # Mantener sin cambios si está dentro del rango
        self.sim.update_agent_velocity(0, tuple(clipped_velocity))
        self.sim.update_agent_velocities()
        self.sim.execute_simulation_step()
        self.sim.current_step += 1

        # Obtener nueva observación
        observations = self._get_obs()

        # Calcular recompensa
        reward = self.calculate_reward(0)

        # Verificar si el episodio ha terminado
        done = self.is_done(0)
        truncated = self.sim.get_state() == SimulationState.STOPPED
        info = self._get_info()

        # Renderizar si es necesario
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
    env = RVOSimulationEnvMIAC('./simulator/worlds/miac/circle/circle_level_0.yaml', render_mode='rgb')
    observations = env.reset()
    done = False
    i = 0
    while not done:
        action = env.action_space.sample()  # Take random actions
        # print(f"Action: {action}")
        observations, reward, done, truncated, info = env.step(action)
        # print(f"Observations: {observations}")

        # Monitorear la posición del agente
        agent_position = env.sim.get_agent_position(0)
        # print(f"Step {i}: Agent position: {agent_position}")
        if done or truncated:
            print(f"Episode done: {done}, truncated: {truncated}")
            break
        # print(f"Step {i} reward: {reward}")
        i += 1
