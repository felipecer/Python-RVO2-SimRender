from typing import Dict, Tuple
import rvo2
import argparse
from pydantic import BaseModel, ValidationError
import math
import sys
import yaml
from rendering.pygame_renderer import PyGameRenderer
from rendering.text_renderer import TextRenderer
from simulator.engines.base import SimulationEngine, SimulationState
from simulator.models.observer import SimulationSubject
from simulator.models.simulation import Simulation
from simulator.models.messages import (
    SimulationInitializedMessage,
    AgentPositionsUpdateMessage,
    ObstaclesProcessedMessage,
    GoalsProcessedMessage
)
from simulator.models.simulation_configuration.simulation_events import GoalReachedEvent


class RVO2SimulatorWrapper(SimulationEngine, SimulationSubject):
    def __init__(self, world_config: BaseModel, simulation_id: str, seed: int = None):
        SimulationEngine.__init__(self, seed=seed)
        SimulationSubject.__init__(self)
        """
        Inicializa el simulador RVO2 con la configuración del mundo y un renderizador opcional.

        Args:
            world_config (BaseModel): La configuración del mundo en formato de Pydantic.
            simulation_id (str): El ID de la simulación actual.
        """
        self.world_config = world_config  # Almacena la configuración del mundo proporcionada
        self.simulation_id = simulation_id  # Almacena el ID de la simulación
        self.sim = None  # Instancia del simulador RVO2, se inicializará más tarde
        self.agent_goals = {}  # Diccionario para almacenar los objetivos de los agentes
        self.steps_buffer = []  # Buffer para almacenar los datos de cada paso de la simulación
        self.obstacles = []
        self.agent_initial_positions = []
        self._manual_velocity_updates = []

    def calculate_preferred_velocity(self, agent_position, goal_position, max_speed):
        vector_to_goal = (
            goal_position[0] - agent_position[0],
            goal_position[1] - agent_position[1]
        )
        distance = math.sqrt(vector_to_goal[0] ** 2 + vector_to_goal[1] ** 2)

        if distance > 0:
            return (
                vector_to_goal[0] / distance * max_speed,
                vector_to_goal[1] / distance * max_speed
            )
        else:
            return (0, 0)

    def initialize_simulation(self):
        """
        Método para inicializar la simulación RVO2 con la configuración proporcionada.
        Este método convertirá los objetos Pydantic en objetos compatibles con RVO2.
        """
        config = self.world_config  # Accede a la configuración del mundo.
        self.sim = rvo2.PyRVOSimulator(
            config.time_step,
            config.agent_defaults.neighbor_dist,
            config.agent_defaults.max_neighbors,
            config.agent_defaults.time_horizon,
            config.agent_defaults.time_horizon_obst,
            config.agent_defaults.radius,
            config.agent_defaults.max_speed
        )

        # Añadir agentes y guardar sus metas
        # Inicialización del contador global de agent_id
        global_agent_id = 0
        agent_behaviours = {}

        # Iteramos sobre cada grupo de agentes
        for agent_group in config.agents:
            positions = agent_group.pattern.generate_positions()

            # Generamos las posiciones de las metas para este grupo si existen
            goals = agent_group.goals.pattern.generate_positions() if agent_group.goals else None

            # Iteramos sobre las posiciones generadas de los agentes
            for local_agent_index, position in enumerate(positions):
                agent_defaults = agent_group.agent_defaults or config.agent_defaults

                # Agregamos el agente a la simulación de rvo2 y obtenemos su ID global
                agent_id = self.sim.addAgent(
                    tuple(position),
                    agent_defaults.neighbor_dist,
                    agent_defaults.max_neighbors,
                    agent_defaults.time_horizon,
                    agent_defaults.time_horizon_obst,
                    agent_defaults.radius,
                    agent_defaults.max_speed,
                    agent_defaults.velocity
                )

                # Configuramos la velocidad preferida del agente
                self.sim.setAgentPrefVelocity(
                    agent_id, agent_defaults.velocity)
                self.agent_initial_positions.append(position)
                # Si hay metas definidas para el grupo de agentes
                if goals:
                    # Asignamos la meta correcta al agente usando el índice local
                    self.agent_goals[agent_id] = goals[local_agent_index]
                    self.notify_observers(GoalsProcessedMessage(
                        step=-1, goals=self.agent_goals))

                # Almacenar el comportamiento del agente en el diccionario
                agent_behaviours[agent_id] = agent_group.behaviour
                # Incrementamos el ID global del agente para el siguiente agente
                global_agent_id += 1

        # Añadir obstáculos a la simulación
        if config.obstacles:
            obstacle_shapes = []
            for obstacle_shape in config.obstacles:
                shape = obstacle_shape.generate_shape()
                self.sim.addObstacle(shape)
                obstacle_shapes.append(shape)
            self.sim.processObstacles()
            self.notify_observers(ObstaclesProcessedMessage(
                step=-1, obstacles=obstacle_shapes))

        agent_initialization_data = [
            {
                "agent_id": agent_id,
                "radius": self.sim.getAgentRadius(agent_id),
                "max_speed": self.sim.getAgentMaxSpeed(agent_id),
                "neighbor_dist": self.sim.getAgentNeighborDist(agent_id),
                "max_neighbors": self.sim.getAgentMaxNeighbors(agent_id),
                "time_horizon": self.sim.getAgentTimeHorizon(agent_id),
                "time_horizon_obst": self.sim.getAgentTimeHorizonObst(agent_id),
                "goal": self.agent_goals[agent_id],
                # Comportamiento del agente
                "behaviour": agent_behaviours.get(agent_id)
            }
            for agent_id in range(self.sim.getNumAgents())
        ]

        # Enviar la información de inicialización a los observadores
        self.notify_observers(SimulationInitializedMessage(
            step=-1, agent_initialization_data=agent_initialization_data))

    def reset(self):
        """Reinicia la simulación a su estado inicial."""
        self.current_step = 0
        self._state = SimulationState.SETUP

        # Restablecer las posiciones de los agentes a las posiciones iniciales
        for agent_id, initial_position in enumerate(self.agent_initial_positions):
            self.sim.setAgentPosition(agent_id, initial_position)

        # Restablecer cualquier otra variable relevante (como metas)
        # Opcional: Si necesitas reiniciar las metas, puedes hacerlo aquí
        for agent_id in self.agent_goals:
            # Si necesitas actualizar las metas puedes hacerlo aquí, si no simplemente reinicializa la simulación
            self.set_goal(agent_id, self.agent_goals[agent_id])

    def step(self):
        """
        Ejecuta la simulación durante un número especificado de pasos.

        Args:
            steps (int): Número de pasos que la simulación debe ejecutar.
        """
        self.update_agent_velocities()
        self.sim.doStep()

        # Detectar si algún agente ha alcanzado su meta
        for agent_id in range(self.sim.getNumAgents()):
            if self.is_goal_reached(agent_id):
                event = GoalReachedEvent(
                    agent_id=agent_id,
                    goal_position=self.agent_goals[agent_id],
                    current_position=self.sim.getAgentPosition(agent_id),
                    step=self.current_step
                )
                self.handle_event(event.alias, event)

        # Recolectar más datos de cada agente
        agent_data = [
            (
                agent_id,
                *self.sim.getAgentPosition(agent_id),  # Posición actual (x, y)
                self.sim.getAgentVelocity(agent_id),  # Velocidad actual
                self.sim.getAgentPrefVelocity(agent_id),  # Velocidad preferida
                math.dist(self.sim.getAgentPosition(agent_id),
                          self.agent_goals[agent_id])  # Distancia a la meta
            )
            for agent_id in range(self.sim.getNumAgents())
        ]
        # Enviar el mensaje con los datos adicionales
        self.notify_observers(AgentPositionsUpdateMessage(
            step=self.current_step, agent_positions=agent_data))

        self.store_step(self.current_step)

    def run_simulation(self, steps: int):
        """
        Ejecuta la simulación durante un número especificado de pasos.

        Args:
            steps (int): Número de pasos que la simulación debe ejecutar.
        """
        for step in range(steps):
            self.update_agent_velocities()
            self.sim.doStep()

            # Detectar si algún agente ha alcanzado su meta
            for agent_id in range(self.sim.getNumAgents()):
                if self.is_goal_reached(agent_id):
                    event = GoalReachedEvent(
                        agent_id=agent_id,
                        goal_position=self.agent_goals[agent_id],
                        current_position=self.sim.getAgentPosition(agent_id),
                        step=step
                    )
                    self.handle_event(event.alias, event)

            agent_positions = [(agent_id, *self.sim.getAgentPosition(agent_id))
                               for agent_id in range(self.sim.getNumAgents())]
            print(
                f"Sending AgentPositionsUpdateMessage for step {self.current_step}")
            self.notify_observers(AgentPositionsUpdateMessage(
                step=self.current_step, agent_positions=agent_positions))
            self.store_step(step)

    # def is_goal_reached(self, agent_id: int) -> bool:
    #     current_position = self.sim.getAgentPosition(agent_id)
    #     goal_position = self.agent_goals[agent_id]
    #     distance = math.dist(current_position, goal_position)
    #     return distance <= 0.25

    def store_step(self, step: int):
        """
        Almacena la información de los agentes en un paso dado.

        Args:
            step (int): El número de paso actual en la simulación.
        """
        step_data = {'step': step, 'agents': []}
        for agent_id in range(self.sim.getNumAgents()):
            position = self.sim.getAgentPosition(agent_id)
            step_data['agents'].append({
                'id': agent_id,
                'position': position
            })
        self.steps_buffer.append(step_data)

    def save_simulation_runs(self):
        """
        Guarda los resultados de la simulación en un archivo.
        """
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # filename = f"nombre_hardcodeado_{self.simulation_id}_{timestamp}.txt".replace(
        #     " ", "_")

        # with open(filename, 'w') as file:
        #     for step_data in self.steps_buffer:
        #         step = step_data['step']
        #         for agent_data in step_data['agents']:
        #             file.write(
        #                 f"{step},{agent_data['id']},{agent_data['position'][0]:.2f},{agent_data['position'][1]:.2f}\n")

        # print(f"Archivo de simulación guardado como: {filename}")
        pass

    def update_agent_velocity(self, agent_id: int, velocity: Tuple[float, float]):
        """
        Registra una actualización manual de la velocidad para un agente específico.
        """
        self._manual_velocity_updates.append((agent_id, velocity))

    def update_agent_velocities(self):
        """
        Actualiza las velocidades preferidas de los agentes en la simulación, considerando actualizaciones manuales.
        """
        # Aplicar actualizaciones manuales primero
        manual_update_ids = set(agent_id for agent_id,
                                _ in self._manual_velocity_updates)
        # print("Manual updates:", self._manual_velocity_updates)
        for agent_id, velocity in self._manual_velocity_updates:
            self.sim.setAgentPrefVelocity(agent_id, velocity)

        # Actualizar el resto de los agentes con la lógica por defecto
        num_goals = len(self.agent_goals)
        for agent_id in range(self.sim.getNumAgents()):
            if agent_id >= num_goals or agent_id in manual_update_ids:
                continue

            agent_position = self.sim.getAgentPosition(agent_id)
            goal_position = self.agent_goals[agent_id]
            if goal_position:
                vector_to_goal = (
                    goal_position[0] - agent_position[0],
                    goal_position[1] - agent_position[1]
                )
                distance = math.sqrt(
                    vector_to_goal[0] ** 2 + vector_to_goal[1] ** 2)
                max_speed = self.sim.getAgentMaxSpeed(agent_id)

                if distance > 0:
                    preferred_velocity = (
                        vector_to_goal[0] / distance * max_speed,
                        vector_to_goal[1] / distance * max_speed
                    )
                else:
                    preferred_velocity = (0, 0)

                self.sim.setAgentPrefVelocity(agent_id, preferred_velocity)

        # Limpiar la cola después de aplicar las actualizaciones
        # self._manual_velocity_updates.clear()

    def clear_buffer(self):
        self.steps_buffer = []

    def get_agent_position(self, agent_id) -> Tuple[float, float]:
        """Devuelve la posicion actual del agente."""
        return self.sim.getAgentPosition(agent_id)

    def get_agent_positions(self) -> Dict[int, Tuple[float, float]]:
        """
        Devuelve las posiciones actuales de todos los agentes en la simulación.

        Returns:
            Dict[int, Tuple[float, float]]: Un diccionario donde las claves son los IDs de los agentes y los valores son las posiciones (x, y).
        """
        agent_positions = {}
        for agent_id in range(self.sim.getNumAgents()):
            position = self.sim.getAgentPosition(agent_id)
            agent_positions[agent_id] = position
        return agent_positions

    def set_goal(self, agent_id: int, goal: Tuple[float, float]) -> None:
        """agrega o actualiza la meta del agente dado su id"""
        self.agent_goals[agent_id] = goal

    def get_goal(self, agent_id: int) -> Tuple[float, float]:
        """
        Devuelve la meta actual de un agente dado su ID.

        Args:
            agent_id (int): El ID del agente.

        Returns:
            Tuple[float, float]: La posición de la meta del agente.
        """
        return self.agent_goals.get(agent_id)

    def is_goal_reached(self, agent_id: int) -> bool:
        """
        Verifica si un agente ha alcanzado su meta.

        Args:
            agent_id (int): El ID del agente.

        Returns:
            bool: True si el agente ha alcanzado su meta, False en caso contrario.
        """
        current_position = self.sim.getAgentPosition(agent_id)
        goal_position = self.get_goal(agent_id)
        if not goal_position:
            return False

        distance = math.sqrt(
            (current_position[0] - goal_position[0]) ** 2 +
            (current_position[1] - goal_position[1]) ** 2
        )
        # Considera que se ha alcanzado la meta si la distancia es menor o igual a un umbral
        return distance <= 0.30


def main():
    # Parsear argumentos de la línea de comandos
    parser = argparse.ArgumentParser(
        description='Simulador de Navegación de Agentes')
    parser.add_argument('world_file', type=str,
                        help='Archivo YAML de configuración del mundo')
    parser.add_argument('--renderer', type=str, choices=['pygame', 'text'], default='pygame',
                        help='El tipo de renderer a usar: pygame o text (por defecto: pygame)')
    args = parser.parse_args()

    # Cargar el archivo YAML
    world_file = args.world_file
    try:
        with open(world_file, 'r') as stream:
            data = yaml.safe_load(stream)
            world_config = Simulation(**data['simulation'])

    except FileNotFoundError:
        print(f"File {world_file} not found.")
        sys.exit(1)
    except yaml.YAMLError as exc:
        print(f"Error reading YAML file: {exc}")
        sys.exit(1)
    except ValidationError as exc:
        print(f"Validation error: {exc}")
        sys.exit(1)

    # Configuración de la ventana
    window_width = int((world_config.map_settings.x_max -
                       world_config.map_settings.x_min) * world_config.map_settings.cell_size)
    window_height = int((world_config.map_settings.y_max -
                        world_config.map_settings.y_min) * world_config.map_settings.cell_size)

    # Inicializar el renderer según el flag --renderer
    if args.renderer == 'pygame':
        renderer = PyGameRenderer(
            window_width,
            window_height,
            obstacles=[], goals={}, cell_size=int(world_config.map_settings.cell_size)
        )
        renderer.setup()
    else:
        renderer = TextRenderer()
        renderer.setup()

    # Inicializar el simulador y registrar el renderizador como observador
    rvo2_simulator = RVO2SimulatorWrapper(world_config, "test_simulation")
    rvo2_simulator.register_observer(renderer)

    # Registrar dinámicas desde el archivo YAML
    for dynamic_config in world_config.dynamics:
        rvo2_simulator.register_dynamic(dynamic_config)

    # Ejecutar la simulación
    rvo2_simulator.run_pipeline(5000)  # Se asume 5000 pasos como ejemplo
    rvo2_simulator.save_simulation_runs()


if __name__ == "__main__":
    main()
