from typing import Optional
import rvo2
from pydantic import BaseModel, ValidationError
import math
import datetime
import sys
import yaml
from rendering.pygame_renderer import Grid, PyGameRenderer
from simulator.models.observer import SimulationSubject
from simulator.models.simulation import Simulation
from simulator.models.messages import (
    SimulationInitializedMessage,
    AgentPositionsUpdateMessage,
    ObstaclesProcessedMessage,
    GoalsProcessedMessage
)

class RVO2SimulatorWrapper(SimulationSubject):
    def __init__(self, world_config: BaseModel, simulation_id: str):
        super().__init__()
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
                self.sim.setAgentPrefVelocity(agent_id, agent_defaults.velocity)
        
                # Si hay metas definidas para el grupo de agentes
                if goals:
                    # Asignamos la meta correcta al agente usando el índice local
                    self.agent_goals[agent_id] = goals[local_agent_index]
                
                # Incrementamos el ID global del agente para el siguiente agente
                global_agent_id += 1

        # Añadir obstáculos a la simulación
        if config.obstacles:
            for obstacle_shape in config.obstacles:
                shape =obstacle_shape.generate_shape()       
                self.sim.addObstacle(shape)
            self.sim.processObstacles()

        # Notificar a los observadores sobre la inicialización
        self.notify_observers(SimulationInitializedMessage())

    def run_simulation(self, steps: int):
        """
        Ejecuta la simulación durante un número especificado de pasos.

        Args:
            steps (int): Número de pasos que la simulación debe ejecutar.
        """
        for step in range(steps):
            self.update_agent_velocities()
            self.sim.doStep()

            agent_positions = [(agent_id, *self.sim.getAgentPosition(agent_id))
                               for agent_id in range(self.sim.getNumAgents())]

            self.notify_observers(AgentPositionsUpdateMessage(step=step, agent_positions=agent_positions))
            self.store_step(step)

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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nombre_hardcodeado_{self.simulation_id}_{timestamp}.txt".replace(
            " ", "_")

        with open(filename, 'w') as file:
            for step_data in self.steps_buffer:
                step = step_data['step']
                for agent_data in step_data['agents']:
                    file.write(
                        f"{step},{agent_data['id']},{agent_data['position'][0]:.2f},{agent_data['position'][1]:.2f}\n")

        print(f"Archivo de simulación guardado como: {filename}")

    def update_agent_velocities(self):
        """
        Actualiza las velocidades preferidas de los agentes en la simulación.
        """
        num_goals = len(self.agent_goals)
        for agent_id in range(self.sim.getNumAgents()):
            if agent_id >= num_goals:
                continue
            agent_position = self.sim.getAgentPosition(agent_id)            
            goal_position = self.agent_goals[agent_id]
            if goal_position:
                vector_to_goal = (
                    goal_position[0] - agent_position[0],
                    goal_position[1] - agent_position[1]
                )
                distance = math.sqrt(vector_to_goal[0] ** 2 + vector_to_goal[1] ** 2)
                max_speed = self.sim.getAgentMaxSpeed(agent_id)

                if distance > 0:
                    preferred_velocity = (
                        vector_to_goal[0] / distance * max_speed,
                        vector_to_goal[1] / distance * max_speed
                    )
                else:
                    preferred_velocity = (0, 0)

                self.sim.setAgentPrefVelocity(agent_id, preferred_velocity)
    
    def clear_buffer(self):
        self.steps_buffer = []

def main():
    if len(sys.argv) != 2:
        print("Usage: python simulator.py <world_file.yaml>")
        sys.exit(1)
    
    world_file = sys.argv[1]

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

    window_width = int((world_config.map_settings.x_max - world_config.map_settings.x_min) * world_config.map_settings.cell_size)
    window_height = int((world_config.map_settings.y_max - world_config.map_settings.y_min) * world_config.map_settings.cell_size)

    renderer = PyGameRenderer(
        window_width,
        window_height,
        obstacles=[], goals={}, cell_size=int(world_config.map_settings.cell_size)
    )
    renderer.setup()

    # Inicializar el simulador y registrar el renderizador como observador
    rvo2_simulator = RVO2SimulatorWrapper(world_config, "test_simulation")
    rvo2_simulator.register_observer(renderer)
    rvo2_simulator.initialize_simulation()
    rvo2_simulator.run_simulation(5000)
    rvo2_simulator.save_simulation_runs()

if __name__ == "__main__":
    main()
