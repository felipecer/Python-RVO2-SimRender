from typing import Dict, Tuple
import rvo2
from pydantic import BaseModel, ValidationError
import math
import datetime
import sys
import yaml
from rendering.pygame_renderer import PyGameRenderer
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

class RVO2SimulatorWrapper(SimulationSubject):
    def __init__(self):
        SimulationSubject.__init__(self)
        self.file = open("/home/neopren/repos/Python-RVO2-SimRender/simulator/models/gaps.txt", "r")
        self.numAgents = int(self.file.readline())
        self.current_step = -1
        _ = self.file.readline()
        aux = self.file.readline().split()
        self.origin = (float(aux[0]), float(aux[1]))
        _ = self.file.readline()
        self.agent_goals = {}
        aux = self.file.readline().split()
        self.world_width, self.world_height = (int(aux[0]), int(aux[1]))

        self.obstacles = []
        

    def update(self, message):
        pass

    def getNumAgents(self):
        return self.numAgents
    
    def register_obstacles(self):
        for _ in range(self.world_height):
            fila = list(map(int, self.file.readline().split()))
            self.obstacles.append(fila)
        _ = self.file.readline()

        self.notify_observers(ObstaclesProcessedMessage(
                step=-1, obstacles=self.obstacles))

    def step(self):
        """
        Ejecuta la simulación durante un número especificado de pasos.

        Args:
            steps (int): Número de pasos que la simulación debe ejecutar.
        """

        # por ahora leeré las pos de los agentes desde un archivo...
        self.current_step, agent_positions = self.read()
        if not self.current_step:
            return False
        self.notify_observers(AgentPositionsUpdateMessage(step=self.current_step, agent_positions=agent_positions))

    def read(self):
        # if not stepLine:
        #     return

        # print(self.file.readline().split())

        stepLine = self.file.readline()
        step = stepLine.split()[1]

        agentsPos = []
        for ag in range(int(self.numAgents)):
            coord = self.file.readline().split()
            curr_xr = coord[0]
            curr_yr = coord[1]
            # (0, 0.2595352828502655, -3.7824056148529053, (0.4923849403858185, 0.8703775405883789), (0.4923849403858185, 0.8703775405883789), 6.643560202644514)
            agentsPos.append((ag, float(curr_xr), float(curr_yr), (0,0), (0,0), 0))
        _ = self.file.readline()
        return step, agentsPos
    
    def initialize_simulation(self):
# goal.xr="3.5" goal.yr="8.5"/>
# goal.xr="5.5" goal.yr="6.5"/>
# goal.xr="4.5" goal.yr="9.5"/>
# goal.xr="4.5" goal.yr="5.5"/>
# goal.xr="5.5" goal.yr="10.5"/>
#  goal.xr="5.5" goal.yr="4.5"/>
#  goal.xr="9.5" goal.yr="11.5"/>
        aux_goals = [(3.5, 8.5), (5.5, 6.5), (4.5, 9.5), (4.5, 5.5)]

        for agent_id in range(self.getNumAgents()):
            self.agent_goals[agent_id] = aux_goals[agent_id]
            self.notify_observers(GoalsProcessedMessage(
                step=-1, goals=self.agent_goals))

        agent_initialization_data = [
            {
                "agent_id": agent_id,
                "radius": 0.3,
                "max_speed": 1,
                "neighbor_dist": 1,
                "max_neighbors": 1,
                "time_horizon": 1,
                "time_horizon_obst": 1,
                "goal": (0,0),
                # Comportamiento del agente
                "behaviour": None
            }
            for agent_id in range(self.getNumAgents())
        ]
        self.notify_observers(SimulationInitializedMessage(
            step=-1, agent_initialization_data=agent_initialization_data))

def main():
    if len(sys.argv) != 2:
        print("Usage: python simulator.py <world_file.yaml>")
        sys.exit(1)
    
    # world_file = sys.argv[1]
    # try:
    #     with open(world_file, 'r') as stream:
    #         data = yaml.safe_load(stream)
    #         world_config = Simulation(**data['simulation'])

    # except FileNotFoundError:
    #     print(f"File {world_file} not found.")
    #     sys.exit(1)
    # except yaml.YAMLError as exc:
    #     print(f"Error reading YAML file: {exc}")
    #     sys.exit(1)
    # except ValidationError as exc:
    #     print(f"Validation error: {exc}")
    #     sys.exit(1)

    # window_width = int((world_config.map_settings.x_max - world_config.map_settings.x_min) * world_config.map_settings.cell_size)
    # window_height = int((world_config.map_settings.y_max - world_config.map_settings.y_min) * world_config.map_settings.cell_size)

    renderer = PyGameRenderer(
        # window_width,
        # window_height,
        # obstacles=[], goals={}, cell_size=int(world_config.map_settings.cell_size)
        1000,
        1000,
        obstacles=[], goals={}, cell_size=50
    )
    renderer.setup()

    # text_renderer = TextRenderer()
    # text_renderer.setup()
    
    # Inicializar el simulador y registrar el renderizador como observador
    rvo2_simulator = RVO2SimulatorWrapper()
    rvo2_simulator.register_observer(renderer)
    rvo2_simulator.register_obstacles()

    # rvo2_simulator.register_observer(text_renderer)
    # Registrar dinámicas desde el archivo YAML
    # for dynamic_config in world_config.dynamics:
    #     rvo2_simulator.register_dynamic(dynamic_config)

    # Inicializar y ejecutar la simulación
    # rvo2_simulator.initialize_simulation()

    # rvo2_simulator.run_pipeline(5000)  # Se asume 5000 pasos como ejemplo
    # rvo2_simulator.save_simulation_runs()

    rvo2_simulator.initialize_simulation()
    while True:
        rvo2_simulator.step()

if __name__ == "__main__":
    main()
