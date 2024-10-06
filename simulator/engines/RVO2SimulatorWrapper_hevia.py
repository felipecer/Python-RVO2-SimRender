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
        self.file = open("/home/neopren/repos/Python-RVO2-SimRender/simulator/models/a.out", "r")
        self.numAgents = self.file.readline()
        self.current_step = -1
        _ = self.file.readline()

    def update(self, message):
        pass

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
        agentsPos = []
        stepLine=self.file.readline()
        if not stepLine:
            return
        step= stepLine.split()[1]
        for ag in range ( int(self.numAgents) ):
            coord=self.file.readline()
            curr_xr = coord.split()[0]
            curr_yr = coord.split()[1]
            agentsPos.append((ag, float(curr_xr), float(curr_yr)))
        _ = self.file.readline()
        return step, agentsPos

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
        obstacles=[], goals={}, cell_size=10
    )
    renderer.setup()

    # text_renderer = TextRenderer()
    # text_renderer.setup()
    
    # Inicializar el simulador y registrar el renderizador como observador
    rvo2_simulator = RVO2SimulatorWrapper()
    rvo2_simulator.register_observer(renderer)

    # rvo2_simulator.register_observer(text_renderer)
    # Registrar dinámicas desde el archivo YAML
    # for dynamic_config in world_config.dynamics:
    #     rvo2_simulator.register_dynamic(dynamic_config)

    # Inicializar y ejecutar la simulación
    # rvo2_simulator.initialize_simulation()

    # rvo2_simulator.run_pipeline(5000)  # Se asume 5000 pasos como ejemplo
    # rvo2_simulator.save_simulation_runs()

    while True:
        rvo2_simulator.step()

if __name__ == "__main__":
    main()
