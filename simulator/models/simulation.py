from typing import List, Union, Optional
from pydantic import BaseModel, ValidationError, model_validator
import yaml
from simulator.models.agent import AgentDefaults, AgentGroup
from pprint import pprint
from simulator.models.simulation_configuration.shapes import SHAPES_REGISTRY, Circle, EquilateralTriangle, Polygon, Rectangle
from simulator.models.simulation_configuration.simulation_dynamics import SIMULATION_DYNAMICS_REGISTRY

class MapSettings(BaseModel):
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    cell_size: int

class Simulation(BaseModel):
    time_step: float
    map_settings: Optional[MapSettings] = None
    agent_defaults: AgentDefaults
    agents: List[AgentGroup]
    obstacles: Optional[List[Union[Rectangle, Circle, EquilateralTriangle, Polygon]]] = None
    dynamics: Optional[List] = None

    @model_validator(mode='before')
    def validate_dynamics(cls, values):
        dynamic_data = values.get('dynamics', [])
        validated_dynamics = []
        for dynamic in dynamic_data:
            dynamic_type = dynamic.get('name')  # Cambiado a 'name'
            if dynamic_type is None:
                raise ValueError("Each dynamic must have a 'name' field.")
            if dynamic_type not in SIMULATION_DYNAMICS_REGISTRY.keys():
                print(f"Registro actual de dinámicas: {SIMULATION_DYNAMICS_REGISTRY.keys()}")
                raise ValueError(f"Unknown dynamic type: {dynamic_type}")
            dynamic_class = SIMULATION_DYNAMICS_REGISTRY[dynamic_type]
            validated_dynamic = dynamic_class(**dynamic)
            validated_dynamics.append(validated_dynamic)
        
        values['dynamics'] = validated_dynamics
        return values

    @model_validator(mode='before')
    def validate_obstacles(cls, values):
        obstacle_data = values.get('obstacles', [])
        validated_obstacles = []
        for obstacle in obstacle_data:
            obstacle_type = obstacle.get('name')
            if obstacle_type in SHAPES_REGISTRY:
                obstacle_class = SHAPES_REGISTRY[obstacle_type]
                validated_obstacle = obstacle_class(**obstacle)
                validated_obstacles.append(validated_obstacle)
            else:
                raise ValueError(f"Unknown obstacle type: {obstacle_type}")
    
        values['obstacles'] = validated_obstacles
        return values
    
    class Config:
        arbitrary_types_allowed = True

def main():
    # Leer datos desde el archivo simulationWorld.yaml
    with open("./simulator/models/simulationWorld.yaml", 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            # pprint(data, indent=2)  # Pretty print del YAML cargado
            
            # Crear la instancia de Simulation usando Pydantic
            simulation = Simulation(**data['simulation'])
            pprint(simulation.dict(), indent=4)  # Pretty print de la configuración de la simulación
            
            # Generar posiciones para cada grupo de agentes y sus metas
            for agent_group in simulation.agents:
                positions = agent_group.pattern.generate_positions()
                # print(f"Generated positions for {agent_group.pattern.__class__.__name__}:")
                # pprint(positions, indent=4)

                if agent_group.goals:
                    goal_positions = agent_group.goals.pattern.generate_positions()
                    # print(f"Generated goal positions for {agent_group.pattern.__class__.__name__}:")
                    # pprint(goal_positions, indent=4)

            # Generar formas para cada obstáculo
            if simulation.obstacles:
                for obstacle in simulation.obstacles:
                    shape = obstacle.generate_shape()
                    # print(f"Generated shape for {obstacle.name}:")
                    # pprint(shape, indent=4)

            # Verificar las dinámicas y sus tipos
            if simulation.dynamics:
                for dynamic in simulation.dynamics:
                    print(f"Dynamic '{dynamic.name}' parsed and registered.")

        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")
        except ValidationError as exc:
            print(f"Validation error: {exc}")

if __name__ == "__main__":
    main()