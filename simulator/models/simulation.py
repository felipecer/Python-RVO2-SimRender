from typing import List, Union, Optional
from pydantic import BaseModel, ValidationError, model_validator
import yaml
from simulator.models.agent import AgentDefaults, AgentGroup, DISTRIBUTION_PATTERNS_REGISTRY
from pprint import pprint
from simulator.models.obstacle import RectangleShape, CircleShape, EquilateralTriangleShape, PolygonShape, OBSTACLE_SHAPES_REGISTRY

class MapSettings(BaseModel):
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    grid_resolution: int

class Simulation(BaseModel):
    time_step: float
    map_settings: Optional[MapSettings] = None
    agent_defaults: AgentDefaults
    agents: List[AgentGroup]
    obstacles: Optional[List[Union[RectangleShape, CircleShape, EquilateralTriangleShape, PolygonShape]]] = None

    @model_validator(mode='before')
    def validate_obstacles(cls, values):
        obstacle_data = values.get('obstacles', [])
        validated_obstacles = []
        for obstacle in obstacle_data:
            obstacle_type = obstacle.get('name')
            if obstacle_type in OBSTACLE_SHAPES_REGISTRY:
                obstacle_class = OBSTACLE_SHAPES_REGISTRY[obstacle_type]
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
            simulation = Simulation(**data['Simulation'])
            pprint(simulation.dict(), indent=2)  # Pretty print de la configuración de la simulación
            
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

        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")
        except ValidationError as exc:
            print(f"Validation error: {exc}")

if __name__ == "__main__":
    main()