from typing import List, Optional
from pydantic import BaseModel, ValidationError, model_validator
import yaml
from simulator.models.agent import AgentDefaults, AgentGroup
from simulator.models.simulation_configuration.registry import global_registry

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
    obstacles: Optional[List] = None
    dynamics: Optional[List] = None

    @staticmethod
    def validate_entities(category: str, entities_data: List[dict]):
        """Valida e instancia las entidades usando el registro global."""
        validated_entities = []
        for entity in entities_data:
            entity_type = entity.get('name')
            if entity_type is None:
                raise ValueError(f"Each {category} must have a 'name' field.")
            validated_entity = global_registry.instantiate(category=category, **entity)
            validated_entities.append(validated_entity)
        return validated_entities

    @model_validator(mode='before')
    def validate_dynamics(cls, values):
        dynamic_data = values.get('dynamics', [])
        values['dynamics'] = cls.validate_entities(category='dynamic', entities_data=dynamic_data)
        return values

    @model_validator(mode='before')
    def validate_obstacles(cls, values):
        obstacle_data = values.get('obstacles', [])
        values['obstacles'] = cls.validate_entities(category='shape', entities_data=obstacle_data)
        return values
    
    # @model_validator(mode='before')
    # def validate_agents(cls, values):
    #     agents_data = values.get('agents', [])
    #     validated_agents = []
    #     for agent in agents_data:
    #         # Validar patrón de distribución de agentes utilizando validate_entities
    #         pattern_data = agent.get('pattern')
    #         if pattern_data is None or 'name' not in pattern_data:
    #             raise ValueError("Each agent pattern must have a 'name' field.")
            
    #         # Usar validate_entities para validar los patrones de distribución
    #         pattern_instance = cls.validate_entities(category='distribution_pattern', entities_data=[pattern_data])[0]
    #         agent['pattern'] = pattern_instance
            
    #         validated_agents.append(agent)
        
    #     values['agents'] = validated_agents
    #     return values

    @model_validator(mode='before')
    def validate_agents(cls, values):
        agents_data = values.get('agents', [])
        validated_agents = []
        
        for agent in agents_data:
            # Validar patrón de distribución de agentes
            pattern_data = agent.get('pattern')
            if pattern_data is None or 'name' not in pattern_data:
                raise ValueError("Each agent pattern must have a 'name' field.")
            
            # Instanciar el patrón usando el global_registry
            pattern_instance = global_registry.instantiate(category='distribution_pattern', **pattern_data)
            agent['pattern'] = pattern_instance  # Asignamos el patrón correctamente

            # Si hay metas asociadas, validar y crear su patrón también
            if agent.get('goals'):
                goal_data = agent['goals']
                goal_pattern_data = goal_data.get('pattern')
                if goal_pattern_data is None or 'name' not in goal_pattern_data:
                    raise ValueError("Each goal pattern must have a 'name' field.")
                
                # Instanciar el patrón de la meta usando el registro global
                goal_pattern_instance = global_registry.instantiate(category='distribution_pattern', **goal_pattern_data)
                agent['goals']['pattern'] = goal_pattern_instance  # Asignamos la instancia correctamente
            
            validated_agents.append(agent)
        
        values['agents'] = validated_agents
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