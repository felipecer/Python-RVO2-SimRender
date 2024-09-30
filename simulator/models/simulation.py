from typing import List, Optional
from pydantic import BaseModel, ValidationError, model_validator
import yaml
from simulator.models.AgentGroup import AgentGroup
from simulator.models.agent import AgentDefaults
from simulator.models.simulation_configuration.registry import global_registry
import argparse


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
            validated_entity = global_registry.instantiate(
                category=category, **entity)
            validated_entities.append(validated_entity)
        return validated_entities

    @model_validator(mode='before')
    def validate_dynamics(cls, values):
        dynamic_data = values.get('dynamics', [])
        values['dynamics'] = cls.validate_entities(
            category='dynamic', entities_data=dynamic_data)
        return values

    @model_validator(mode='before')
    def validate_obstacles(cls, values):
        obstacle_data = values.get('obstacles', [])
        values['obstacles'] = cls.validate_entities(
            category='shape', entities_data=obstacle_data)
        return values

    @model_validator(mode='before')
    def validate_events(cls, values):
        events_data = values.get('events', [])
        validated_events = []

        for event in events_data:
            event_type = event.get('name')
            if event_type is None:
                raise ValueError("Each event must have a 'name' field.")

            # Instanciar el evento desde el registro global
            event_instance = global_registry.instantiate(
                category='event', **event)
            validated_events.append(event_instance)

        values['events'] = validated_events
        return values

    @model_validator(mode='before')
    def validate_agents(cls, values):
        agents_data = values.get('agents', [])
        validated_agents = []

        for agent in agents_data:
            # Validar patrón de distribución de agentes
            pattern_data = agent.get('pattern')
            if pattern_data is None or 'name' not in pattern_data:
                raise ValueError(
                    "Each agent pattern must have a 'name' field.")

            # Instanciar el patrón usando el global_registry
            pattern_instance = global_registry.instantiate(
                category='distribution_pattern', **pattern_data)
            # Asignamos el patrón correctamente
            agent['pattern'] = pattern_instance

            # Si hay metas asociadas, validar y crear su patrón también
            if agent.get('goals'):
                goal_data = agent['goals']
                goal_pattern_data = goal_data.get('pattern')
                if goal_pattern_data is None or 'name' not in goal_pattern_data:
                    raise ValueError(
                        "Each goal pattern must have a 'name' field.")

                # Instanciar el patrón de la meta usando el registro global
                goal_pattern_instance = global_registry.instantiate(
                    category='distribution_pattern', **goal_pattern_data)
                # Asignamos la instancia correctamente
                agent['goals']['pattern'] = goal_pattern_instance

            validated_agents.append(agent)

        values['agents'] = validated_agents
        return values

    class Config:
        arbitrary_types_allowed = True


def main():
    # Parsear los argumentos de la línea de comandos
    parser = argparse.ArgumentParser(
        description="Simulador de Navegación de Agentes")
    parser.add_argument('world_file', type=str,
                        help='Archivo YAML de configuración del mundo')
    args = parser.parse_args()

    # Leer datos desde el archivo YAML proporcionado
    with open(args.world_file, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            print("Datos YAML cargados:")
            print(data['simulation'])  # Imprimir el contenido cargado del YAML

            # Crear la instancia de Simulation usando Pydantic
            print("Creando la instancia de Simulation...")
            simulation = Simulation(**data['simulation'])
            print("Instancia de Simulation creada con éxito.")

            # Generar posiciones para cada grupo de agentes y sus metas
            for agent_group in simulation.agents:
                print(
                    f"Procesando agente con comportamiento: {agent_group.behaviour}")
                positions = agent_group.pattern.generate_positions()
                print(
                    f"Posiciones generadas para el patrón {agent_group.pattern.__class__.__name__}:")
                print(positions)

                if agent_group.goals:
                    goal_positions = agent_group.goals.pattern.generate_positions()
                    print(
                        f"Posiciones de meta generadas para el patrón {agent_group.goals.pattern.__class__.__name__}:")
                    print(goal_positions)

            # Generar formas para cada obstáculo
            if simulation.obstacles:
                for obstacle in simulation.obstacles:
                    shape = obstacle.generate_shape()
                    print(
                        f"Forma generada para el obstáculo {obstacle.__class__.__name__}:")
                    print(shape)

            # Verificar las dinámicas y sus tipos
            if simulation.dynamics:
                for dynamic in simulation.dynamics:
                    print(f"Dynamic '{dynamic.name}' parsed and registered.")

        except FileNotFoundError:
            print(f"File {args.world_file} not found.")
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")
        except ValidationError as exc:
            print(f"Validation error: {exc}")


if __name__ == '__main__':
    main()
