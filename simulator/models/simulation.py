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
        """Validate and instantiate entities using the global registry."""
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

            # Instantiate the event from the global registry
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
            # Validate agent distribution pattern
            pattern_data = agent.get('pattern')
            if pattern_data is None or 'name' not in pattern_data:
                raise ValueError(
                    "Each agent pattern must have a 'name' field.")

            # Instantiate the pattern using the global_registry
            pattern_instance = global_registry.instantiate(
                category='distribution_pattern', **pattern_data)
            # Assign the pattern correctly
            agent['pattern'] = pattern_instance

            # If there are associated goals, validate and create their pattern as well
            if agent.get('goals'):
                goal_data = agent['goals']
                goal_pattern_data = goal_data.get('pattern')
                if goal_pattern_data is None or 'name' not in goal_pattern_data:
                    raise ValueError(
                        "Each goal pattern must have a 'name' field.")

                # Instantiate the goal pattern using the global registry
                goal_pattern_instance = global_registry.instantiate(
                    category='distribution_pattern', **goal_pattern_data)
                # Assign the instance correctly
                agent['goals']['pattern'] = goal_pattern_instance

            validated_agents.append(agent)

        values['agents'] = validated_agents
        return values

    class Config:
        arbitrary_types_allowed = True


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Agent Navigation Simulator")
    parser.add_argument('world_file', type=str,
                        help='YAML configuration file for the world')
    args = parser.parse_args()

    # Read data from the provided YAML file
    with open(args.world_file, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            print("YAML data loaded:")
            print(data['simulation'])  # Print the loaded YAML content

            # Create the Simulation instance using Pydantic
            print("Creating the Simulation instance...")
            simulation = Simulation(**data['simulation'])
            print("Simulation instance created successfully.")

            # Generate positions for each agent group and their goals
            for agent_group in simulation.agents:
                print(
                    f"Processing agent with behavior: {agent_group.behaviour}")
                positions = agent_group.pattern.generate_positions()
                print(
                    f"Positions generated for the pattern {agent_group.pattern.__class__.__name__}:")
                print(positions)

                if agent_group.goals:
                    goal_positions = agent_group.goals.pattern.generate_positions()
                    print(
                        f"Goal positions generated for the pattern {agent_group.goals.pattern.__class__.__name__}:")
                    print(goal_positions)

            # Generate shapes for each obstacle
            if simulation.obstacles:
                for obstacle in simulation.obstacles:
                    shape = obstacle.generate_shape()
                    print(
                        f"Shape generated for the obstacle {obstacle.__class__.__name__}:")
                    print(shape)

            # Verify dynamics and their types
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
