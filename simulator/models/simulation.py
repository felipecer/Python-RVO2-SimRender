from typing import List, Union
from pydantic import BaseModel, ValidationError
import yaml
from agent import AgentDefaults, LineDistributionPattern, CircleDistributionPattern, ExplicitDistributionPattern, InsufficientPositionsError, DISTRIBUTION_PATTERNS_REGISTRY


class Simulation(BaseModel):
    timeStep: float
    agentDefaults: AgentDefaults
    agents: List[Union[LineDistributionPattern, CircleDistributionPattern, ExplicitDistributionPattern]]

    class Config:
        arbitrary_types_allowed = True

def main():
    # Leer datos desde el archivo simulationWorld.yaml
    with open("./simulator/models/simulationWorld.yaml", 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            print(data)
            
            # Preparar la lista de agentes para Pydantic
            agent_objects = []
            for agent_data in data['Simulation']['agents']:
                pattern_name = agent_data.get('pattern')  # Obtener el nombre del patrón sin eliminarlo
                
                if pattern_name in DISTRIBUTION_PATTERNS_REGISTRY:
                    pattern_class = DISTRIBUTION_PATTERNS_REGISTRY[pattern_name]
                    agent_object = pattern_class(**agent_data)  # pattern permanece en agent_data
                    agent_objects.append(agent_object)
                else:
                    print(f"Pattern {pattern_name} not found in registry.")

            # Crear la instancia de Simulation usando Pydantic
            simulation = Simulation(
                timeStep=data['Simulation']['timeStep'],
                agentDefaults=AgentDefaults(**data['Simulation']['agentDefaults']),
                agents=agent_objects
            )
            print(simulation)
            
            # Generar posiciones para cada grupo de agentes
            all_agents = []
            for agent_group in simulation.agents:
                agent_defaults = agent_group.agent_defaults or simulation.agentDefaults
                
                try:
                    positions = agent_group.generate_positions()
                    for position in positions:
                        agent = {
                            "position": position,
                            "maxSpeed": agent_defaults.maxSpeed,
                            "radius": agent_defaults.radius,
                            "timeHorizon": agent_defaults.timeHorizon,
                            "timeHorizonObst": agent_defaults.timeHorizonObst,
                            "maxNeighbors": agent_defaults.maxNeighbors,
                            "neighborDist": agent_defaults.neighborDist,
                            "velocity": agent_defaults.velocity
                        }
                        all_agents.append(agent)
                    print(f"Generated agents for {agent_group.__class__.__name__}: {all_agents}")
                except InsufficientPositionsError as e:
                    print(f"Error in pattern '{agent_group.__class__.__name__}': {e}")

        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")
        except ValidationError as exc:
            print(f"Validation error: {exc}")

if __name__ == "__main__":
    main()
