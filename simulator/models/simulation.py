from typing import List, Union
from pydantic import BaseModel, ValidationError
import yaml
from agent import AgentDefaults, LineDistributionPattern, CircleDistributionPattern, ExplicitDistributionPattern, InsufficientPositionsError, AgentGroup, DISTRIBUTION_PATTERNS_REGISTRY
from pprint import pprint

class Simulation(BaseModel):
    timeStep: float
    agentDefaults: AgentDefaults
    agents: List[AgentGroup]

    class Config:
        arbitrary_types_allowed = True

def main():
    # Leer datos desde el archivo simulationWorld.yaml
    with open("./simulator/models/simulationWorld.yaml", 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            pprint(data, indent=2)  # Pretty print del YAML cargado
            
            # Crear la instancia de Simulation usando Pydantic
            simulation = Simulation(
                timeStep=data['Simulation']['timeStep'],
                agentDefaults=AgentDefaults(**data['Simulation']['agentDefaults']),
                agents=[AgentGroup(**group) for group in data['Simulation']['agents']]
            )
            pprint(simulation.dict(), indent=2)  # Pretty print de la configuración de la simulación
            
            # Generar posiciones para cada grupo de agentes y sus metas
            all_agents = []
            for agent_group in simulation.agents:
                pattern = agent_group.pattern  # Acceder al patrón del grupo de agentes
                agent_defaults = agent_group.agent_defaults or simulation.agentDefaults
                
                try:
                    positions = pattern.generate_positions()
                    print(f"Generated positions for {pattern.__class__.__name__}:")
                    pprint(positions, indent=4)
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
                    print(f"Generated agents for {pattern.__class__.__name__}:")
                    pprint(all_agents, indent=4)

                    # Imprimir las posiciones de las metas si están definidas
                    if agent_group.goals:
                        goal_positions = agent_group.goals.pattern.generate_positions()
                        print(f"Generated goal positions for {pattern.__class__.__name__}:")
                        pprint(goal_positions, indent=4)

                except InsufficientPositionsError as e:
                    print(f"Error in pattern '{pattern.__class__.__name__}': {e}")

        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")
        except ValidationError as exc:
            print(f"Validation error: {exc}")

if __name__ == "__main__":
    main()
