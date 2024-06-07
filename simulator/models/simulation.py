from typing import List, Optional, Literal, Union
from pydantic import BaseModel
from agent import Agent
from obstacle import Obstacle
from pydantic import ValidationError
import yaml

class AgentDefaults(BaseModel):
    neighborDist: float
    maxNeighbors: float
    timeHorizon: float 
    timeHorizonObst: float 
    radius: float
    maxSpeed: float

class Simulation(BaseModel):
    timeStep: float
    agentDefaults: AgentDefaults
    agents: List[Agent]
    obstacles: List[Obstacle]


def main():
    # read data from simulationWorld.yaml
    with open("./simulator/models/simulationWorld.yaml", 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            print(data)
            simulation = Simulation(**data['Simulation'])
            print(simulation)
        except yaml.YAMLError as exc:
            print(exc)
        except ValidationError as exc:
            print(exc)

if __name__ == "__main__":
    main()