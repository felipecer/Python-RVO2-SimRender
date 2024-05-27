from typing import List, Optional, Literal, Union
from pydantic import BaseModel
from agent import Agent
from obstacle import Obstacle

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
