from typing import Tuple, List, Optional, Literal, Union
from pydantic import BaseModel

class Agent(BaseModel):
    maxNeighbors: int | None
    neighborDist: float | None
    radius: float
    timeHorizon: float | None
    timeHorizonObst: float | None
    position: Tuple[float, float]
    goal: Tuple[float, float]
    preferredVelocity: Tuple[float, float]
    maxSpeed: float | None