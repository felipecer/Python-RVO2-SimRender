from typing import Tuple, List, Optional, Literal, Union
from pydantic import BaseModel

class Agent(BaseModel):
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    goal: Tuple[float, float]
    preferredVelocity: Tuple[float, float]
    maxSpeed: float