from typing import Any
from typing import Tuple

from pydantic import BaseModel


class AgentDefaults(BaseModel):
    neighbor_dist: float
    max_neighbors: float
    time_horizon: float
    time_horizon_obst: float
    radius: float
    max_speed: float
    velocity: Tuple[float, float] = (0.0, 0.0)


class GoalGroup(BaseModel):
    radius: float
    pattern: Any
