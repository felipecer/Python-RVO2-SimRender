from typing import Any, Tuple, Optional
from pydantic import BaseModel
from typing import Optional, Tuple

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

class AgentGroup(BaseModel):
    agent_defaults: Optional[AgentDefaults] = None
    pattern: Any
    goals: Optional[GoalGroup] = None