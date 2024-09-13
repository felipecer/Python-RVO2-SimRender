from typing import Tuple, Optional, Union
from pydantic import BaseModel

from typing import Optional, Union, Tuple

from simulator.models.simulation_configuration.distribution_patterns import DISTRIBUTION_PATTERNS_REGISTRY, CircleDistributionPattern, ExplicitDistributionPattern, LineDistributionPattern

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
    pattern: Union[LineDistributionPattern, CircleDistributionPattern, ExplicitDistributionPattern]

def instantiate_pattern(data):
    pattern_name = data.pop('name')
    if pattern_name in DISTRIBUTION_PATTERNS_REGISTRY:
        pattern_class = DISTRIBUTION_PATTERNS_REGISTRY[pattern_name]
        return pattern_class(**data)
    else:
        raise ValueError(f"Pattern {pattern_name} not found in registry.")

class AgentGroup(BaseModel):
    agent_defaults: Optional[AgentDefaults] = None
    pattern: Union[LineDistributionPattern, CircleDistributionPattern, ExplicitDistributionPattern]
    goals: Optional[GoalGroup] = None