from pydantic import BaseModel
from typing import Tuple
from simulator.models.simulation_configuration.registry import register

class SimulationEvent(BaseModel):
    class Config:
        arbitrary_types_allowed = True

@register(alias="goal_reached", category="event")
class GoalReachedEvent(SimulationEvent):
    agent_id: int
    goal_position: Tuple[float, float]
    current_position: Tuple[float, float]
    step: int
    alias: str = "goal_reached"