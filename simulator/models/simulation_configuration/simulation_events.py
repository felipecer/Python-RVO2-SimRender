from pydantic import BaseModel, ValidationError, model_validator
from typing import Tuple

from typing import Dict, Type

# Registro global para los tipos de eventos
EVENT_TYPES_REGISTRY: Dict[str, Type['SimulationEvent']] = {}

def register_event_type(cls=None, *, alias=None):
    def wrapper(cls):
        name = alias if alias else cls.__name__
        EVENT_TYPES_REGISTRY[name] = cls
        return cls

    if cls is None:
        return wrapper
    else:
        return wrapper(cls)

# Clase base para eventos de simulación
class SimulationEvent(BaseModel):
    @model_validator(mode='after')
    def validate_event_type(cls, values):
        if cls.__name__ not in EVENT_TYPES_REGISTRY:
            raise ValueError(f"Event type {cls.__name__} is not registered in EVENT_TYPES_REGISTRY.")
        return values    
    class Config:
        arbitrary_types_allowed = True

@register_event_type(alias="goal_reached")
class GoalReachedEvent(SimulationEvent):
    agent_id: int
    goal_position: Tuple[float, float]
    current_position: Tuple[float, float]
    step: int
