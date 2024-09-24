from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
from pydantic import BaseModel, model_validator
from typing import Optional, Tuple
from simulator.models.simulation_configuration.registry import global_registry

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
    behaviour: Optional[str] = None
    agent_defaults: Optional[AgentDefaults] = None
    pattern: Any
    goals: Optional[GoalGroup] = None

    @model_validator(mode='before')
    def resolve_behaviour_or_defaults(cls, values):
        behaviour_name = values.get('behaviour')
        agent_defaults = values.get('agent_defaults')

        if behaviour_name and agent_defaults:
            raise ValueError("Specify either 'behaviour' or 'agent_defaults', not both.")
        
        if behaviour_name:
            # Validar y obtener el behaviour del registro
            behaviour_instance = global_registry.get('behaviour', behaviour_name)
            if not behaviour_instance:
                raise ValueError(f"Behaviour '{behaviour_name}' not found in registry.")
            # Asignar los defaults desde la instancia del behaviour registrado
            values['agent_defaults'] = behaviour_instance.get_agent_params()
        
        # No se especificó ni agent_defaults ni behaviour, usar los valores globales por defecto
        if not agent_defaults and not behaviour_name:
            # Aquí podrías aplicar los agent_defaults globales si fuera necesario
            values['agent_defaults'] = values.get('global_agent_defaults')

        return values


# class AgentGroup(BaseModel):
#     behaviour: str
#     agent_defaults: Optional[AgentDefaults] = None
#     pattern: Any
#     goals: Optional[GoalGroup] = None    