from simulator.models.agent import AgentDefaults, GoalGroup
from simulator.models.simulation_configuration.registry import global_registry


from pydantic import BaseModel, model_validator


from typing import Any, Optional


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
            raise ValueError(
                "Specify either 'behaviour' or 'agent_defaults', not both.")

        if behaviour_name:
            # Validate and obtain the behaviour from the registry
            behaviour_instance = global_registry.get(
                'behaviour', behaviour_name)
            if not behaviour_instance:
                raise ValueError(
                    f"Behaviour '{behaviour_name}' not found in registry.")
            # Assign the defaults from the registered behaviour instance
            values['agent_defaults'] = behaviour_instance.get_agent_params()

        # Neither agent_defaults nor behaviour was specified, use global default values
        if not agent_defaults and not behaviour_name:
            # Here you could apply the global agent_defaults if necessary
            values['agent_defaults'] = values.get('global_agent_defaults')

        return values
