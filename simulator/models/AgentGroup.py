from simulator.models.agent import AgentDefaults, GoalGroup
from simulator.models.simulation_configuration.registry import global_registry


from pydantic import BaseModel, model_validator, Field


from typing import Any, List, Optional


class AgentGroup(BaseModel):
    behaviour: Optional[str] = None
    agent_defaults: Optional[AgentDefaults] = None
    pattern: Any
    goals: Optional[GoalGroup] = None
    assigned_behaviors: List[str] = Field(default_factory=list)

    class Config:
        fields = {
            'assigned_behaviors': {'exclude': True}
        }

    @model_validator(mode='before')
    def resolve_behaviour_or_defaults(cls, values):
        pattern_obj = values.get('pattern')
        # Safely retrieve the count from the pattern object
        agent_count = getattr(pattern_obj, 'count', 0)

        behaviour_name = values.get('behaviour')
        agent_defaults = values.get('agent_defaults')

        if behaviour_name and agent_defaults:
            raise ValueError(
                "Specify either 'behaviour' or 'agent_defaults', not both.")

        if behaviour_name:
            behaviour_instance = global_registry.get('behaviour', behaviour_name)
            if not behaviour_instance:
                raise ValueError(f"Behaviour '{behaviour_name}' not found in registry.")
            agent_defaults = behaviour_instance.get_agent_params()
            values['agent_defaults'] = agent_defaults

            # Step 2: if it's a regular ORCA or any single behavior, fill assigned_behaviors
            # (We'll handle EitherBehaviour logic separately)
            from simulator.models.behaviours.ORCA.orca import EitherBehaviour
            if isinstance(behaviour_instance, EitherBehaviour):
                assigned = []
                for _ in range(agent_count):
                    # Generate a behavior name for each agent
                    chosen = behaviour_instance._select_next_behavior()
                    assigned.append(chosen)
                values['assigned_behaviors'] = assigned
                # print(assigned)
            else:
                values['assigned_behaviors'] = [behaviour_name] * agent_count

        # Neither agent_defaults nor behaviour was specified, use global default values
        if not agent_defaults and not behaviour_name:
            # print("Falling back to default behaviour")
            # Here you could apply the global agent_defaults if necessary
            values['agent_defaults'] = values.get('global_agent_defaults')
        return values
    
    # def update_with_behavior_params(self, agent_id, behavior_name):
    #     """Explicitly apply behavior parameters to an agent after creation"""
    #     if behavior_name:
    #         behavior = global_registry.get('behaviour', behavior_name)
    #         agent_defaults = behavior.get_agent_params()
            
    #         # Force apply these parameters to the agent
    #         agent = self.simulator.get_agent(agent_id)
    #         agent.max_speed = agent_defaults.max_speed
    #         agent.radius = agent_defaults.radius
    #         agent.time_horizon = agent_defaults.time_horizon
    #         agent.time_horizon_obst = agent_defaults.time_horizon_obst
    #         agent.max_neighbors = agent_defaults.max_neighbors
    #         agent.neighbor_dist = agent_defaults.neighbor_dist
    #         agent.velocity = agent_defaults.velocity
            
    #         print(f"Agent {agent_id} updated with {behavior_name} parameters")
