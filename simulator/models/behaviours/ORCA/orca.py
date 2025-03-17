from abc import ABC, abstractmethod
import random
import threading
from typing import Any, ClassVar, Dict, List

import yaml
from pydantic import BaseModel, Field

from simulator.models.agent import AgentDefaults
from simulator.models.simulation_configuration.registry import global_registry


class Behaviour(BaseModel, ABC):
    name: str
    _agent_params: dict

    @abstractmethod
    def get_agent_params(self) -> Any:
        pass


class Behaviour(BaseModel, ABC):
    name: str
    agent_params: dict = Field(default_factory=dict)

    @abstractmethod
    def get_agent_params(self) -> Any:
        pass


class OrcaBehaviour(Behaviour):
    agent_defaults: AgentDefaults

    def __init__(self, **data):
        super().__init__(**data)
        self.agent_params = self.agent_defaults.dict()

    def get_agent_params(self) -> AgentDefaults:
        return self.agent_defaults


class EitherBehaviour(Behaviour):
    """
    A complex behavior that allows agents to adopt behaviors according to a specified distribution.
    This ensures that across many agents, the distribution of behaviors matches the specified weights.
    """
    behaviors: List[str]  # List of behavior names to choose from
    weights: List[float] = None  # Optional weights for behavior distribution
    
    # Class variables to track distribution across instances
    _behavior_counts: ClassVar[Dict[str, Dict[str, int]]] = {}
    _behavior_lock: ClassVar[threading.Lock] = threading.Lock()
    
    def __init__(self, **data):
        super().__init__(**data)
        self._name = data.get('name')
        
        # Initialize behavior counts for this pattern if not exists
        with EitherBehaviour._behavior_lock:
            if self._name not in EitherBehaviour._behavior_counts:
                EitherBehaviour._behavior_counts[self._name] = {b: 0 for b in self.behaviors}

    def get_agent_params(self) -> AgentDefaults:
        """
        Return agent parameters based on selecting a behavior from the available options.
        This distributes behaviors according to weights across multiple calls.
        """
        # Validate that all referenced behaviors exist
        for behavior_name in self.behaviors:
            behavior = global_registry.get('behaviour', behavior_name)
            if not behavior:
                raise ValueError(f"Behavior '{behavior_name}' referenced in EitherBehaviour '{self._name}' is not registered")
        
        # If weights are provided, ensure they match the behaviors list length
        if self.weights and len(self.weights) != len(self.behaviors):
            raise ValueError(f"Number of weights ({len(self.weights)}) must match number of behaviors ({len(self.behaviors)})")
            
        # Select behavior based on distributed strategy
        selected_behavior_name = self._select_next_behavior()
        print(f"EitherBehaviour '{self._name}' selected behavior: {selected_behavior_name}")
        
        behavior = global_registry.get('behaviour', selected_behavior_name)
        params = behavior.get_agent_params()
        print(f"Got params from {selected_behavior_name}: {params}")
        return params
    
    def _select_next_behavior(self) -> str:
        """
        Select the next behavior to maintain the distribution according to weights.
        Uses a greedy approach to select the most "underrepresented" behavior.
        """
        with EitherBehaviour._behavior_lock:
            weights = self.weights if self.weights else [1.0/len(self.behaviors)] * len(self.behaviors)
            
            # Get current counts
            counts = EitherBehaviour._behavior_counts[self._name]
            total_count = sum(counts.values())
            
            if total_count == 0:
                # First agent - just pick based on weights
                selected = random.choices(self.behaviors, weights=weights, k=1)[0]
            else:
                # Calculate current vs. target proportions
                target_props = {b: w for b, w in zip(self.behaviors, weights)}
                current_props = {b: counts[b]/total_count if total_count > 0 else 0 for b in self.behaviors}
                
                # Find the most underrepresented behavior
                diffs = {b: target_props[b] - current_props[b] for b in self.behaviors}
                selected = max(diffs.items(), key=lambda x: x[1])[0]
            
            # Update counts
            counts[selected] += 1
            return selected

def load_behaviours_from_yaml(yaml_file: str):
    with open(yaml_file, 'r') as file:
        try:
            data = yaml.safe_load(file)
            
            # Process standard ORCA behaviors first
            orca_behaviours = data.get('orcabehaviours', {})
            for name, params in orca_behaviours.items():
                behaviour_instance = OrcaBehaviour(
                    name=name, agent_defaults=AgentDefaults(**params))
                global_registry.register(
                    alias=name, category='behaviour', instance=behaviour_instance)
            
            # Process "either" behaviors after standard behaviors are registered
            either_behaviours = data.get('eitherbehaviours', {})
            if either_behaviours:
                for name, params in either_behaviours.items():
                    behaviour_instance = EitherBehaviour(
                        name=name, **params)
                    global_registry.register(
                        alias=name, category='behaviour', instance=behaviour_instance)
                # print(global_registry)

        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")


# Cargar y registrar behaviours al iniciar la simulación
load_behaviours_from_yaml('./simulator/models/behaviours/ORCA/behaviours.yaml')
