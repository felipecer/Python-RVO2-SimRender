from abc import ABC, abstractmethod
from typing import Any
from simulator.models.agent import AgentDefaults
import json
import yaml
from pydantic import BaseModel, ValidationError, Field
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
    
def load_behaviours_from_yaml(yaml_file: str):
    with open(yaml_file, 'r') as file:
        try:
            data = yaml.safe_load(file)
            behaviours = data.get('orcabehaviours', {})

            for name, params in behaviours.items():
                print(f"Loading behaviour: {name}")
                # Instanciar OrcaBehaviour con los parámetros cargados
                behaviour = OrcaBehaviour(name=name, agent_defaults=AgentDefaults(**params))

                # Registrar el behaviour en el registro global
                global_registry.register(cls=behaviour.__class__, alias=name, category='behaviour')
        
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")

# Cargar y registrar behaviours al iniciar la simulación
load_behaviours_from_yaml('./simulator/models/behaviours/ORCA/behaviours.yaml')