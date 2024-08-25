from dataclasses import dataclass
from typing import List, Tuple, Optional
from typing import Optional

@dataclass
class BaseMessage:
    """
    Clase base para todos los mensajes enviados a través del patrón Observer.
    """
    step: int 

@dataclass
class SimulationInitializedMessage(BaseMessage):
    message: str = "simulation_initialized"

@dataclass
class AgentPositionsUpdateMessage(BaseMessage):
    agent_positions: List[Tuple[int, float, float]]

@dataclass
class ObstaclesProcessedMessage(BaseMessage):
    obstacles: List[List[Tuple[float, float]]]

@dataclass
class GoalsProcessedMessage(BaseMessage):
    goals: dict

@dataclass
class GoalPositionUpdatedMessage(BaseMessage):
    goal_id: int
    new_position: Tuple[float, float]

@dataclass
class NewObstacleAddedMessage(BaseMessage):
    obstacle: List[Tuple[float, float]]
