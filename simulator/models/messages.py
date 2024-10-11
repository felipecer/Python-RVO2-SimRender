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
    agent_initialization_data: Optional[List[dict]] = None


@dataclass
class AgentPositionsUpdateMessage(BaseMessage):
    # Además de la posición, también incluimos la velocidad, velocidad preferida, y la distancia a la meta
    agent_positions: List[Tuple[int, float, float,
                                Tuple[float, float], Tuple[float, float], float]]


@dataclass
class ObstaclesProcessedMessage(BaseMessage):
    obstacles: List[List[int]]
    # obstacles: List[List[Tuple[float, float]]]


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
