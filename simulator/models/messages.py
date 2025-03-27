from dataclasses import dataclass
from typing import List, Tuple
from typing import Optional


@dataclass
class BaseMessage:
    """
    Base class for all messages sent through the Observer pattern.
    """
    step: int

@dataclass
class AgentGoal:
    agent_id: int
    goal: Tuple[float, float]

@dataclass
class AllGoalsProcessedMessage(BaseMessage):
    goals: List[AgentGoal]

@dataclass
class SimulationInitializedMessage(BaseMessage):
    message: str = "simulation_initialized"
    agent_initialization_data: Optional[List[dict]] = None

@dataclass
class AgentPositionsUpdateMessage(BaseMessage):
    # In addition to the position, we also include the velocity, preferred velocity, and the distance to the goal
    agent_positions: List[Tuple[int, float, float,
                                Tuple[float, float], Tuple[float, float], float]]

@dataclass
class ObstaclesProcessedMessage(BaseMessage):
    obstacles: List[List[Tuple[float, float]]]

@dataclass
class GoalPositionUpdatedMessage(BaseMessage):
    goal_id: int
    new_position: Tuple[float, float]


@dataclass
class NewObstacleAddedMessage(BaseMessage):
    obstacle: List[Tuple[float, float]]

@dataclass
class RayCastingUpdateMessage(BaseMessage):
    intersections: List[Tuple[Optional[float], Optional[float]]]
