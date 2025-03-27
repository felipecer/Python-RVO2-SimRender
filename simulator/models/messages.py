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
class AgentState:
    agent_id: int
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    preferred_velocity: Tuple[float, float]
    distance_to_goal: float

@dataclass
class AllGoalsProcessedMessage(BaseMessage):
    goals: List[AgentGoal]

@dataclass
class SimulationInitializedMessage(BaseMessage):
    message: str = "simulation_initialized"
    agent_initialization_data: Optional[List[dict]] = None

@dataclass
class AgentsStateUpdateMessage(BaseMessage):
    # In addition to the position, we also include the velocity, preferred velocity, and the distance to the goal
    agent_state_list: List[AgentState]

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
