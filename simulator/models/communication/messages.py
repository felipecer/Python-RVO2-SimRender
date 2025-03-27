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
class AgentInitData:
    agent_id: int
    radius: float
    max_speed: float
    neighbor_dist: float
    max_neighbors: int
    time_horizon: float
    time_horizon_obst: float
    goal: Tuple[float, float]
    behaviour: Optional[str] = None

@dataclass
class Obstacle:
    vertices: List[Tuple[float, float]]

@dataclass
class AllGoalsProcessedMessage(BaseMessage):
    goals: List[AgentGoal]

@dataclass
class SimulationInitializedMessage(BaseMessage):
    agent_initialization_data: List[AgentInitData]
    obstacles: List[Obstacle]
    goals: List[AgentGoal]

@dataclass
class AgentsStateUpdateMessage(BaseMessage):
    # In addition to the position, we also include the velocity, preferred velocity, and the distance to the goal
    agent_state_list: List[AgentState]

@dataclass
class ObstaclesProcessedMessage(BaseMessage):
    obstacles: List[List[Tuple[float, float]]]

@dataclass
class GoalPositionUpdatedMessage(BaseMessage):
    goals: List[AgentGoal] 

@dataclass
class NewObstacleAddedMessage(BaseMessage):
    obstacle: List[Tuple[float, float]]

@dataclass
class RayHit:
    x: Optional[float]
    y: Optional[float]

@dataclass
class RayCastingUpdateMessage(BaseMessage):
    agent_id: int
    hits: List[RayHit]  # 360 expected

