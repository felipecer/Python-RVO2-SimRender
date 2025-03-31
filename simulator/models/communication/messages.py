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
    position: Tuple[float, float] = (0,0)

@dataclass
class Obstacle:
    vertices: List[Tuple[float, float]]

@dataclass
class AllGoalsProcessedMessage(BaseMessage):
    goals: List[AgentGoal]
    simulation_id: str

@dataclass
class SimulationInitializedMessage(BaseMessage):
    agent_initialization_data: List[AgentInitData]
    obstacles: List[Obstacle]
    goals: List[AgentGoal]
    simulation_id: str

@dataclass
class AgentsStateUpdateMessage(BaseMessage):
    # In addition to the position, we also include the velocity, preferred velocity, and the distance to the goal
    agent_state_list: List[AgentState]
    simulation_id: str

@dataclass
class ObstaclesProcessedMessage(BaseMessage):
    obstacles: List[List[Tuple[float, float]]]
    simulation_id: str

@dataclass
class GoalPositionUpdatedMessage(BaseMessage):
    goals: List[AgentGoal] 
    simulation_id: str

@dataclass
class NewObstacleAddedMessage(BaseMessage):
    obstacle: List[Tuple[float, float]]
    simulation_id: str

@dataclass
class RayHit:
    x: Optional[float]
    y: Optional[float]

@dataclass
class RayCastingUpdateMessage(BaseMessage):
    agent_id: int
    hits: List[RayHit]  # 360 expected
    simulation_id: str

@dataclass
class SimulationTerminatedMessage(BaseMessage):
    simulation_id: str
    reason: str = "Terminated"