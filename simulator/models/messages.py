from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class SimulationInitializedMessage:
    message: str = "simulation_initialized"

@dataclass
class AgentPositionsUpdateMessage:
    step: int
    agent_positions: List[Tuple[int, float, float]]

@dataclass
class ObstaclesProcessedMessage:
    obstacles: List[List[Tuple[float, float]]]

@dataclass
class GoalsProcessedMessage:
    goals: dict

@dataclass
class GoalPositionUpdatedMessage:
    goal_id: int
    new_position: Tuple[float, float]

@dataclass
class NewObstacleAddedMessage:
    obstacle: List[Tuple[float, float]]
