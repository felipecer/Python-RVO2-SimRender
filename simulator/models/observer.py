from abc import ABC, abstractmethod
from typing import List
from simulator.models.messages import (
    AgentPositionsUpdateMessage,
    ObstaclesProcessedMessage,
    GoalsProcessedMessage,
    GoalPositionUpdatedMessage,
    NewObstacleAddedMessage,
    SimulationInitializedMessage
)

class SimulationObserver(ABC):
    @abstractmethod
    def update(self, message: SimulationInitializedMessage):
        pass

    @abstractmethod
    def obstacles_processed(self, message: ObstaclesProcessedMessage):
        pass

    @abstractmethod
    def goals_processed(self, message: GoalsProcessedMessage):
        pass

    @abstractmethod
    def goal_position_updated(self, message: GoalPositionUpdatedMessage):
        pass

    @abstractmethod
    def new_obstacle_added(self, message: NewObstacleAddedMessage):
        pass

        pass

class SimulationObservable(ABC):
    @abstractmethod
    def register_observer(self, observer: SimulationObserver):
        pass

    @abstractmethod
    def remove_observer(self, observer: SimulationObserver):
        pass

    @abstractmethod
    def notify_observers(self, message: SimulationInitializedMessage):
        pass

    @abstractmethod
    def notify_obstacles_processed(self, message: ObstaclesProcessedMessage):
        pass

    @abstractmethod
    def notify_goals_processed(self, message: GoalsProcessedMessage):
        pass

    @abstractmethod
    def notify_goal_position_updated(self, message: GoalPositionUpdatedMessage):
        pass

    @abstractmethod
    def notify_new_obstacle_added(self, message: NewObstacleAddedMessage):
        pass

class SimulationSubject(SimulationObservable):
    def __init__(self):
        self._observers: List[SimulationObserver] = []

    def register_observer(self, observer: SimulationObserver):
        self._observers.append(observer)

    def remove_observer(self, observer: SimulationObserver):
        self._observers.remove(observer)

    def notify_observers(self, message: SimulationInitializedMessage):
        for observer in self._observers:
            observer.update(message)

    def notify_obstacles_processed(self, message: ObstaclesProcessedMessage):
        for observer in self._observers:
            observer.obstacles_processed(message)

    def notify_goals_processed(self, message: GoalsProcessedMessage):
        for observer in self._observers:
            observer.goals_processed(message)

    def notify_goal_position_updated(self, message: GoalPositionUpdatedMessage):
        for observer in self._observers:
            observer.goal_position_updated(message)

    def notify_new_obstacle_added(self, message: NewObstacleAddedMessage):
        for observer in self._observers:
            observer.new_obstacle_added(message)
