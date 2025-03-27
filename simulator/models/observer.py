from abc import ABC, abstractmethod
from typing import List

from simulator.models.communication.messages import (
    BaseMessage
)


class SimulationObserver(ABC):
    @abstractmethod
    def update(self, message: BaseMessage):
        pass

class SimulationObservable(ABC):
    @abstractmethod
    def register_observer(self, observer: SimulationObserver):
        pass

    @abstractmethod
    def remove_observer(self, observer: SimulationObserver):
        pass

    @abstractmethod
    def notify_observers(self, message: BaseMessage):
        pass

class SimulationSubject(SimulationObservable):
    def __init__(self):
        self._observers: List[SimulationObserver] = []

    def register_observer(self, observer: SimulationObserver):
        self._observers.append(observer)

    def remove_observer(self, observer: SimulationObserver):
        self._observers.remove(observer)

    def notify_observers(self, message: BaseMessage):
        for observer in self._observers:
            observer.update(message)
