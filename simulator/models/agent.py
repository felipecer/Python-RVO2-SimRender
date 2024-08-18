from typing import Tuple, List, Optional, Literal, Union
from pydantic import BaseModel, model_validator

from abc import ABC, abstractmethod
from typing import List, Optional, Literal, Union, Tuple

DISTRIBUTION_PATTERNS_REGISTRY = {}

def register_distribution_pattern(cls):
    DISTRIBUTION_PATTERNS_REGISTRY[cls.__name__] = cls
    return cls

class AgentDefaults(BaseModel):
    neighborDist: float
    maxNeighbors: float
    timeHorizon: float 
    timeHorizonObst: float 
    radius: float
    maxSpeed: float
    velocity: Tuple[float, float] = (0.0, 0.0)

class SpatialDistributionPattern(BaseModel, ABC):
    count: int
    agent_defaults: Optional[AgentDefaults] = None
    std_dev: Optional[float] = 0.0
    pattern: Optional[str] = None

    @abstractmethod
    def generate_positions(self) -> List[Tuple[float, float]]:  
        pass

    @model_validator(mode='after')
    def validate_pattern(cls, values):
        # Asignar pattern basado en el nombre de la clase si no está presente
        if not values.pattern:
            values.pattern = cls.__name__

        print("Values during validation:", values)  # Aquí deberías ver 'pattern'
        print("Registry:", DISTRIBUTION_PATTERNS_REGISTRY)
        
        if values.pattern not in DISTRIBUTION_PATTERNS_REGISTRY:
            raise ValueError(f"Pattern {values.pattern} is not registered.")
        
        return values

@register_distribution_pattern
class LineDistributionPattern(SpatialDistributionPattern):
    x_value: float
    y_start: float
    y_end: float

    def generate_positions(self) -> List[Tuple[float, float]]:
        count = self.count
        positions = []
        y_step = (self.y_end - self.y_start) / (count - 1)

        for i in range(count):
            y = self.y_start + i * y_step
            positions.append((self.x_value, y))
        
        # Aplica la variabilidad en x si std_dev es mayor a 0
        if self.std_dev > 0:
            positions = self.apply_variability(positions)
        
        return positions
    
    def apply_variability(self, positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        # Aplica ruido en la dirección x
        import random
        return [(x + random.gauss(0, self.std_dev), y) for x, y in positions]

@register_distribution_pattern
class CircleDistributionPattern(SpatialDistributionPattern):
    center: Tuple[float, float]
    radius: float

    def generate_positions(self) -> List[Tuple[float, float]]:
        import math
        positions = []
        for i in range(self.count):
            angle = 2 * math.pi * i / self.count
            x = self.center[0] + self.radius * math.cos(angle)
            y = self.center[1] + self.radius * math.sin(angle)
            positions.append((x, y))
        
        if self.std_dev and self.std_dev > 0:
            positions = self.apply_variability(positions)
        
        return positions

    def apply_variability(self, positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        import random
        return [(x + random.gauss(0, self.std_dev), y) for x, y in positions]

class InsufficientPositionsError(Exception):
    pass

@register_distribution_pattern
class ExplicitDistributionPattern(SpatialDistributionPattern):
    positions: List[Tuple[float, float]]

    def generate_positions(self) -> List[Tuple[float, float]]:
        if len(self.positions) < self.count:
            raise InsufficientPositionsError(
                f"Expected {self.count} positions, but only {len(self.positions)} were provided."
            )
        return self.positions[:self.count]

    
