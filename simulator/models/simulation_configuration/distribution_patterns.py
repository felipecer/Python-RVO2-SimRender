from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from pydantic import BaseModel
from simulator.models.simulation_configuration.registry import register

class SpatialDistributionPattern(BaseModel, ABC):
    count: int    
    std_dev: Optional[float] = 0.0
    name: Optional[str] = None

    @abstractmethod
    def generate_positions(self) -> List[Tuple[float, float]]:  
        pass

@register(alias='line', category="distribution_pattern")
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
        
        # Apply variability in x if std_dev is greater than 0
        if self.std_dev > 0:
            positions = self.apply_variability(positions)
        
        return positions
    
    def apply_variability(self, positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        # Apply noise in the x direction
        import random
        return [(x + random.gauss(0, self.std_dev), y) for x, y in positions]

@register(alias='circle', category="distribution_pattern")
class CircleDistributionPattern(SpatialDistributionPattern):
    center: Tuple[float, float]
    radius: float
    start_position: str = "right"  # Options: "right", "top", "left", "bottom"

    def generate_positions(self) -> List[Tuple[float, float]]:
        import math
        positions = []
        
        # Determine the starting angle based on the chosen position
        start_angles = {
            "right": 0,          # x = center[0] + radius, y = center[1]
            "top": math.pi / 2,  # x = center[0], y = center[1] + radius
            "left": math.pi,      # x = center[0] - radius, y = center[1]
            "bottom": 3 * math.pi / 2  # x = center[0], y = center[1] - radius
        }
        
        start_angle = start_angles.get(self.start_position, 0)  # Default: "right"
        
        for i in range(self.count):
            angle = start_angle - 2 * math.pi * i / self.count  # Counterclockwise
            x = self.center[0] + self.radius * math.cos(angle)
            y = self.center[1] + self.radius * math.sin(angle)
            positions.append((x, y))
        
        if self.std_dev and self.std_dev > 0:
            positions = self.apply_variability(positions)
        
        return positions

    def apply_variability(self, positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        import random
        return [(x + random.gauss(0, self.std_dev), y + random.gauss(0, self.std_dev)) for x, y in positions]


class InsufficientPositionsError(Exception):
    pass

@register(alias='explicit', category="distribution_pattern")
class ExplicitDistributionPattern(SpatialDistributionPattern):
    positions: List[Tuple[float, float]]

    def generate_positions(self) -> List[Tuple[float, float]]:
        if len(self.positions) < self.count:
            raise InsufficientPositionsError(
                f"Expected {self.count} positions, but only {len(self.positions)} were provided."
            )
        return self.positions[:self.count]