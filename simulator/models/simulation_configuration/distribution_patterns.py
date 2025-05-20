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

@register(alias='ring', category="distribution_pattern")
class RingDistributionPattern(SpatialDistributionPattern):
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
    
@register(alias='rectangle', category="distribution_pattern")
class RectangleDistributionPattern(SpatialDistributionPattern):
    """
    Distribution pattern that places agents inside a rectangle in a grid formation.
    Agents are evenly distributed throughout the rectangle area.
    """
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def generate_positions(self) -> List[Tuple[float, float]]:
        import math
        positions = []
        
        # Calculate rectangle dimensions
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        
        # Determine optimal grid dimensions
        # Try to make cells roughly square by maintaining aspect ratio
        aspect_ratio = width / height
        num_cols = math.ceil(math.sqrt(self.count * aspect_ratio))
        num_rows = math.ceil(self.count / num_cols)
        
        # Adjust if we end up with too many grid cells
        while num_rows * num_cols > self.count * 1.5:
            if num_cols > num_rows:
                num_cols -= 1
            else:
                num_rows -= 1
        
        # Calculate step sizes
        x_step = width / max(1, num_cols - 1) if num_cols > 1 else 0
        y_step = height / max(1, num_rows - 1) if num_rows > 1 else 0
        
        # Special case for single agent
        if self.count == 1:
            positions.append((self.x_min + width/2, self.y_min + height/2))
        else:
            # Generate grid positions
            count = 0
            for row in range(num_rows):
                for col in range(num_cols):
                    if count >= self.count:
                        break
                    
                    # Calculate position
                    if num_cols == 1:
                        x = self.x_min + width/2
                    else:
                        x = self.x_min + col * x_step
                        
                    if num_rows == 1:
                        y = self.y_min + height/2
                    else:
                        y = self.y_min + row * y_step
                        
                    positions.append((x, y))
                    count += 1
                    
                if count >= self.count:
                    break
        
        # Apply variability if specified
        if self.std_dev and self.std_dev > 0:
            positions = self.apply_variability(positions)
        
        return positions

    def apply_variability(self, positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        import random
        
        # Apply Gaussian noise to both coordinates
        return [(x + random.gauss(0, self.std_dev), y + random.gauss(0, self.std_dev)) 
                for x, y in positions]
    
@register(alias='rectangle', category="distribution_pattern")
class RectangleDistributionPattern(SpatialDistributionPattern):
    """
    Distribution pattern that places agents inside a rectangle in a grid formation.
    Agents are evenly distributed throughout the rectangle area.
    """
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def generate_positions(self) -> List[Tuple[float, float]]:
        import math
        positions = []
        
        # Calculate rectangle dimensions
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        
        # Determine optimal grid dimensions
        # Try to make cells roughly square by maintaining aspect ratio
        aspect_ratio = width / height
        num_cols = math.ceil(math.sqrt(self.count * aspect_ratio))
        num_rows = math.ceil(self.count / num_cols)
        
        # Adjust if we end up with too many grid cells
        while num_rows * num_cols > self.count * 1.5:
            if num_cols > num_rows:
                num_cols -= 1
            else:
                num_rows -= 1
        
        # Calculate step sizes
        x_step = width / max(1, num_cols - 1) if num_cols > 1 else 0
        y_step = height / max(1, num_rows - 1) if num_rows > 1 else 0
        
        # Special case for single agent
        if self.count == 1:
            positions.append((self.x_min + width/2, self.y_min + height/2))
        else:
            # Generate grid positions
            count = 0
            for row in range(num_rows):
                for col in range(num_cols):
                    if count >= self.count:
                        break
                    
                    # Calculate position
                    if num_cols == 1:
                        x = self.x_min + width/2
                    else:
                        x = self.x_min + col * x_step
                        
                    if num_rows == 1:
                        y = self.y_min + height/2
                    else:
                        y = self.y_min + row * y_step
                        
                    positions.append((x, y))
                    count += 1
                    
                if count >= self.count:
                    break
        
        # Apply variability if specified
        if self.std_dev and self.std_dev > 0:
            positions = self.apply_variability(positions)
        
        return positions

    def apply_variability(self, positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        import random
        
        # Apply Gaussian noise to both coordinates
        return [(x + random.gauss(0, self.std_dev), y + random.gauss(0, self.std_dev)) 
                for x, y in positions]