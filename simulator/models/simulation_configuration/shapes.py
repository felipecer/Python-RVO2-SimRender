from abc import ABC, abstractmethod
from typing import Tuple, List, Optional

from pydantic import BaseModel

from simulator.models.simulation_configuration.registry import register


class BaseShape(BaseModel, ABC):
    name: Optional[str] = None

    @abstractmethod
    def generate_shape(self) -> List[Tuple[float, float]]:
        pass

# Registro con alias simplificados
@register(alias="rectangle", category="shape")
class Rectangle(BaseShape):
    center: Tuple[float, float]
    width: float
    height: float

    def generate_shape(self) -> List[Tuple[float, float]]:
        cx, cy = self.center
        w_half = self.width / 2
        h_half = self.height / 2
        return [
            (cx - w_half, cy - h_half),
            (cx + w_half, cy - h_half),
            (cx + w_half, cy + h_half),
            (cx - w_half, cy + h_half)
        ]

@register(alias="triangle", category="shape")
class EquilateralTriangle(BaseShape):
    center: Tuple[float, float]
    side_length: float

    def generate_shape(self) -> List[Tuple[float, float]]:
        import math
        cx, cy = self.center
        h = (math.sqrt(3) / 2) * self.side_length
        return [
            (cx, cy + (2/3) * h),  # Vértice superior
            (cx - self.side_length / 2, cy - (1/3) * h),  # Vértice inferior izquierdo
            (cx + self.side_length / 2, cy - (1/3) * h)   # Vértice inferior derecho
        ]

@register(alias="circle", category="shape")
class Circle(BaseShape):
    center: Tuple[float, float]
    radius: float

    def generate_shape(self) -> List[Tuple[float, float]]:
        import math
        return [
            (self.center[0] + math.cos(2 * math.pi * i / 36) * self.radius,
             self.center[1] + math.sin(2 * math.pi * i / 36) * self.radius)
            for i in range(36)  # Aproximación con 36 puntos
        ]

@register(alias="polygon", category="shape")
class Polygon(BaseShape):
    vertices: List[Tuple[float, float]]

    def generate_shape(self) -> List[Tuple[float, float]]:
        return self.vertices
