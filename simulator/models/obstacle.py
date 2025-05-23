# from typing import Tuple, List, Optional, Union
# from pydantic import BaseModel, model_validator
# from abc import ABC, abstractmethod

# # Registro global para las formas de obstáculos
# OBSTACLE_SHAPES_REGISTRY = {}

# def register_obstacle_shape(cls=None, *, alias=None):
#     def wrapper(cls):
#         name = alias if alias else cls.__name__
#         OBSTACLE_SHAPES_REGISTRY[name] = cls
#         return cls

#     if cls is None:
#         return wrapper
#     else:
#         return wrapper(cls)


# class ObstacleShape(BaseModel, ABC):
#     name: Optional[str] = None

#     @abstractmethod
#     def generate_shape(self) -> List[Tuple[float, float]]:
#         pass

#     @model_validator(mode='after')
#     def validate_shape(cls, values):
#         # Asignar name basado en el nombre de la clase si no está presente
#         if not values.name:
#             values.name = cls.__name__ 
        
#         if values.name not in OBSTACLE_SHAPES_REGISTRY:
#             raise ValueError(f"Shape {values.name} is not registered.")
        
#         return values

# # Registro con alias simplificados
# @register_obstacle_shape(alias="rectangle")
# class RectangleShape(ObstacleShape):
#     center: Tuple[float, float]
#     width: float
#     height: float

#     def generate_shape(self) -> List[Tuple[float, float]]:
#         cx, cy = self.center
#         w_half = self.width / 2
#         h_half = self.height / 2
#         return [
#             (cx - w_half, cy - h_half),
#             (cx + w_half, cy - h_half),
#             (cx + w_half, cy + h_half),
#             (cx - w_half, cy + h_half)
#         ]

# @register_obstacle_shape(alias="triangle")
# class EquilateralTriangleShape(ObstacleShape):
#     center: Tuple[float, float]
#     side_length: float

#     def generate_shape(self) -> List[Tuple[float, float]]:
#         import math
#         cx, cy = self.center
#         h = (math.sqrt(3) / 2) * self.side_length
#         return [
#             (cx, cy + (2/3) * h),  # Vértice superior
#             (cx - self.side_length / 2, cy - (1/3) * h),  # Vértice inferior izquierdo
#             (cx + self.side_length / 2, cy - (1/3) * h)   # Vértice inferior derecho
#         ]

# @register_obstacle_shape(alias="circle")
# class CircleShape(ObstacleShape):
#     center: Tuple[float, float]
#     radius: float

#     def generate_shape(self) -> List[Tuple[float, float]]:
#         import math
#         return [
#             (self.center[0] + math.cos(2 * math.pi * i / 36) * self.radius,
#              self.center[1] + math.sin(2 * math.pi * i / 36) * self.radius)
#             for i in range(36)  # Aproximación con 36 puntos
#         ]

# @register_obstacle_shape(alias="polygon")
# class PolygonShape(ObstacleShape):
#     vertices: List[Tuple[float, float]]

#     def generate_shape(self) -> List[Tuple[float, float]]:
#         return self.vertices
