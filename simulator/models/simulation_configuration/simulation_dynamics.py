from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, model_validator, PrivateAttr
from enum import Enum
from typing import Dict, List, Type, Tuple, Optional
import numpy as np
from simulator.models.messages import GoalPositionUpdatedMessage
from simulator.models.simulation_configuration.registry import register
from simulator.models.simulation_configuration.simulation_events import GoalReachedEvent, SimulationEvent

# Enum para los tipos de ejecución de las dinámicas
class ExecutionTiming(Enum):
    BEFORE = "before"
    AFTER = "after"    

class SimulationDynamic(BaseModel, ABC):
    name: str
    when: ExecutionTiming
    _simulator: Optional['SimulationEngine'] = PrivateAttr(None)  # Uso de PrivateAttr para evitar la validación

    def register_simulator(self, simulator):
        """Registra el simulador con la dinámica."""
        # Importación diferida para evitar el ciclo
        from simulator.engines.base import SimulationEngine
        if not isinstance(simulator, SimulationEngine):
            raise ValueError("Simulator must be an instance of SimulationEngine")
        self._simulator = simulator

    @abstractmethod
    def apply(self):
        """Método que debe ser implementado por cada dinámica."""
        pass

    class Config:
        arbitrary_types_allowed = True

class EventBasedDynamic(SimulationDynamic):
    event_type: str  # Define el tipo de evento que manejará esta dinámica

    def apply(self, event: Optional[SimulationEvent] = None):
        self.execute(event)

    @abstractmethod
    def execute(self, event: SimulationEvent):
        """Método abstracto que debe ser implementado para manejar el evento."""
        pass

class OnStepDynamic(SimulationDynamic):
    every_n_steps: Optional[int] = 1  # Se ejecuta cada n pasos, por defecto en cada paso

    def apply(self):
        """Aplicar la dinámica dependiendo del step actual."""
        if self.every_n_steps and self._simulator.current_step % self.every_n_steps == 0:
            self.execute()

    @abstractmethod
    def execute(self):
        """Método abstracto que debe ser implementado por dinámicas OnStep."""
        pass

class OnceDynamic(SimulationDynamic):
    _executed: bool = PrivateAttr(default=False)

    def apply(self):
        """Ejecuta la dinámica una sola vez si no ha sido ejecutada aún."""
        if not self._executed:
            self.execute()
            self._executed = True

    @abstractmethod
    def execute(self):
        """Método abstracto que debe ser implementado por las dinámicas OneTime."""
        pass
    
# @register(alias="goal_respawn", category="dynamic")
# class GoalRespawnDynamic(EventBasedDynamic):
#     num_iterations: int = 20
#     max_radius: float = 5.0
#     step_radius: float = 1.0
#     empty_radius: float = 1.5
#     _generated_points: List[Tuple[float, float]] = PrivateAttr(default_factory=list)

#     def __init__(self, **data):
#         super().__init__(**data)

#     def execute(self, event: GoalReachedEvent):
#         new_goal = self._generate_new_goal()
#         self._simulator.agent_goals[event.agent_id] = new_goal
#         self._simulator.notify_observers(GoalPositionUpdatedMessage(
#             step=event.step,
#             goal_id=event.agent_id,
#             new_position=new_goal
#         ))

#     def _generate_new_goal(self) -> Tuple[float, float]:
#         if not self._generated_points:
#             self._generate_points()
#         return self._generated_points.pop(0)

#     def _generate_points_in_annulus(self, inner_radius, outer_radius, num_points):
#         points = []
#         rng = self._simulator.get_rng()  # Usar el RNG del SimulationEngine
#         while len(points) < num_points:
#             r = np.sqrt(rng.uniform(inner_radius**2, outer_radius**2))
#             theta = rng.uniform(0, 2 * np.pi)
#             x = r * np.cos(theta)
#             y = r * np.sin(theta)
#             points.append((x, y))
#         return points

#     def _generate_points(self):
#         self._generated_points = []
#         current_radius = self.empty_radius
#         while current_radius < self.max_radius:
#             inner_radius = current_radius
#             outer_radius = current_radius + self.step_radius
#             annulus_area = np.pi * (outer_radius**2 - inner_radius**2)
#             base_area = np.pi * self.step_radius**2
#             num_points = int(self.num_iterations * (annulus_area / base_area))
#             points = self._generate_points_in_annulus(
#                 inner_radius, outer_radius, num_points)
#             self._generated_points.extend(points)
#             current_radius += self.step_radius

class GoalSpawnerDynamic(EventBasedDynamic, ABC):
    _generated_points: List[Tuple[float, float]] = PrivateAttr(default_factory=list)

    def execute(self, event: GoalReachedEvent):
        new_goal = self._generate_new_goal()
        self._simulator.agent_goals[event.agent_id] = new_goal
        self._simulator.notify_observers(GoalPositionUpdatedMessage(
            step=event.step,
            goal_id=event.agent_id,
            new_position=new_goal
        ))

    def _generate_new_goal(self) -> Tuple[float, float]:
        if not self._generated_points:
            self._generate_points()
        return self._generated_points.pop(0)

    @abstractmethod
    def _generate_points(self):
        """Método abstracto para ser implementado por subclases específicas."""
        pass

@register(alias="annulus_goal_spawner", category="dynamic")
class AnnulusGoalSpawnerDynamic(GoalSpawnerDynamic):
    num_iterations: int = 20
    max_radius: float = 5.0
    step_radius: float = 1.0
    empty_radius: float = 1.5

    def _generate_points(self):
        current_radius = self.empty_radius
        while current_radius < self.max_radius:
            inner_radius = current_radius
            outer_radius = current_radius + self.step_radius
            annulus_area = np.pi * (outer_radius**2 - inner_radius**2)
            base_area = np.pi * self.step_radius**2
            num_points = int(self.num_iterations * (annulus_area / base_area))
            points = self._generate_points_in_annulus(
                inner_radius, outer_radius, num_points)
            self._generated_points.extend(points)
            current_radius += self.step_radius

    def _generate_points_in_annulus(self, inner_radius, outer_radius, num_points):
        points = []
        rng = self._simulator.get_rng()  # Usar el RNG del SimulationEngine
        while len(points) < num_points:
            r = np.sqrt(rng.uniform(inner_radius**2, outer_radius**2))
            theta = rng.uniform(0, 2 * np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append((x, y))
        return points

@register(alias="max_steps", category="dynamic")
class MaxStepsReachedDynamic(OnStepDynamic):
    max_steps: int

    def execute(self):
        if self._simulator.current_step >= self.max_steps:
            # print(f"Maximum steps of {self.max_steps} reached. Stopping simulation.")
            self._simulator.stop_simulation()

@register(alias="log_step_info", category="dynamic")
class LogStepInfoDynamic(OnStepDynamic):
    log_message: str = "Step executed"

    def execute(self):
        pass
        # print(f"Step {self._simulator.current_step}: {self.log_message}")

@register(alias="cleanup_resources", category="dynamic")
class ResourceCleanupDynamic(OnceDynamic):
    def execute(self):
        print("Cleaning up resources and shutting down.")
        # Lógica para limpiar recursos

@register(alias="update_initial_position_on_goal_reached", category="dynamic")
class UpdateInitialPositionOnGoalReachedDynamic(EventBasedDynamic):
    """
    Esta dinámica actualiza la posición inicial del agente cuando alcanza su meta,
    de modo que la próxima vez que se inicialice el simulador, el agente comience
    desde esa meta alcanzada.
    """
    def __init__(self, **data):
        super().__init__(**data)

    def execute(self, event: GoalReachedEvent):
        agent_id = event.agent_id
        current_position = event.current_position
        # Actualizamos la posición inicial en el simulador
        self._simulator.agent_initial_positions[agent_id] = current_position
        
