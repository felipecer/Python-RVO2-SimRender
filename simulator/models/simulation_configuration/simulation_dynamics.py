from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, model_validator, PrivateAttr
from enum import Enum
from typing import Dict, List, Type, Tuple, Optional
import numpy as np
from simulator.models.messages import GoalPositionUpdatedMessage
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

# Registro global para las dinámicas de simulación
SIMULATION_DYNAMICS_REGISTRY: Dict[str, Type['SimulationDynamic']] = {}

def register_simulation_dynamic(cls=None, *, alias=None):
    def wrapper(cls):
        name = alias if alias else cls.__name__
        SIMULATION_DYNAMICS_REGISTRY[name] = cls
        return cls

    if cls is None:
        return wrapper
    else:
        return wrapper(cls)

class EventBasedDynamic(SimulationDynamic):
    event_type: str  # Define el tipo de evento que manejará esta dinámica

    def apply(self, event: Optional[SimulationEvent] = None):
        """Aplica la dinámica en respuesta a un evento."""
        if isinstance(event, GoalReachedEvent):
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
@register_simulation_dynamic(alias="goal_respawn")
class GoalRespawnDynamic(EventBasedDynamic):
    num_iterations: int = 20
    max_radius: float = 5.0
    step_radius: float = 1.0
    empty_radius: float = 1.5
    _generated_points: List[Tuple[float, float]] = PrivateAttr(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)

    def handle_event(self, event):
        if isinstance(event, GoalReachedEvent):
            new_goal = self._generate_new_goal()
            self._simulator.agent_goals[event.agent_id] = new_goal
            self._simulator.notify_observers(GoalPositionUpdatedMessage(
                step=event.step,
                goal_id=event.agent_id,
                new_position=new_goal
            ))

    def execute(self, event: GoalReachedEvent):
        print(f"Goal reached by agent {event.agent_id} at step {event.step}. Respawning goal.")
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

    def _generate_points(self):
        self._generated_points = []
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


@register_simulation_dynamic(alias="max_steps")
class MaxStepsReachedDynamic(OnStepDynamic):
    max_steps: int

    def execute(self):
        if self._simulator.current_step >= self.max_steps:
            print(f"Maximum steps of {self.max_steps} reached. Stopping simulation.")
            self._simulator.stop_simulation()

@register_simulation_dynamic(alias="log_step_info")
class LogStepInfoDynamic(OnStepDynamic):
    log_message: str = "Step executed"

    def execute(self):
        print(f"Step {self._simulator.current_step}: {self.log_message}")

@register_simulation_dynamic(alias="cleanup_resources")
class ResourceCleanupDynamic(OnceDynamic):
    def execute(self):
        print("Cleaning up resources and shutting down.")
        # Lógica para limpiar recursos