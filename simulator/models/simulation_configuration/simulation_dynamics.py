from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Optional

import numpy as np
from pydantic import BaseModel, PrivateAttr

from simulator.models.messages import GoalPositionUpdatedMessage
from simulator.models.simulation_configuration.registry import register
from simulator.models.simulation_configuration.simulation_events import GoalReachedEvent, SimulationEvent


# Enum for the types of execution timings
class ExecutionTiming(Enum):
    BEFORE = "before"
    AFTER = "after"


class SimulationDynamic(BaseModel, ABC):
    name: str
    when: ExecutionTiming
    _simulator: Optional['SimulationEngine'] = PrivateAttr(
        None)  # Use of PrivateAttr to avoid validation

    def register_simulator(self, simulator):
        """Registers the simulator with the dynamic."""
        # Deferred import to avoid cycle
        from simulator.engines.base import SimulationEngine
        if not isinstance(simulator, SimulationEngine):
            raise ValueError(
                "Simulator must be an instance of SimulationEngine")
        self._simulator = simulator

    @abstractmethod
    def apply(self):
        """Method that must be implemented by each dynamic."""
        pass

    class Config:
        arbitrary_types_allowed = True


class EventBasedDynamic(SimulationDynamic):
    event_type: str  # Defines the type of event that this dynamic will handle

    def apply(self, event: Optional[SimulationEvent] = None):
        self.execute(event)

    @abstractmethod
    def execute(self, event: SimulationEvent):
        """Abstract method that must be implemented to handle the event."""
        pass


class OnStepDynamic(SimulationDynamic):
    # Executes every n steps, by default every step
    every_n_steps: Optional[int] = 1

    def apply(self):
        """Apply the dynamic depending on the current step."""
        if self.every_n_steps and self._simulator.current_step % self.every_n_steps == 0:
            self.execute()

    @abstractmethod
    def execute(self):
        """Abstract method that must be implemented by OnStep dynamics."""
        pass


class OnceDynamic(SimulationDynamic):
    _executed: bool = PrivateAttr(default=False)

    def apply(self):
        """Executes the dynamic only once if it has not been executed yet."""
        if not self._executed:
            self.execute()
            self._executed = True

    @abstractmethod
    def execute(self):
        """Abstract method that must be implemented by OneTime dynamics."""
        pass


class GoalSpawnerDynamic(EventBasedDynamic, ABC):
    _max_provisioned_points: int = 2000
    _generated_points: np.ndarray = PrivateAttr(
        default_factory=lambda: np.zeros((2000, 2)))
    _generated_count: int = PrivateAttr(
        default=0)  # Counter of generated points
    _current_index: int = PrivateAttr(default=0)

    def execute(self, event: GoalReachedEvent):
        new_goal = self._generate_new_goal()
        self._simulator.agent_goals[event.agent_id] = new_goal
        self._simulator.notify_observers(GoalPositionUpdatedMessage(
            step=event.step,
            goal_id=event.agent_id,
            new_position=new_goal
        ))

    def _generate_new_goal(self) -> Tuple[float, float]:
        if self._generated_count == 0 or self._current_index >= self._generated_count:
            self._generate_points()  # Generate points if there are no valid points
            self._current_index = 0  # Reset the index

        # Extract the current point and update the index
        self._current_index += 1  # Advance the index for the next point
        new_goal = tuple(self._generated_points[self._current_index, :])
        return new_goal

    @abstractmethod
    def _generate_points(self):
        pass


@register(alias="annulus_goal_spawner", category="dynamic")
class AnnulusGoalSpawnerDynamic(GoalSpawnerDynamic):
    density_factor: int = 20
    max_radius: float = 5.0
    step_radius: float = 1.0
    empty_radius: float = 1.5

    def _generate_points(self):
        point_index = 0
        current_radius = self.empty_radius
        while current_radius < self.max_radius:
            inner_radius = current_radius
            outer_radius = current_radius + self.step_radius
            annulus_area = np.pi * (outer_radius**2 - inner_radius**2)
            base_area = np.pi * self.step_radius**2
            num_points = int(self.density_factor * (annulus_area / base_area))

            # Generate points in the annulus
            points = self._generate_points_in_annulus(
                inner_radius, outer_radius, num_points)

            # Store the points in the preallocated array from the base class
            self._generated_points[point_index:point_index +
                                   num_points, :] = points
            point_index += num_points

            current_radius += self.step_radius
        # Adjust the size of the array to the number of generated points
        self._generated_count = point_index
        self._generated_points = self._generated_points[:self._generated_count]

    def _generate_points_in_annulus(self, inner_radius, outer_radius, num_points):
        rng = self._simulator.get_rng()  # Use the RNG from the SimulationEngine

        # Generate all radii and angles in a single call
        r = np.sqrt(rng.uniform(inner_radius**2, outer_radius**2, num_points))
        theta = rng.uniform(0, 2 * np.pi, num_points)

        # Convert to Cartesian coordinates in a single vectorized operation
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Return the points as an Nx2 matrix
        return np.column_stack((x, y))


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
        # Logic to clean up resources


@register(alias="update_initial_position_on_goal_reached", category="dynamic")
class UpdateInitialPositionOnGoalReachedDynamic(EventBasedDynamic):
    """
    This dynamic updates the agent's initial position when it reaches its goal,
    so that the next time the simulator is initialized, the agent starts
    from that reached goal.
    """

    def __init__(self, **data):
        super().__init__(**data)

    def execute(self, event: GoalReachedEvent):
        agent_id = event.agent_id
        current_position = event.current_position
        # Update the initial position in the simulator
        self._simulator.agent_initial_positions[agent_id] = current_position

@register(alias="stop_on_exit_area", category="dynamic")
class StopOnExitAreaDynamic(OnStepDynamic):
    boundary_x: float
    boundary_y: float
    agent_id: int
    def execute(self):
        x, y = self._simulator.get_agent_position(self.agent_id)
        if not (-self.boundary_x <= x <= self.boundary_x and -self.boundary_y <= y <= self.boundary_y):
            # print(f"Position: {x},{y} is outside the boundary x[-{self.boundary_x}:+{self.boundary_x}], y[-{self.boundary_y}:+{self.boundary_y}]")
            # print(f"Agent {self.agent_id} stepped outside the boundary. Stopping simulation.")
            self._simulator.stop_simulation()