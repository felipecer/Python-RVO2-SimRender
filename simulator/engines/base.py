from abc import abstractmethod, ABC
from typing import Callable, List, Dict, Tuple
from enum import Enum

import numpy as np
from simulator.models.simulation_configuration.simulation_dynamics import EventBasedDynamic, ExecutionTiming, OnceDynamic

class DynamicsQueueManager:
    def __init__(self):
        self.initial_tasks: List[Callable] = []        # Tasks that run once at the beginning
        self.cycle_dynamics_before: List[Callable] = []  # Dynamics that run in each cycle before the step
        self.cycle_dynamics_after: List[Callable] = []   # Dynamics that run in each cycle after the step
        self.on_demand_before: List[Callable] = []       # On-demand dynamics before the step (removed after execution)
        self.on_demand_after: List[Callable] = []        # On-demand dynamics after the step (removed after execution)
        self.final_tasks: List[Callable] = []          # Tasks that run once at the end

    def add_dynamic(self, dynamic: Callable, timing: ExecutionTiming, on_demand: bool = False):        
        """Adds a dynamic to the corresponding list."""
        # print(dynamic)
        if timing == ExecutionTiming.BEFORE:
            if on_demand:
                self.on_demand_before.append(dynamic)
            else:
                self.cycle_dynamics_before.append(dynamic)
        elif timing == ExecutionTiming.AFTER:
            if on_demand:
                self.on_demand_after.append(dynamic)
            else:
                self.cycle_dynamics_after.append(dynamic)

    def add_once_dynamic(self, dynamic: Callable, timing: ExecutionTiming):
        """Adds dynamics that run once at the beginning or end."""
        if timing == ExecutionTiming.BEFORE:
            self.initial_tasks.append(dynamic)
        elif timing == ExecutionTiming.AFTER:
            self.final_tasks.append(dynamic)

    def run_initial_tasks(self):
        """Executes all initial dynamics."""
        for task in self.initial_tasks:
            task()

    def run_before_step_dynamics(self):
        """Executes all dynamics that should run before each step."""
        # Execute dynamics that run in each cycle
        for dynamic in self.cycle_dynamics_before:
            dynamic()
        
        # Execute all on-demand dynamics, without removing them yet
        for dynamic in self.on_demand_before:
            dynamic()

        # Now that all dynamics have been executed, clear the list
        self.on_demand_before.clear()

    def run_after_step_dynamics(self):
        """Executes all dynamics that should run after each step."""
        # Execute dynamics that run in each cycle
        for dynamic in self.cycle_dynamics_after:
            dynamic()

        # Execute and remove on-demand dynamics
        while self.on_demand_after:
            dynamic = self.on_demand_after.pop(0)
            dynamic()

    def run_final_tasks(self):
        """Executes all final tasks at the end of the simulation."""
        for task in self.final_tasks:
            task()

class SimulationState(Enum):
    SETUP = "setup"
    RUNNING = "running"
    PAUSED = "paused"  # We will leave this for later
    STOPPED = "stopped"

class SimulationEngine(ABC):
    _default_seed = 11
    def __init__(self, seed: int = None):
        self._dynamics_manager = DynamicsQueueManager()
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._state = SimulationState.SETUP
        self.current_step: int = 0
        self._seed = seed if seed is not None else self._default_seed  # Use the provided seed or the default
        self._random_number_generator = np.random.default_rng(self._seed)  # RNG initialization

    def reset_rng_with_seed(self, seed: int = None):
        """Resets the random number generator with a new seed."""
        if seed is None:
            seed = self._default_seed  # Use the default seed if none is provided
        self._seed = seed
        self._random_number_generator = np.random.default_rng(seed)
    
    def get_seed(self) -> int:
        """Returns the current seed used by the RNG."""
        return self._seed

    def get_rng(self):
        """Returns the current random number generator."""
        return self._random_number_generator

    def register_dynamic(self, dynamic):
        """Registers a dynamic to be executed during the simulation."""
        if self._state != SimulationState.SETUP:
            raise RuntimeError("Cannot register dynamics after setup is complete.")

        # Associate the simulator with the dynamic
        dynamic.register_simulator(self)

        # Handling dynamics that run once
        if isinstance(dynamic, OnceDynamic):
            self._dynamics_manager.add_once_dynamic(dynamic.apply, dynamic.when)
        # Handling event-based dynamics
        elif isinstance(dynamic, EventBasedDynamic):
            # print(f"Registering event handler {dynamic.name} for event type: {dynamic.event_type}")
            # Register the event handler but do not enqueue it yet
            self.register_event_handler(dynamic.event_type, dynamic.apply, dynamic.when)
        # Handling dynamics that run in each cycle (OnStepDynamic or similar)
        else:
            self._dynamics_manager.add_dynamic(dynamic.apply, dynamic.when)

    def register_event_handler(self, event_type: str, handler: Callable, when: ExecutionTiming = ExecutionTiming.BEFORE):   
        """Allows a dynamic to subscribe to a specific event."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append((handler, when))

    def handle_event(self, event_type: str, *args, **kwargs):
        """Handles a received event and enqueues the corresponding actions."""
        handlers = self._event_handlers.get(event_type, [])
        for handler, timing in handlers:
            # Capture the handler and arguments in a function so they are not overwritten
            def make_action(handler, *args, **kwargs):
                return lambda: handler(*args, **kwargs)

            # Enqueue the action with the correct context
            action = make_action(handler, *args, **kwargs)
            self._dynamics_manager.add_dynamic(action, timing, on_demand=True)


    @abstractmethod
    def initialize_simulation(self):
        """Initializes the simulation, creating agents, goals, and obstacles."""
        pass

    @abstractmethod
    def reset(self):
        """Resets the simulation to its initial state."""
        pass

    @abstractmethod
    def step(self):
        """Executes a step in the simulation."""
        pass

    @abstractmethod
    def run_simulation(self, step: int):
        """Runs the simulation cycle for a specified number of steps."""
        pass

    def stop_simulation(self):
        """Stops the simulation."""
        self._state = SimulationState.STOPPED
        self._dynamics_manager.run_final_tasks()  # Execute final tasks when stopping the simulation

    @abstractmethod
    def get_agent_position(self, agent_id) -> Tuple[float, float]:
        """Returns the current position of the agent."""
        pass

    @abstractmethod
    def get_agent_positions(self) -> Dict[int, Tuple[float, float]]:
        """Returns the current positions of the agents."""
        pass

    @abstractmethod
    def get_goal(self, agent_id: int) -> Tuple[float, float]:
        """Returns the current goal of an agent given its ID."""
        pass

    @abstractmethod
    def set_goal(self, agent_id: int, goal: Tuple[float, float]) -> None:
        """Adds or updates the goal of the agent given its ID."""
        pass

    @abstractmethod
    def is_goal_reached(self, agent_id: int) -> bool:
        """Checks if an agent has reached its goal."""
        pass

    def execute_simulation_step(self):
        """
        Executes a simulation step, including dynamics and agent updates.
        
        Args:
            step (int): The current step number in the simulation.
        """
        # Execute dynamics before the step
        self._dynamics_manager.run_before_step_dynamics()
        
        # Execute the simulation step
        # self.run_simulation(1)
        self.step()
        
        # Execute dynamics after the step
        self._dynamics_manager.run_after_step_dynamics()
    
    def run_pipeline(self, steps: int):
        if self._state != SimulationState.SETUP:
            raise RuntimeError("Simulation can only start from the SETUP state.")
        
        # Initialize the simulation
        self.initialize_simulation()

        # Execute initial dynamics (before + once)
        self._dynamics_manager.run_initial_tasks()

        # Transition to the RUNNING state after executing initial tasks
        self._state = SimulationState.RUNNING

        for step in range(steps):
            self.current_step = step
            if self._state == SimulationState.STOPPED:
                break
            
            # Execute a complete simulation step
            self.execute_simulation_step()

            # Check if the state has changed to STOPPED during the dynamics
            if self._state == SimulationState.STOPPED:
                break

        # If the simulation was not stopped, stop it at the end of the pipeline
        if self._state != SimulationState.STOPPED:
            self.stop_simulation()

    def get_state(self) -> SimulationState:
        """Returns the current state of the simulation."""
        return self._state
