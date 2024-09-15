from abc import abstractmethod, ABC
from typing import Callable, List, Dict, Tuple
from enum import Enum

import numpy as np
from simulator.models.simulation_configuration.simulation_dynamics import EventBasedDynamic, ExecutionTiming, OnceDynamic

class DynamicsQueueManager:
    def __init__(self):
        self.initial_tasks: List[Callable] = []        # Tareas que se ejecutan una vez al inicio
        self.cycle_dynamics_before: List[Callable] = []  # Dinámicas que se ejecutan en cada ciclo antes del paso
        self.cycle_dynamics_after: List[Callable] = []   # Dinámicas que se ejecutan en cada ciclo después del paso
        self.on_demand_before: List[Callable] = []       # Dinámicas bajo demanda antes del paso (se eliminan tras ejecutarse)
        self.on_demand_after: List[Callable] = []        # Dinámicas bajo demanda después del paso (se eliminan tras ejecutarse)
        self.final_tasks: List[Callable] = []          # Tareas que se ejecutan una vez al final

    def add_dynamic(self, dynamic: Callable, timing: ExecutionTiming, on_demand: bool = False):
        """Agrega una dinámica a la lista correspondiente."""
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
        """Agrega dinámicas que se ejecutan una sola vez al inicio o al final."""
        if timing == ExecutionTiming.BEFORE:
            self.initial_tasks.append(dynamic)
        elif timing == ExecutionTiming.AFTER:
            self.final_tasks.append(dynamic)

    def run_initial_tasks(self):
        """Ejecuta todas las dinámicas iniciales."""
        for task in self.initial_tasks:
            task()

    def run_before_step_dynamics(self):
        """Ejecuta todas las dinámicas que deben correr antes de cada step."""
        # Ejecuta dinámicas que corren en cada ciclo
        for dynamic in self.cycle_dynamics_before:
            dynamic()

        # Ejecuta y elimina dinámicas bajo demanda
        while self.on_demand_before:
            dynamic = self.on_demand_before.pop(0)
            dynamic()

    def run_after_step_dynamics(self):
        """Ejecuta todas las dinámicas que deben correr después de cada step."""
        # Ejecuta dinámicas que corren en cada ciclo
        for dynamic in self.cycle_dynamics_after:
            dynamic()

        # Ejecuta y elimina dinámicas bajo demanda
        while self.on_demand_after:
            dynamic = self.on_demand_after.pop(0)
            dynamic()

    def run_final_tasks(self):
        """Ejecuta todas las tareas finales al terminar la simulación."""
        for task in self.final_tasks:
            task()

class SimulationState(Enum):
    SETUP = "setup"
    RUNNING = "running"
    PAUSED = "paused"  # Este lo dejaremos para después
    STOPPED = "stopped"

class SimulationEngine(ABC):
    _default_seed = 11
    def __init__(self, seed: int = None):
        self._dynamics_manager = DynamicsQueueManager()
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._state = SimulationState.SETUP
        self.current_step: int = 0
        self._seed = seed if seed is not None else self._default_seed  # Usa el seed proporcionado o el default
        self._random_number_generator = np.random.default_rng(self._seed)  # Inicialización del RNG

    def reset_rng_with_seed(self, seed: int = None):
        """Reinicia el generador de números aleatorios con un nuevo seed."""
        if seed is None:
            seed = self._default_seed  # Usa el seed por defecto si no se proporciona otro
        self._seed = seed
        self._random_number_generator = np.random.default_rng(seed)
    
    def get_seed(self) -> int:
        """Devuelve el seed actual utilizado por el RNG."""
        return self._seed

    def get_rng(self):
        """Devuelve el generador de números aleatorios actual."""
        return self._random_number_generator

    def register_dynamic(self, dynamic):
        """Registra una dinámica para que se ejecute durante la simulación."""
        if self._state != SimulationState.SETUP:
            raise RuntimeError("Cannot register dynamics after setup is complete.")

        # Asocia el simulador con la dinámica
        dynamic.register_simulator(self)

        # Manejo de dinámicas que se ejecutan una vez
        if isinstance(dynamic, OnceDynamic):
            self._dynamics_manager.add_once_dynamic(dynamic.apply, dynamic.when)
        # Manejo de dinámicas basadas en eventos
        elif isinstance(dynamic, EventBasedDynamic):
            # print(f"Registering event handler for event type: {dynamic.event_type}")
            # Registra el manejador de eventos pero no lo encola todavía
            self.register_event_handler(dynamic.event_type, dynamic.apply, dynamic.when)
        # Manejo de dinámicas que se ejecutan en cada ciclo (OnStepDynamic o similares)
        else:
            self._dynamics_manager.add_dynamic(dynamic.apply, dynamic.when)

    def register_event_handler(self, event_type: str, handler: Callable, when: ExecutionTiming = ExecutionTiming.BEFORE):   
        """Permite que una dinámica se suscriba a un evento específico."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append((handler, when))

    def handle_event(self, event_type: str, *args, **kwargs):
        """Maneja un evento recibido y encola las acciones correspondientes."""
        handlers = self._event_handlers.get(event_type, [])
        for handler, timing in handlers:
            action = lambda: handler(*args, **kwargs)
            self._dynamics_manager.add_dynamic(action, timing, on_demand=True)


    @abstractmethod
    def initialize_simulation(self):
        """Inicializa la simulación, creando agentes, metas, y obstáculos."""
        pass

    @abstractmethod
    def reset(self):
        """Reinicia la simulación a su estado inicial."""
        pass

    @abstractmethod
    def step(self):
        """Ejecuta un paso en la simulación."""
        pass

    @abstractmethod
    def run_simulation(self, step: int):
        """Ejecuta el ciclo de la simulación por un número de pasos especificado."""
        pass

    def stop_simulation(self):
        """Detiene la simulación."""
        self._state = SimulationState.STOPPED
        self._dynamics_manager.run_final_tasks()  # Ejecutar las tareas finales al detener la simulación

    @abstractmethod
    def get_agent_position(self, agent_id) -> Tuple[float, float]:
        """Devuelve la posicion actual del agente."""
        pass

    @abstractmethod
    def get_agent_positions(self) -> Dict[int, Tuple[float, float]]:
        """Devuelve las posiciones actuales de los agentes."""
        pass

    @abstractmethod
    def get_goal(self, agent_id: int) -> Tuple[float, float]:
        """Devuelve la meta actual de un agente dado su ID."""
        pass

    @abstractmethod
    def set_goal(self, agent_id: int, goal: Tuple[float, float]) -> None:
        """agrega o actualiza la meta del agente dado su id"""
        pass

    @abstractmethod
    def is_goal_reached(self, agent_id: int) -> bool:
        """Verifica si un agente ha alcanzado su meta."""
        pass

    def execute_simulation_step(self):
        """
        Ejecuta un paso de la simulación, incluyendo las dinámicas y la actualización de los agentes.
        
        Args:
            step (int): El número del paso actual en la simulación.
        """
        # Ejecutar dinámicas antes del paso
        self._dynamics_manager.run_before_step_dynamics()
        
        # Ejecutar el paso de simulación
        # self.run_simulation(1)
        self.step()
        
        # Ejecutar dinámicas después del paso
        self._dynamics_manager.run_after_step_dynamics()
    
    def run_pipeline(self, steps: int):
        if self._state != SimulationState.SETUP:
            raise RuntimeError("Simulation can only start from the SETUP state.")
        
        # Inicializa la simulación
        self.initialize_simulation()

        # Ejecutar dinámicas iniciales (before + once)
        self._dynamics_manager.run_initial_tasks()

        # Transición al estado RUNNING después de ejecutar las tareas iniciales
        self._state = SimulationState.RUNNING

        for step in range(steps):
            self.current_step = step
            if self._state == SimulationState.STOPPED:
                break
            
            # Ejecutar un paso completo de la simulación
            self.execute_simulation_step()

            # Verificar si el estado ha cambiado a STOPPED durante las dinámicas
            if self._state == SimulationState.STOPPED:
                break

        # Si la simulación no fue detenida, detenerla al final del pipeline
        if self._state != SimulationState.STOPPED:
            self.stop_simulation()

    def get_state(self) -> SimulationState:
        """Devuelve el estado actual de la simulación."""
        return self._state
