from abc import abstractmethod, ABC
from typing import Callable, List, Dict, Tuple
from enum import Enum
from simulator.models.simulation_configuration.simulation_dynamics import ExecutionTiming, OnceDynamic

class DynamicsQueueManager:
    def __init__(self):
        self.initial_tasks: List[Callable] = []     # Lista de tareas que se ejecutan una vez al inicio (before + once)
        self.before_step_list: List[Callable] = []  # Lista de tareas que se ejecutan antes de cada step
        self.after_step_list: List[Callable] = []   # Lista de tareas que se ejecutan después de cada step
        self.final_tasks: List[Callable] = []       # Lista de tareas que se ejecutan una vez al final (after + once)

    def add_dynamic(self, dynamic: Callable, timing: ExecutionTiming, on_demand: bool = False):
        """Agrega una dinámica a la lista correspondiente."""
        if timing == ExecutionTiming.BEFORE:
            if on_demand:
                self.before_step_list.insert(0, dynamic)  # On demand se agrega al principio de la lista before
            else:
                self.before_step_list.append(dynamic)
        elif timing == ExecutionTiming.AFTER:
            if on_demand:
                self.after_step_list.insert(0, dynamic)  # On demand se agrega al principio de la lista after
            else:
                self.after_step_list.append(dynamic)

    def add_once_dynamic(self, dynamic: Callable, timing: ExecutionTiming):
        if not isinstance(dynamic, OnceDynamic):
            raise TypeError(f"Expected OnceDynamic, got {type(dynamic).__name__}")

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
        """Ejecuta todas las dinámicas en la lista before_step en el orden adecuado."""
        while self.before_step_list:
            dynamic = self.before_step_list.pop(0)
            dynamic()

    def run_after_step_dynamics(self):
        """Ejecuta todas las dinámicas en la lista after_step en el orden adecuado."""
        while self.after_step_list:
            dynamic = self.after_step_list.pop(0)
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
    def __init__(self):
        self._dynamics_manager = DynamicsQueueManager()
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._state = SimulationState.SETUP
        self.current_step: int = 0

    def register_dynamic(self, dynamic):
        """Registra una dinámica para que se ejecute durante la simulación."""
        if self._state != SimulationState.SETUP:
            raise RuntimeError("Cannot register dynamics after setup is complete.")
        
        dynamic.register_simulator(self)
        if isinstance(dynamic, OnceDynamic):
            self._dynamics_manager.add_once_dynamic(dynamic.apply, dynamic.when)
        else:
            self._dynamics_manager.add_dynamic(dynamic.apply, dynamic.when)

    def register_event_handler(self, event_type: str, handler: Callable):
        """Permite que una dinámica se suscriba a un evento específico."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def handle_event(self, event_type: str, *args, **kwargs):
        """Maneja un evento recibido y encola las acciones correspondientes."""
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            action = lambda: handler(*args, **kwargs)
            # Aquí decidimos en qué momento se debe ejecutar la acción
            # Por defecto, encolamos la acción en la lista "before_step"
            self._dynamics_manager.add_dynamic(action, ExecutionTiming.BEFORE)

    @abstractmethod
    def initialize_simulation(self):
        """Inicializa la simulación, creando agentes, metas, y obstáculos."""
        pass

    @abstractmethod
    def run_simulation(self, step: int):
        """Ejecuta el ciclo de la simulación por un número de pasos especificado."""
        pass

    def stop_simulation(self):
        """Detiene la simulación."""
        self._state = SimulationState.STOPPED
        print("Simulation stopped.")
        self._dynamics_manager.run_final_tasks()  # Ejecutar las tareas finales al detener la simulación

    @abstractmethod
    def get_agent_positions(self) -> Dict[int, Tuple[float, float]]:
        """Devuelve las posiciones actuales de los agentes."""
        pass

    @abstractmethod
    def get_agent_goal(self, agent_id: int) -> Tuple[float, float]:
        """Devuelve la meta actual de un agente dado su ID."""
        pass

    @abstractmethod
    def is_goal_reached(self, agent_id: int) -> bool:
        """Verifica si un agente ha alcanzado su meta."""
        pass
    
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
            
            # Ejecutar dinámicas antes del paso
            self._dynamics_manager.run_before_step_dynamics()
            
            # Ejecutar el paso de simulación
            self.run_simulation(step)
            
            # Ejecutar dinámicas después del paso
            self._dynamics_manager.run_after_step_dynamics()

            # Verificar si el estado ha cambiado a STOPPED durante las dinámicas
            if self._state == SimulationState.STOPPED:
                break

        # Si la simulación no fue detenida, detenerla al final del pipeline
        if self._state != SimulationState.STOPPED:
            self.stop_simulation()

    def get_state(self) -> SimulationState:
        """Devuelve el estado actual de la simulación."""
        return self._state
