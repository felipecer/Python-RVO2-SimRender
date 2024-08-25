from abc import abstractmethod, ABC
from typing import List, Dict, Tuple

# Importación diferida
# from simulator.models.simulation_configuration.simulation_dynamics import SimulationDynamic

class SimulationEngine(ABC):
    def __init__(self):
        self._dynamics: List['SimulationDynamic'] = []

    def register_dynamic(self, dynamic):
        """Registra una dinámica para que se ejecute durante la simulación."""
        # Importación diferida para evitar el ciclo
        from simulator.models.simulation_configuration.simulation_dynamics import SimulationDynamic
        if not isinstance(dynamic, SimulationDynamic):
            raise ValueError("Dynamic must be an instance of SimulationDynamic")
        self._dynamics.append(dynamic)
        dynamic.register_simulator(self)

    def apply_dynamics(self):
        """Aplica todas las dinámicas registradas en el simulador."""
        for dynamic in self._dynamics:
            dynamic.apply()

    @abstractmethod
    def initialize_simulation(self):
        """Inicializa la simulación, creando agentes, metas, y obstáculos."""
        pass

    @abstractmethod
    def run_simulation(self, step: int):
        """Ejecuta el ciclo de la simulación por un número de pasos especificado."""
        pass

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
        """Ejecuta la simulación en un pipeline definido."""
        self.initialize_simulation()
        for step in range(steps):
            self.apply_dynamics()  # Aplicar dinámicas antes del paso
            self.run_simulation(step)  # Ejecutar el paso de simulación
            self.apply_dynamics()  # Aplicar dinámicas después del paso, si es necesario
