# simulator/models/simulation_configuration/__init__.py
from . import shapes, simulation_dynamics, simulation_events, distribution_patterns
from .registry import global_registry
import inspect

def auto_register(module, category: str):
    """Registra dinámicamente todas las clases decoradas en un módulo dado."""
    # Inspeccionamos las clases dentro del módulo directamente
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if hasattr(obj, '__decorated__'):  # Verificamos si la clase está decorada
            print(f"Registrando automáticamente {name} en la categoría {category}")
            global_registry.register(cls=obj, category=category)(obj)

# Ejecutar el auto-registro para formas, dinámicas, eventos y patrones
auto_register(shapes, 'shapes')
auto_register(simulation_dynamics, 'dynamics')
auto_register(simulation_events, 'events')
auto_register(distribution_patterns, 'distribution_patterns')
