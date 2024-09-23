class Registry:
    def __init__(self, categories=None):
        """Inicializa el registro con categorías predefinidas o personalizadas."""
        self._registries = categories or {}

    def add_category(self, category: str):
        """Agrega una nueva categoría al registro."""
        if category in self._registries:
            raise ValueError(f"Categoría {category} ya existe.")
        self._registries[category] = {}

    # En simulator/models/simulation_configuration/registry.py

    def register(cls=None, *, alias=None, category=None, instance=None):
        """Registra una clase o instancia en el registro global bajo una categoría específica."""
        if category not in global_registry._registries:
            raise ValueError(f"Categoría {category} no soportada.")

        def wrapper(cls_or_instance):
            name = alias if alias else cls_or_instance.__class__.__name__
            global_registry._registries[category][name] = cls_or_instance
            return cls_or_instance

        if instance:
            # Si se proporciona una instancia, registrarla directamente
            return wrapper(instance)
        elif cls:
            return wrapper(cls)
        else:
            return wrapper


    def get(self, category: str, name: str):
        """Devuelve la clase registrada bajo el nombre o alias especificado en la categoría dada."""
        if category not in self._registries:
            raise ValueError(f"Categoría {category} no soportada.")
        if name not in self._registries[category]:
            raise ValueError(f"No se encontró el registro en la categoría {category} con el nombre: {name}")
        return self._registries[category][name]

    def instantiate(self, category: str, **kwargs):
        name = kwargs.get('name', None)
        if name is None:
            raise ValueError("El campo 'name' es requerido para instanciar la clase.")
        
        # Obtener la clase del registro usando la categoría y el nombre
        cls = self.get(category, name)
        
        # Instanciar la clase con los kwargs, incluyendo 'name'
        return cls(**kwargs)

# Instancia global del registro con categorías predefinidas
global_registry = Registry(categories={
    'dynamic': {},
    'event': {},
    'shape': {},
    'distribution_pattern': {},
    'behaviour': {}
})

def register(cls=None, *, alias=None, category=None, instance=None):    
    if not category:
        raise ValueError("Category must be specified for registration.")
    
    if instance:
        # Si se proporciona una instancia, registrarla directamente
        if alias is None:
            alias = instance.__class__.__name__
        global_registry._registries[category][alias] = instance
        return instance
    
    if cls:
        # Si se proporciona una clase, registrarla como antes
        if alias is None:
            alias = cls.__name__
        
        def wrapper(cls):
            global_registry._registries[category][alias] = cls
            return cls
        
        return wrapper(cls)
    
    return lambda cls: register(cls=cls, alias=alias, category=category)

