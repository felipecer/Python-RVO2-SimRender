class Registry:
    def __init__(self, categories=None):
        """Inicializa el registro con categorías predefinidas o personalizadas."""
        self._registries = categories or {}

    def add_category(self, category: str):
        """Agrega una nueva categoría al registro."""
        if category in self._registries:
            raise ValueError(f"Categoría {category} ya existe.")
        self._registries[category] = {}

    def register(self, cls=None, *, alias=None, category=None):
        """Registra una clase en el registro global bajo una categoría específica."""
        if category not in self._registries:
            raise ValueError(f"Categoría {category} no soportada.")

        def wrapper(cls):
            name = alias if alias else cls.__name__
            self._registries[category][name] = cls
            return cls

        if cls is None:
            return wrapper
        else:
            return wrapper(cls)

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

# Función decoradora que especifica alias y categoría
def register(cls=None, *, alias=None, category=None):
    return global_registry.register(cls=cls, alias=alias, category=category)
