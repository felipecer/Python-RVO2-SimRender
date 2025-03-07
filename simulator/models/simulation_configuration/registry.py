class Registry:
    def __init__(self, categories=None):
        """Initialize the registry with predefined or custom categories."""
        self._registries = categories or {}

    def add_category(self, category: str):
        """Add a new category to the registry."""
        if category in self._registries:
            raise ValueError(f"Category {category} already exists.")
        self._registries[category] = {}

    def register(cls=None, *, alias=None, category=None, instance=None):
        """Register a class or instance in the global registry under a specific category."""
        if category not in global_registry._registries:
            raise ValueError(f"Category {category} not supported.")

        def wrapper(cls_or_instance):
            name = alias if alias else cls_or_instance.__class__.__name__
            global_registry._registries[category][name] = cls_or_instance
            return cls_or_instance

        if instance:
            # If an instance is provided, register it directly
            return wrapper(instance)
        elif cls:
            return wrapper(cls)
        else:
            return wrapper

    def get(self, category: str, name: str):
        """Return the class registered under the specified name or alias in the given category."""
        if category not in self._registries:
            raise ValueError(f"Category {category} not supported.")
        if name not in self._registries[category]:
            raise ValueError(f"Registry not found in category {category} with name: {name}")
        return self._registries[category][name]

    def instantiate(self, category: str, **kwargs):
        name = kwargs.get('name', None)
        if name is None:
            raise ValueError("The 'name' field is required to instantiate the class.")
        
        # Get the class from the registry using the category and name
        cls = self.get(category, name)
        
        # Instantiate the class with the kwargs, including 'name'
        return cls(**kwargs)

# Global instance of the registry with predefined categories
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
        # If an instance is provided, register it directly
        if alias is None:
            alias = instance.__class__.__name__
        global_registry._registries[category][alias] = instance
        return instance
    
    if cls:
        # If a class is provided, register it as before
        if alias is None:
            alias = cls.__name__
        
        def wrapper(cls):
            global_registry._registries[category][alias] = cls
            return cls
        
        return wrapper(cls)
    
    return lambda cls: register(cls=cls, alias=alias, category=category)
