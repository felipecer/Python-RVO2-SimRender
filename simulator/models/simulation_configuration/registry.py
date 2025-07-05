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
    
    def __str__(self):
        """Return a human-readable string representation of the registry contents."""
        lines = ["Registry Contents:"]
        
        for category, items in self._registries.items():
            lines.append(f"\n• Category: {category} ({len(items)} items)")
            
            if not items:
                lines.append("  [Empty]")
                continue
            
            # Sort items alphabetically
            for name in sorted(items.keys()):
                item = items[name]
                
                # Get class name
                if hasattr(item, '__class__'):
                    class_name = item.__class__.__name__
                else:
                    class_name = "Unknown"
                
                # Get additional details based on item type
                details = self._get_item_details(category, name, item)
                lines.append(f"  - {name} ({class_name}{details})")
        
        return "\n".join(lines)
    
    def _get_item_details(self, category, name, item):
        """Extract readable details from registry items based on their type."""
        details = ""
        
        # For behavior objects, show additional info
        if category == 'behaviour':
            if hasattr(item, 'behaviors'):  # EitherBehaviour
                details += f", behaviors: {item.behaviors}"
                if hasattr(item, 'weights') and item.weights:
                    details += f", weights: {item.weights}"
            elif hasattr(item, 'agent_defaults'):  # OrcaBehaviour
                if hasattr(item.agent_defaults, 'max_speed'):
                    details += f", max_speed: {item.agent_defaults.max_speed}"
                if hasattr(item.agent_defaults, 'radius'):
                    details += f", radius: {item.agent_defaults.radius}"
        
        # For distribution patterns
        elif category == 'distribution_pattern':
            if hasattr(item, 'count'):
                details += f", count: {item.count}"
        
        return details
    
    def __repr__(self):
        """Return the string representation."""
        return self.__str__()
    

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
