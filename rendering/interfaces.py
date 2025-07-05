import abc

class RendererInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'setup') and 
                callable(subclass.setup) and 
                hasattr(subclass, 'is_active') and 
                callable(subclass.is_active) and
                hasattr(subclass, 'render_step_with_agents') and 
                callable(subclass.render_step_with_agents) or 
                NotImplemented)
    
    @abc.abstractmethod
    def setup(self):
        """Initializes and setups internal components of renderer"""
        pass

    @abc.abstractmethod
    def is_active(self):
        """returns True when renderer is working"""
        pass
    
    @abc.abstractmethod
    def render_step_with_agents(self, agents, step):
        """Renders the agents at their specified location at a particular timestep"""
        pass
