from rendering.interfaces import RendererInterface

class TextRenderer(RendererInterface):
    def __init__(self):
        self.active = True  # This can be set to False to stop the renderer

    def setup(self):
        """Initial setup for the renderer (optional for text-based rendering)."""
        print("TextRenderer setup complete.")
    
    def is_active(self):
        """Check if the renderer is still active. Always True for simplicity."""
        return self.active
    
    def render_step_with_agents(self, agents, step):
        """Render a simulation step by printing agent states to the console, all on one line."""
        agent_descriptions = [f"Agent {agent_id} at position ({x}, {y})" for agent_id, x, y in agents]
        print(f"Step {step}: {'; '.join(agent_descriptions)}")
    
    def stop(self):
        """Method to stop the rendering loop."""
        self.active = False
