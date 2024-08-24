from simulator.models.observer import SimulationObserver
from simulator.models.messages import (
    BaseMessage,
    AgentPositionsUpdateMessage,
    ObstaclesProcessedMessage,
    GoalsProcessedMessage,
    GoalPositionUpdatedMessage,
    NewObstacleAddedMessage,
    SimulationInitializedMessage
)

class TextRenderer(SimulationObserver):
    def __init__(self):
        self.active = True  # This can be set to False to stop the renderer

    def setup(self):
        """Initial setup for the renderer (optional for text-based rendering)."""
        print("TextRenderer setup complete.")
    
    def is_active(self):
        """Check if the renderer is still active. Always True for simplicity."""
        return self.active

    def stop(self):
        """Method to stop the rendering loop."""
        self.active = False

    # Implementación de SimulationObserver
    def update(self, message: BaseMessage):
        if isinstance(message, SimulationInitializedMessage):
            print(f"Simulation Initialized: {message.message}")
        elif isinstance(message, ObstaclesProcessedMessage):
            print(f"Obstacles Processed: {len(message.obstacles)} obstacles")
        elif isinstance(message, GoalsProcessedMessage):
            print(f"Goals Processed: {len(message.goals)} goals")
        elif isinstance(message, AgentPositionsUpdateMessage):
            agent_descriptions = [f"Agent {agent_id} at position ({x}, {y})" for agent_id, x, y in message.agent_positions]
            print(f"Step {message.step}: {'; '.join(agent_descriptions)}")
        elif isinstance(message, GoalPositionUpdatedMessage):
            print(f"Goal {message.goal_id} updated to new position {message.new_position}")
        elif isinstance(message, NewObstacleAddedMessage):
            print(f"New Obstacle Added: {message.obstacle}")
