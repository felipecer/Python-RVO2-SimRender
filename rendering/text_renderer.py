from simulator.models.observer import SimulationObserver
from simulator.models.messages import (
    AllGoalsProcessedMessage,
    BaseMessage,
    AgentsStateUpdateMessage,
    ObstaclesProcessedMessage,
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

    # Implementation of SimulationObserver
    def update(self, message: BaseMessage):
        if isinstance(message, SimulationInitializedMessage):
            print(f"Simulation Initialized: {message.message}")
            if message.agent_initialization_data:
                for agent in message.agent_initialization_data:
                    # Get the agent's behavior
                    behaviour = agent.get('behaviour', 'default')
                    print(f"Agent {agent['agent_id']}: Radius={agent['radius']}, MaxSpeed={agent['max_speed']}, "
                          f"NeighborDist={agent['neighbor_dist']}, MaxNeighbors={agent['max_neighbors']}, "
                          f"TimeHorizon={agent['time_horizon']}, TimeHorizonObst={agent['time_horizon_obst']}, "
                          f"Goal={agent['goal']}, Behaviour={behaviour}")  # Print the behavior

        elif isinstance(message, ObstaclesProcessedMessage):
            # Print the number of processed obstacles
            print(f"Obstacles Processed: {len(message.obstacles)} obstacles")
            # Optionally, print the coordinates of the obstacles
            for i, obstacle in enumerate(message.obstacles):
                print(f"Obstacle {i}: {obstacle}")

        elif isinstance(message, AllGoalsProcessedMessage):
            # Print the number of processed goals
            print(f"Goals Processed: {len(message.goals)} goals")
            # Print the positions of the goals if necessary
            for agent_goal in message.goals:
                goal = agent_goal.goal
                agent_id = agent_goal.agent_id
                print(f"Agent {agent_id} goal at position {goal}")

        elif isinstance(message, AgentsStateUpdateMessage):
            # Assuming we are now passing more data, such as velocity and distance to the goal
            for agent_state in message.agent_state_list:
                agent_id = agent_state.agent_id
                (x, y) = agent_state.position
                # If you are sending other parameters like velocity and pref_velocity                
                velocity = agent_state.velocity
                pref_velocity = agent_state.preferred_velocity
                # In case you are also passing the distance
                distance_to_goal = agent_state.distance_to_goal
                print(f"Step {message.step}: Agent {agent_id} at position ({x}, {y}), velocity ({velocity}), preferred velocity ({pref_velocity}), distance to goal: {distance_to_goal}")

        elif isinstance(message, GoalPositionUpdatedMessage):
            # Print the new position of a specific agent's goal
            print(
                f"Goal {message.goal_id} updated to new position {message.new_position}")

        elif isinstance(message, NewObstacleAddedMessage):
            # Print the details of the new obstacle added
            print(f"New Obstacle Added: {message.obstacle}")
