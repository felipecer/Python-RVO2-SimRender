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
            if message.agent_initialization_data:
                for agent in message.agent_initialization_data:
                    # Obtener el comportamiento del agente
                    behaviour = agent.get('behaviour', 'default')
                    print(f"Agent {agent['agent_id']}: Radius={agent['radius']}, MaxSpeed={agent['max_speed']}, "
                          f"NeighborDist={agent['neighbor_dist']}, MaxNeighbors={agent['max_neighbors']}, "
                          f"TimeHorizon={agent['time_horizon']}, TimeHorizonObst={agent['time_horizon_obst']}, "
                          f"Goal={agent['goal']}, Behaviour={behaviour}")  # Imprimir el comportamiento

        elif isinstance(message, ObstaclesProcessedMessage):
            # Imprime el número de obstáculos procesados
            print(f"Obstacles Processed: {len(message.obstacles)} obstacles")
            # Opcionalmente, imprime las coordenadas de los obstáculos
            for i, obstacle in enumerate(message.obstacles):
                print(f"Obstacle {i}: {obstacle}")

        elif isinstance(message, GoalsProcessedMessage):
            # Imprime el número de metas procesadas
            print(f"Goals Processed: {len(message.goals)} goals")
            # Imprime las posiciones de las metas si es necesario
            for agent_id, goal in message.goals.items():
                print(f"Agent {agent_id} goal at position {goal}")

        elif isinstance(message, AgentPositionsUpdateMessage):
            # Asumiendo que ahora estamos pasando más datos, como velocidad y distancia a la meta
            for agent_data in message.agent_positions:
                agent_id = agent_data[0]
                x, y = agent_data[1], agent_data[2]
                # Si estás enviando otros parámetros como velocidad y pref_velocity
                if len(agent_data) > 3:
                    velocity = agent_data[3]
                    pref_velocity = agent_data[4]
                    # En caso de que también pases la distancia
                    distance_to_goal = agent_data[5] if len(
                        agent_data) > 5 else None
                    print(f"Step {message.step}: Agent {agent_id} at position ({x}, {y}), velocity ({velocity}), preferred velocity ({pref_velocity}), distance to goal: {distance_to_goal}")
                else:
                    print(
                        f"Step {message.step}: Agent {agent_id} at position ({x}, {y})")

        elif isinstance(message, GoalPositionUpdatedMessage):
            # Imprime la nueva posición de la meta de un agente específico
            print(
                f"Goal {message.goal_id} updated to new position {message.new_position}")

        elif isinstance(message, NewObstacleAddedMessage):
            # Imprime los detalles del nuevo obstáculo agregado
            print(f"New Obstacle Added: {message.obstacle}")
