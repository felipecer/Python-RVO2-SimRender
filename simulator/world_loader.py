#!/usr/bin/env python3
import yaml
import rvo2


class WorldLoader:
    def __init__(self, yaml_file):
        self.yaml_file = yaml_file
        self.world_name = ""
        self.config = {}
        self.obstacles = []
        self.sim = None
        self.agent_goals = {}

    def get_simulation(self):
        return self.world_name, self.sim, self.agent_goals

    def load_simulation(self):
        with open(self.yaml_file, 'r') as file:
            config = yaml.safe_load(file)

        self.config = config

        # Cargar el nombre del mundo
        self.world_name = self.config.get('worldName', 'Mundo sin nombre')

        # Crear la instancia de la simulación.
        sim = rvo2.PyRVOSimulator(
            config['simulation']['timeStep'],
            config['simulation']['agentDefaults']['neighborDist'],
            config['simulation']['agentDefaults']['maxNeighbors'],
            config['simulation']['agentDefaults']['timeHorizon'],
            config['simulation']['agentDefaults']['timeHorizonObst'],
            config['simulation']['agentDefaults']['radius'],
            config['simulation']['agentDefaults']['maxSpeed']
        )

        # Añadir agentes y sus metas.
        agent_goals = {}
        for agent in config['simulation']['agents']:
            agent_id = sim.addAgent(
                tuple(agent['position']),
                config['simulation']['agentDefaults']['neighborDist'],
                config['simulation']['agentDefaults']['maxNeighbors'],
                config['simulation']['agentDefaults']['timeHorizon'],
                config['simulation']['agentDefaults']['timeHorizonObst'],
                config['simulation']['agentDefaults']['radius'],
                agent['maxSpeed'],
                tuple(agent['velocity'])
            )
            sim.setAgentPrefVelocity(
                agent_id, tuple(agent['preferredVelocity']))
            agent_goals[agent_id] = tuple(agent['goal'])

        # Añadir obstáculos.
        for obstacle in config['simulation']['obstacles']:
            sim.addObstacle([tuple(vertex) for vertex in obstacle])
            vertices = [tuple(vertex) for vertex in obstacle]
            self.obstacles.append(vertices)
        sim.processObstacles()
        self.sim = sim
        self.agent_goals = agent_goals
        # Devolver la simulación configurada y los IDs de los agentes con sus metas.
        return self.world_name, sim, agent_goals

    def get_world_name(self):
        return self.world_name

    def get_obstacles(self):
        """Devuelve la lista de obstáculos almacenados."""
        return self.obstacles

    def get_goals(self):
        return self.agent_goals


def load_world(yaml_file):
    loader = WorldLoader(yaml_file)
    name, sim, agent_goals = loader.load_simulation()
    return name, sim, agent_goals


if __name__ == "__main__":
    name, sim, agent_goals = load_world('./worlds/base_scenario.yaml')
