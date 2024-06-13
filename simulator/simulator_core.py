#!/usr/bin/env python3
import datetime
import math
from world_loader import WorldLoader
from rendering.pygame_renderer import PyGameRenderer, Grid
from rendering.text_renderer import TextRenderer
from rendering.interfaces import RendererInterface
import sys


class SimulationCore:
    def __init__(self, world_loader: WorldLoader, simulation_id: str, renderer: RendererInterface = None):
        self.world_loader = world_loader
        self.simulation_id = simulation_id.replace(" ", "_")
        self.steps_buffer = []
        self.renderer = renderer
        self.sim_name, self.sim, self.agent_goals = self.world_loader.get_simulation()

    def _init_renderer(self):
        if not self.renderer:
            return
        self.renderer.init_window()

    def calculate_preferred_velocity(self, agent_position, goal_position, max_speed):
        vector_to_goal = (
            goal_position[0] - agent_position[0], goal_position[1] - agent_position[1])
        distance = math.sqrt(vector_to_goal[0] ** 2 + vector_to_goal[1] ** 2)

        if distance > 0:
            return (vector_to_goal[0] / distance * max_speed, vector_to_goal[1] / distance * max_speed)
        else:
            return (0, 0)

    def update_agent_velocities(self):
        for agent_id in range(self.sim.getNumAgents()):
            agent_position = self.sim.getAgentPosition(agent_id)
            goal_position = self.agent_goals[agent_id]
            max_speed = self.sim.getAgentMaxSpeed(agent_id)
            preferred_velocity = self.calculate_preferred_velocity(
                agent_position, goal_position, max_speed)
            self.sim.setAgentPrefVelocity(agent_id, preferred_velocity)

    def run_simulation(self, steps):
        for step in range(steps):
            self.update_agent_velocities()
            self.sim.doStep()

            for agent_id in range(self.sim.getNumAgents()):
                agent_velocity = self.sim.getAgentVelocity(agent_id)
                agent_pref_velocity = self.sim.getAgentPrefVelocity(agent_id)
                print(
                    f"Step: {step} | Agent {agent_id}: Actual Velocity = {agent_velocity}, Preferred Velocity = {agent_pref_velocity}")

            if self.renderer:
                agent_positions = [(agent_id, *self.sim.getAgentPosition(agent_id))
                                   for agent_id in range(self.sim.getNumAgents())]
                if self.renderer.is_active():
                    self.renderer.render_step_with_agents(
                        agent_positions, step)
            self.store_step(step)

    def store_step(self, step):
        step_data = {'step': step, 'agents': []}
        for agent_id in range(self.sim.getNumAgents()):
            position = self.sim.getAgentPosition(agent_id)
            step_data['agents'].append({
                'id': agent_id,
                'position': position
            })
        self.steps_buffer.append(step_data)

    def save_simulation_runs(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.world_loader.get_world_name()}_{self.simulation_id}_{timestamp}.txt".replace(
            " ", "_")

        with open(filename, 'w') as file:
            for step_data in self.steps_buffer:
                step = step_data['step']
                for agent_data in step_data['agents']:
                    file.write(
                        f"{step},{agent_data['id']},{agent_data['position'][0]:.2f},{agent_data['position'][1]:.2f}\n")

        print(f"Archivo de simulación guardado como: {filename}")

    def clear_buffer(self):
        self.steps_buffer = []


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simulator_core.py <world_file.yaml>")
        sys.exit(1)
    world_file = sys.argv[1]
    loader = WorldLoader(world_file)
    _ = loader.load_simulation()
    obstacles = loader.get_obstacles()
    goals = loader.get_goals()
    grid = Grid(1000, 1000, 100)
    renderer = PyGameRenderer(
        1000, 1000, obstacles=obstacles, goals=goals, grid=grid, cell_size=grid.spacing)
    # renderer = TextRenderer()
    renderer.setup()
    sim_core = SimulationCore(loader, "test", renderer=renderer)
    sim_core.run_simulation(5000)
    sim_core.save_simulation_runs()
