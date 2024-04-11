#!/usr/bin/env python3
import datetime
import math
from world_loader import WorldLoader

class SimulationCore:
    def __init__(self, world_loader: WorldLoader, simulation_id: str):
        self.world_loader = world_loader
        self.simulation_id = simulation_id.replace(" ", "_")
        self.steps_buffer = []        
        self.sim_name, self.sim, self.agent_goals = self.world_loader.load_simulation()

    def calculate_preferred_velocity(self, agent_position, goal_position, max_speed):
        print(agent_position)
        print(goal_position)
        vector_to_goal = (goal_position[1][0] - agent_position[0], goal_position[1][1] - agent_position[1])
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
            preferred_velocity = self.calculate_preferred_velocity(agent_position, goal_position, max_speed)
            self.sim.setAgentPrefVelocity(agent_id, preferred_velocity)

    def run_simulation(self, steps):        
        for step in range(steps):
            self.update_agent_velocities()
            self.sim.doStep()
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
        filename = f"{self.world_loader.get_world_name()}_{self.simulation_id}_{timestamp}.txt".replace(" ", "_")
        
        with open(filename, 'w') as file:
            for step_data in self.steps_buffer:
                step = step_data['step']
                for agent_data in step_data['agents']:
                    file.write(f"{step},{agent_data['id']},{agent_data['position'][0]:.2f},{agent_data['position'][1]:.2f}\n")
        
        print(f"Archivo de simulación guardado como: {filename}")
    
    def clear_buffer(self):
        self.steps_buffer = []

if __name__ == "__main__":
    loader = WorldLoader("./worlds/base_scenario.yaml")    
    sim_core = SimulationCore(loader, "test")   
    sim_core.run_simulation(5000) 
    sim_core.save_simulation_runs()