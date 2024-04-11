#!/usr/bin/env python3
import datetime
from world_loader import WorldLoader

class SimulationCore:
    def __init__(self, world_loader: WorldLoader, simulation_id: str):
        self.world_loader = world_loader
        self.simulation_id = simulation_id.replace(" ", "_")
        self.steps_buffer = []        
        _, self.sim, _ = self.world_loader.load_simulation()

    def run_simulation(self, steps):        
        for step in range(steps):
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
    sim_core.run_simulation(500) 
    sim_core.save_simulation_runs()