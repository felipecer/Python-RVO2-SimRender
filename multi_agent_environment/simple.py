#!/usr/bin/env python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simulator.world_loader import WorldLoader
import rvo2

class RVOSimulationEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self, config_file=None):
			super(RVOSimulationEnv, self).__init__()
			self.loader = WorldLoader(config_file)
			self.world_name, self.sim, self.agent_goals = self.loader.load_simulation()

			self.num_agents = self.sim.getNumAgents()
			self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
			self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
																					shape=(4 * (self.num_agents - 1),), dtype=np.float32)
			# Initialize distances and max time per episode
			self.initial_distance = np.linalg.norm(np.array(self.sim.getAgentPosition(0)) - np.array(self.agent_goals[0][1]))
			self.time_limit = 1000  # Set the time limit per episode
			self.current_step = 0
	
	def _get_obs(self):
		return np.array([self.sim.getAgentPosition(i) for i in range(self.num_agents)])
	
	def _get_info(self):
		return {}

	def step(self, action):
		# Action affects only the first agent
		vx, vy = action
		self.sim.setAgentPrefVelocity(0, (vx, vy))
		self.sim.doStep()
		self.current_step += 1

		observations = []
		for agent_id in range(1, self.num_agents):  # Observe other agents
				pref_vel = self.sim.getAgentPrefVelocity(agent_id)
				actual_vel = self.sim.getAgentVelocity(agent_id)
				observations.extend([pref_vel[0], actual_vel[0], pref_vel[1], actual_vel[1]])

		flattened_observations = np.array(observations)
		reward = self.calculate_reward(0)
		terminated = self.is_done(0)
		truncated = self.current_step >= self.time_limit
		info = self._get_info()
		return flattened_observations, reward, terminated, truncated, info
		
	def reset(self, seed=None, options=None):
		# Cargar nuevamente la configuración de la simulación y asegurarse de asignar correctamente los valores
		self.world_name, self.sim, self.agent_goals = self.loader.load_simulation()
		self.current_step = 0
		self.initial_distance = np.linalg.norm(np.array(self.sim.getAgentPosition(0)) - np.array(self.agent_goals[0][1]))
		info = self._get_info()
		obs = self._get_obs()
		return obs, info

	def render(self, mode='human'):
		# Optional: Implement visualization of the simulation
		pass

	def calculate_reward(self, agent_id):
		reward = 0
		if self.check_collision(agent_id):
				reward -= 10  # Collision penalty
		# Reward for getting closer to the goal
		current_distance = np.linalg.norm(np.array(self.sim.getAgentPosition(agent_id)) - np.array(self.agent_goals[agent_id][1]))
		reward += 100 * (self.initial_distance - current_distance) / self.initial_distance
		return reward

	def is_done(self, agent_id):
		_, goal = self.agent_goals[agent_id]
		return np.linalg.norm(np.array(self.sim.getAgentPosition(agent_id)) - np.array(goal)) < 0.1

	def check_collision(self, agent_id):
		position = self.sim.getAgentPosition(agent_id)
		radius = self.sim.getAgentRadius(agent_id)
		for other_id in range(self.num_agents):
				if other_id != agent_id:
						other_position = self.sim.getAgentPosition(other_id)
						other_radius = self.sim.getAgentRadius(other_id)
						dist = np.sqrt((position[0] - other_position[0]) ** 2 + (position[1] - other_position[1]) ** 2)
						if dist < (radius + other_radius):
								return True
		return False

if __name__ == "__main__":
	env = RVOSimulationEnv('./simulator/worlds/base_scenario.yaml')
	observations = env.reset()
	done = False
	i = 0
	while not done:
		action = env.action_space.sample()  # Take random actions
		observations, reward, done, truncated, info = env.step(action)
		print(f"Step {i} reward: {reward}")
		i += 1			
