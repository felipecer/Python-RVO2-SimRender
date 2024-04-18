import gym
from gym import spaces
import numpy as np
from world_loader import WorldLoader
import rvo2

class RVOSimulationEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self, config_file):
		super(RVOSimulationEnv, self).__init__()
		self.loader = WorldLoader(config_file)
		self.world_name, self.sim, self.agent_goals = self.loader.load_simulation()

		# Define action and observation space
		self.num_agents = self.sim.getNumAgents()
		self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
		self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
											shape=(4 * (self.num_agents - 1),), dtype=np.float32)

	def step(self, action):
		# Set the preferred velocity for the controlled agent (agent_id = 0)
		vx, vy = action
		self.sim.setAgentPrefVelocity(0, (vx, vy))

		# Update the simulation
		self.sim.doStep()

		# Gather observations for all other agents
		observations = []
		for agent_id in range(1, self.num_agents):
			pref_vel = self.sim.getAgentPrefVelocity(agent_id)
			actual_vel = self.sim.getAgentVelocity(agent_id)
			observations.extend([pref_vel[0], pref_vel[1], actual_vel[0], actual_vel[1]])

		# Calculate reward and done status
		reward = self.calculate_reward()
		done = self.is_done()

		return np.array(observations), reward, done, {}

	def reset(self):
		# Reset the simulation to initial state
		self.sim = rvo2.PyRVOSimulator(timeStep=0.25, neighborDist=15.0, maxNeighbors=10,
										timeHorizon=5.0, timeHorizonObst=2.0, radius=1.5, maxSpeed=2.0)
		_, self.agent_goals = self.loader.load_simulation(self.sim)
		return self.step(np.array([0.0, 0.0]))[0]  # return initial observation

	def render(self, mode='human'):
		# Optional: Implement visualization of the simulation
		pass

	def calculate_reward(self):
		# Define how rewards are calculated. Example: negative reward for collision
		return -1 if self.check_collision() else 0

	def is_done(self):
		# Define conditions to end the episode
		return False

	def check_collision(self):
		# Check for collision between agents
		for agent_id in range(self.num_agents):
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
	env = RVOSimulationEnv()
	observations = env.reset()
	done = False
	while not done:
		action = env.action_space.sample()  # Take random actions
		observations, reward, done, info = env.step(action)
		print(f"Step reward: {reward}")
