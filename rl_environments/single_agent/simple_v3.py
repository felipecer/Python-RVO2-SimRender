#!/usr/bin/env python
import gymnasium as gym
from gymnasium import spaces, logger
import numpy as np
from simulator.world_loader import WorldLoader
import rvo2
from rendering.pygame_renderer import Grid, PyGameRenderer
from rendering.text_renderer import TextRenderer
import math


class RVOSimulationEnv(gym.Env):
    metadata = {'render.modes': ['ansi', 'rbg']}

    def __init__(self, config_file=None, render_mode="rgb"):
        super(RVOSimulationEnv, self).__init__()
        self.loader = WorldLoader(config_file)
        self.world_name, self.sim, self.agent_goals = self.loader.load_simulation()

        self.num_agents = self.sim.getNumAgents()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4 * (self.num_agents - 1),), dtype=np.float32)
        # Initialize distances and max time per episode
        self.initial_distance = np.linalg.norm(
            np.array(self.sim.getAgentPosition(0)) - np.array(self.agent_goals[0][1]))
        self.time_limit = 750  # Set the time limit per episode
        self.current_step = 0
        self.render_mode = render_mode
        self._render_buffer = []
        if render_mode is not None and render_mode == "rgb":
            obstacles = self.loader.get_obstacles()
            goals = self.loader.get_goals()
            grid = Grid(1000, 1000, 100)
            self._gui_renderer = PyGameRenderer(
                1000, 1000, obstacles=obstacles, goals=goals, grid=grid, cell_size=grid.spacing)
            self._gui_renderer.setup()

        self._text_renderer = TextRenderer()
        self.sim.setAgentPrefVelocity(0, (1, 1))

    def _get_obs(self):
        observations = []
        for i in range(1, self.num_agents):
            pos = self.sim.getAgentPosition(i)
            vel = self.sim.getAgentVelocity(i)
            observations.extend([pos[0], pos[1], vel[0], vel[1]])
        return np.array(observations, dtype=np.float32)

    def _get_info(self):
        return {}

    def calculate_preferred_velocity(self, agent_position, goal_position, max_speed):
        vector_to_goal = (
            goal_position[0] - agent_position[0], goal_position[1] - agent_position[1])
        distance = math.sqrt(vector_to_goal[0] ** 2 + vector_to_goal[1] ** 2)

        if distance > 0:
            return (vector_to_goal[0] / distance * max_speed, vector_to_goal[1] / distance * max_speed)
        else:
            return (0, 0)

    def update_agent_velocities(self, agent_id, action=None):
        if action is not None:
            vx, vy = action
            self.sim.setAgentPrefVelocity(agent_id, (vx, vy))
            return
        agent_position = self.sim.getAgentPosition(agent_id)
        goal_position = self.agent_goals[agent_id]
        max_speed = self.sim.getAgentMaxSpeed(agent_id)
        preferred_velocity = self.calculate_preferred_velocity(
            agent_position, goal_position, max_speed)
        self.sim.setAgentPrefVelocity(agent_id, preferred_velocity)

    def step(self, action):
        # Action affects only the first agent
        self.sim.doStep()
        self.update_agent_velocities(agent_id=0, action=action)
        self.update_agent_velocities(agent_id=1)

        self.current_step += 1
        agent_positions = [(agent_id, *self.sim.getAgentPosition(agent_id))
                           for agent_id in range(self.sim.getNumAgents())]
        self._render_buffer.append((self.current_step, agent_positions))
        if self.render_mode is not None:
            self.render()

        flattened_observations = self._get_obs()
        reward = self.calculate_reward(0)
        terminated = self.is_done(0)
        truncated = self.current_step >= self.time_limit
        info = self._get_info()
        return flattened_observations, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Cargar nuevamente la configuración de la simulación y asegurarse de asignar correctamente los valores
        self.world_name, self.sim, self.agent_goals = self.loader.load_simulation()
        self.current_step = 0
        self.initial_distance = np.linalg.norm(
            np.array(self.sim.getAgentPosition(0)) - np.array(self.agent_goals[0][1]))
        info = self._get_info()
        obs = self._get_obs()
        if self.render_mode is not None:
            self.render()
        return obs, info

    def render(self):
        if self.render_mode is None:
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
        elif self.render_mode == "ansi":
            return self._render_text()
        else:
            return self._render_gui()

    def _render_text(self):
        while self._render_buffer:
            step, agents = self._render_buffer.pop(0)
            self._text_renderer.render_step_with_agents(
                agents=agents, step=step)

    def _render_gui(self):
        while self._render_buffer:
            step, agents = self._render_buffer.pop(0)
            self._gui_renderer.render_step_with_agents(
                agents=agents, step=step)

    # def calculate_reward(self, agent_id):
    #     reward = 0
    #     agent_position = np.array(self.sim.getAgentPosition(agent_id))
    #     goal_position = np.array(self.agent_goals[agent_id][1])

    #     current_distance = np.sum(np.abs(agent_position - goal_position))

    #     # Adjusting the penalty and reward based on the distance
    #     distance_penalty = 20 * (current_distance / self.initial_distance)

    #     # Collision penalty
    #     if self.check_collision(agent_id):
    #         reward -= 10

    #     # Proportional reward for moving closer to the goal
    #     # reward += 2 * (self.initial_distance - current_distance) / self.initial_distance

    #     # Major reward for reaching the goal and scaled penalty based on distance
    #     if self.is_done(agent_id):
    #         reward += 10000
    #     else:
    #         reward -= distance_penalty

    #     return reward
    def calculate_reward(self, agent_id):
        reward = 0
        agent_position = np.array(
            self.sim.getAgentPosition(agent_id))
        goal_position = np.array(self.agent_goals[agent_id][1])
        current_distance = np.sum(
            np.abs(agent_position - goal_position))

        # Adjusting the penalty and reward based on the distance
        distance_penalty = 20 * \
            (current_distance / self.initial_distance)
        # distance_penalty = 0
        reward -= -10

        # Collision penalty
        if self.check_collision(agent_id):
            reward -= 10

            # Proportional reward for moving closer to the goal
            # reward += 2 * (self.initial_distance - current_distance) / self.initial_distance

            # Major reward for reaching the goal and scaled penalty based on distance
        if self.is_done(agent_id):
            reward += 50000
        else:
            reward -= distance_penalty
        return reward

    def is_done(self, agent_id):
        _, goal = self.agent_goals[agent_id]
        distance = np.linalg.norm(
            np.array(self.sim.getAgentPosition(agent_id)) - np.array(goal))
        return bool(distance < 0.01)

    def check_collision(self, agent_id):
        position = self.sim.getAgentPosition(agent_id)
        radius = self.sim.getAgentRadius(agent_id)
        for other_id in range(self.num_agents):
            if other_id != agent_id:
                other_position = self.sim.getAgentPosition(other_id)
                other_radius = self.sim.getAgentRadius(other_id)
                dist = np.sqrt(
                    (position[0] - other_position[0]) ** 2 + (position[1] - other_position[1]) ** 2)
                if dist < (radius + other_radius):
                    return True
        return False


if __name__ == "__main__":
    env = RVOSimulationEnv(
        './simulator/worlds/base_scenario.yaml', render_mode='rgb')
    observations = env.reset()
    done = False
    i = 0
    while not done:
        action = env.action_space.sample()  # Take random actions
        observations, reward, done, truncated, info = env.step(action)
        print(f"Step {i} reward: {reward}")
        i += 1
