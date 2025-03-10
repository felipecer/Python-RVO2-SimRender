#!/usr/bin/env python
# import pygame_gui
# from pygame_gui.elements import UIHorizontalSlider
import sys

import pygame

# Import the parser to load color schemes
from rendering.color_scheme_parser import load_color_schemes
from rendering.drawing_utils import draw_text, draw_arrow, draw_detection_radius, draw_distance_to_goal, Grid
from rendering.interfaces import RendererInterface
from simulator.models.messages import (
    AgentPositionsUpdateMessage,
    RayCastingUpdateMessage,
    SimulationInitializedMessage,
    ObstaclesProcessedMessage,
    GoalsProcessedMessage,
    GoalPositionUpdatedMessage,
    NewObstacleAddedMessage
)
from simulator.models.observer import SimulationObserver


class PyGameRenderer(RendererInterface, SimulationObserver):
    def __init__(self, width, height, color_scheme_file='./rendering/color_schemes.yaml', color_scheme_name='orca-behaviors',
                 map=None, simulation_steps={}, obstacles=[], goals={}, agents=[], display_caption='Simulador de Navegación de Agentes',
                 font_size=8, font_name='arial', cell_size=50):
        # Load the color scheme
        self.color_scheme_config = load_color_schemes(color_scheme_file)
        self.color_scheme = self.color_scheme_config.schemes[color_scheme_name]

        self.font_name = font_name
        self.font_size = font_size
        self.map = map
        self.obstacles = obstacles
        self.goals = goals
        self.clock = pygame.time.Clock()
        self.agents = agents
        self.simulation_steps = simulation_steps
        self.cell_size = cell_size
        self.grid = Grid(width, height, cell_size)
        self.raycasting_intersections = None

        # Window settings
        self.window = None
        self.window_width, self.window_height = width, height
        self.display_caption = display_caption
        self._rendering_is_active = False

        self.delay_slider = None
        self.delay = 10

    def _pygame_event_manager(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._rendering_is_active = False
                break
            # self.ui_manager.process_events(event)

    def is_active(self):
        return self._rendering_is_active

    def setup(self):
        # Pygame Initialization
        pygame.init()
        self.window = pygame.display.set_mode(
            (self.window_width, self.window_height))
        pygame.display.set_caption(self.display_caption)
        self._rendering_is_active = True

    def transform_coordinates(self, x, y):
        scale = self.cell_size
        x_new = self.window_width / 2 + x * scale
        y_new = self.window_height / 2 - y * scale
        return int(x_new), int(y_new)

    def draw_grid(self):
        self.grid.draw(self.window)

    def draw_terrain(self):
        pass

    def draw_obstacles(self):
        for obstacle in self.obstacles:
            vertices_transformed = [
                self.transform_coordinates(x, y) for x, y in obstacle]
            pygame.draw.polygon(
                self.window, self.color_scheme.obstacle_color, vertices_transformed, 3)

    def draw_agents(self, step):
        if step in self.simulation_steps:
            for agent_id, x, y in self.simulation_steps[step]:
                x, y = self.transform_coordinates(x, y)
                pygame.draw.circle(
                    self.window, self.color_scheme.obstacle_color, (x, y), 10)

    def draw_goals(self):
        for agent_id, goal in self.goals.items():
            x, y = self.transform_coordinates(*goal)
            # Use the same radius as the agent
            radius = self.agent_radii.get(agent_id, 10)
            # Draw the goal circle with the same radius as the agent
            pygame.draw.circle(self.window, self.color_scheme.goal_color, (x, y),
                               int(radius * self.cell_size))

            # Add text inside the goal circle
            draw_text(self.window, f"G_{agent_id}", x, y)

    def draw_intersections(self):
        if self.raycasting_intersections:
            for intersection in self.raycasting_intersections:
                if intersection[0] != None and intersection[1] != None:
                    x, y = self.transform_coordinates(*intersection)
                    pygame.draw.circle(self.window, (128, 0, 128), (x, y), 5)

    def game_loop(self):
        step = 0
        while True:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.render_step(step)
            step += 1
            self.update_display()

    def render_step(self, step):
        self.window.fill(self.background_color)
        self.draw_grid()
        self.draw_obstacles()
        self.draw_agents(step)
        self.draw_goals()
        draw_text(self.window, f"step: {step}", self.window_width - 10, 10)

    def render_step_with_agents2(self, agents, step):
        self._pygame_event_manager()
        if not self._rendering_is_active:
            return
        self.window.fill(self.color_scheme.background_color)
        self.draw_grid()
        self.draw_obstacles()

        for agent_data in agents:
            agent_id = agent_data[0]
            x, y = agent_data[1], agent_data[2]
            velocity = agent_data[3]  # Current velocity
            pref_velocity = agent_data[4]  # Preferred velocity
            distance_to_goal = agent_data[5]  # Distance to goal
            radius = self.agent_radii.get(agent_id, 10)

            x_screen, y_screen = self.transform_coordinates(x, y)

            # Get the agent's behavior
            behaviour = self.agent_behaviours.get(agent_id, "default")
            # Get the appropriate color for the agent
            agent_color = self.color_scheme.get_agent_color(behaviour)
            pygame.draw.circle(self.window, agent_color,
                               (x_screen, y_screen), int(radius * self.cell_size))

            # Add text inside the agent circle
            draw_text(self.window, f"A_{agent_id}", x_screen, y_screen)
            
            # Draw the agent's detection radius (neighbor_dist)
            detection_radius = self.agent_neighbour_dist.get(agent_id, 10)
            # draw_detection_radius(self.window,
            #                       (x_screen, y_screen), detection_radius, cell_size=self.cell_size, color=self.color_scheme.detection_radius_color, border_width=2
            #                       )

            # Draw the current velocity arrow (red) with greater width
            # draw_arrow(self.window, (x_screen, y_screen), velocity, self.color_scheme.velocity_color,
            #            scale=100, width=16)

            # Draw the preferred velocity arrow (blue) with smaller width
            # draw_arrow(self.window, (x_screen, y_screen), pref_velocity, self.color_scheme.pref_velocity_color,
            #            scale=100, width=6)

            # Get the goal position
            # goal_x, goal_y = self.transform_coordinates(*self.goals[agent_id])

            # Draw the distance to goal line with perpendicular markers
            # draw_distance_to_goal(self.window,
            #                       (x_screen, y_screen), (goal_x,
            #                                              goal_y), color=self.color_scheme.distance_line_color, line_width=4
            #                       )

        draw_text(self.window, f"step: {step}", self.window_width - 150, 50)
        self.draw_intersections()
        self.draw_goals()
        self.update_display()
        pygame.time.delay(int(self.delay))

    def render_step_with_agents(self, agents, step):
        self._pygame_event_manager()
        if not self._rendering_is_active:
            return
        self.window.fill(self.color_scheme.background_color)
        self.draw_grid()
        self.draw_obstacles()

        for agent_data in agents:
            agent_id = agent_data[0]
            x, y = agent_data[1], agent_data[2]
            velocity = agent_data[3]  # Current velocity
            pref_velocity = agent_data[4]  # Preferred velocity
            distance_to_goal = agent_data[5]  # Distance to goal
            radius = self.agent_radii.get(agent_id, 10)

            x_screen, y_screen = self.transform_coordinates(x, y)

            # Get the agent's behavior
            behaviour = self.agent_behaviours.get(agent_id, "default")
            # Get the appropriate color for the agent
            agent_color = self.color_scheme.get_agent_color(behaviour)
            pygame.draw.circle(self.window, agent_color,
                               (x_screen, y_screen), int(radius * self.cell_size))

            # Add text inside the agent circle
            draw_text(self.window, f"A_{agent_id}", x_screen, y_screen)

            # Draw the agent's detection radius (neighbor_dist)
            detection_radius = self.agent_neighbour_dist.get(agent_id, 10)
            # Properly scale the detection radius
            # scaled_detection_radius = detection_radius * self.cell_size

            # Check if the radius is large enough to be drawn
            # if detection_radius > 0:
            #     draw_detection_radius(
            #         self.window, (x_screen, y_screen), detection_radius,
            #         color=self.color_scheme.detection_radius_color, border_width=2, cell_size=self.cell_size
            #     )

            # Draw the current velocity arrow (red) with greater width
            # draw_arrow(self.window, (x_screen, y_screen), velocity, self.color_scheme.velocity_color,
            #            scale=100, width=16)

            # Draw the preferred velocity arrow (blue) with smaller width
            # draw_arrow(self.window, (x_screen, y_screen), pref_velocity, self.color_scheme.pref_velocity_color,
            #            scale=100, width=6)

            # Get the goal position
            # goal_x, goal_y = self.transform_coordinates(*self.goals[agent_id])

            # Draw the distance to goal line with perpendicular markers
            # draw_distance_to_goal(self.window,
            #                       (x_screen, y_screen), (goal_x,
            #                                              goal_y), color=self.color_scheme.distance_line_color, line_width=4
            #                       )

        draw_text(self.window, f"step: {step}", self.window_width - 150, 50)
        self.draw_goals()
        self.draw_intersections()
        self.update_display()
        pygame.time.delay(int(self.delay))

    def update_display(self):
        pygame.display.flip()
        self.clock.tick(60)

    def dispose(self):
        pygame.quit()

    def update(self, message):
        """
        Method that will be called when the `subject` notifies its observers.
        """
        if isinstance(message, SimulationInitializedMessage):
            print("Simulation initialized.")
            # Save the agent radii and neighbourDist
            self.agent_radii = {agent_data["agent_id"]: agent_data["radius"]
                                for agent_data in message.agent_initialization_data}
            self.agent_neighbour_dist = {agent_data["agent_id"]: agent_data["neighbor_dist"]
                                         for agent_data in message.agent_initialization_data}
            self.agent_behaviours = {
                agent_data["agent_id"]: agent_data["behaviour"]
                for agent_data in message.agent_initialization_data
            }
        elif isinstance(message, AgentPositionsUpdateMessage):
            self.render_step_with_agents(message.agent_positions, message.step)
        elif isinstance(message, ObstaclesProcessedMessage):
            self.obstacles_processed(message.obstacles)
        elif isinstance(message, GoalsProcessedMessage):
            self.goals_processed(message.goals)
        elif isinstance(message, GoalPositionUpdatedMessage):
            self.goal_position_updated(message.goal_id, message.new_position)
        elif isinstance(message, NewObstacleAddedMessage):
            self.new_obstacle_added(message.obstacle)
        elif isinstance(message, RayCastingUpdateMessage):
            self.raycasting_updated(message.intersections)
        
    def raycasting_updated(self, intersections):
        self.raycasting_intersections = intersections

    def obstacles_processed(self, obstacles: list):
        self.obstacles = obstacles

    def goals_processed(self, goals: dict):
        # Ensure that the goals are processed correctly
        self.goals = goals
        # print(f"Processed {len(goals)} goals.")

    def goal_position_updated(self, goal_id: int, new_position: tuple):
        if goal_id in self.goals:
            self.goals[goal_id] = new_position

    def new_obstacle_added(self, obstacle: list):
        self.obstacles.append(obstacle)


if __name__ == '__main__':
    # Basic configuration for testing
    grid = Grid(1000, 1000, 20)
    renderer = PyGameRenderer(1000, 1000, grid=grid, cell_size=grid.spacing)
    renderer.setup()

    # Simple loop to keep the window open
    try:
        renderer.game_loop()
    except KeyboardInterrupt:
        renderer.dispose()
