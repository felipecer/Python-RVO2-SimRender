#!/usr/bin/env python
import math
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
                 font_size=8, font_name='arial', cell_size=50, show_goals='all', intelligent_agent_id=0):
        # Load the color scheme
        self.color_scheme_config = load_color_schemes(color_scheme_file)
        self.color_scheme = self.color_scheme_config.schemes[color_scheme_name]
        self.show_goals = show_goals
        self.max_ray_length = 18
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
        self.intelligent_agent_id = intelligent_agent_id

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
                if agent_id == self.intelligent_agent_id:
                    pygame.draw.circle(
                        self.window, self.color_scheme.obstacle_color, (x, y), 10)
                else: 
                    pygame.draw.circle(
                        self.window, self.color_scheme.obstacle_color, (x, y), 10)

    def draw_goals(self):
        if len(self.goals) == 0:
            return
        if self.show_goals == 'none':
            return
        if self.show_goals == 'all':
            for agent_id, goal in self.goals.items():
                x, y = self.transform_coordinates(*goal)
                # Use the same radius as the agent
                radius = self.agent_radii.get(agent_id, 10)
                # Draw the goal circle with the same radius as the agent
                pygame.draw.circle(self.window, self.color_scheme.goal_color, (x, y),
                                int(radius * self.cell_size))
                # Add text inside the goal circle
                draw_text(self.window, f"G_{agent_id}", x, y)
        elif self.show_goals == 'intelligent_agent':
            goal = self.goals[self.intelligent_agent_id]
            x,y = self.transform_coordinates(*goal)
            radius = self.agent_radii.get(self.intelligent_agent_id, 10) 
            pygame.draw.circle(self.window, self.color_scheme.goal_color, (x, y),
                                int(radius * self.cell_size))
            # Add text inside the goal circle
            draw_text(self.window, f"G_{self.intelligent_agent_id}", x, y)

    def draw_intersections(self, agent_x, agent_y):
        # if self.raycasting_intersections.any():
        if self.raycasting_intersections is None:
            return
        # if len(self.raycasting_intersections) == 0:
        #     return
        max_ray_length = self.max_ray_length
        for intersection in self.raycasting_intersections:
            if intersection[2] != 0:                
                x = agent_x 
                y = agent_y
                ray_ang = intersection[0]
                ray_len = intersection[1] * max_ray_length  # Correctly scale ray_len
                ray_x = x + ray_len * math.cos(ray_ang)
                ray_y = y + ray_len * math.sin(ray_ang)
                t_ray_x, t_ray_y = self.transform_coordinates(ray_x, ray_y)                    
                pygame.draw.circle(self.window, (255, 105, 180), (t_ray_x, t_ray_y), 2)

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

        draw_text(self.window, f"Step: {step}", self.window_width - 150, 50)
        self.draw_goals()
        self.draw_intersections(agents[0][1], agents[0][2])
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
