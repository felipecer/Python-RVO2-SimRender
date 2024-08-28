#!/usr/bin/env python
import pygame
# import pygame_gui
# from pygame_gui.elements import UIHorizontalSlider
import sys
from rendering.interfaces import RendererInterface
from simulator.models.observer import SimulationObserver
from simulator.models.messages import (
    AgentPositionsUpdateMessage,
    SimulationInitializedMessage,
    ObstaclesProcessedMessage,
    GoalsProcessedMessage,
    GoalPositionUpdatedMessage,
    NewObstacleAddedMessage
)

class Grid:
    def __init__(self, window_width, window_height, spacing):
        self.window_width = window_width
        self.window_height = window_height
        self.spacing = spacing

    def draw(self, window):
        color = (180, 203, 211)
        color_axis = (0, 0, 0)

        # Draw vertical lines
        for x in range(0, self.window_width // 2, self.spacing):
            pygame.draw.line(window, color, (self.window_width // 2 + x, 0),
                             (self.window_width // 2 + x, self.window_height))
            pygame.draw.line(window, color, (self.window_width // 2 - x, 0),
                             (self.window_width // 2 - x, self.window_height))

            # Draw small vertical axes every 10 cells
            if ((x/self.spacing) % 10) == 0:
                # pygame.draw.line(window, color_axis, (self.window_width // 2 - x, 0), (self.window_width // 2 - x, self.window_height))
                pygame.draw.line(window, color_axis, (self.window_width // 2 - x, self.window_height // 2 -
                                 self.spacing // 2), (self.window_width // 2 - x, self.window_height // 2 + self.spacing // 2))
                pygame.draw.line(window, color_axis, (self.window_width // 2 + x, self.window_height // 2 -
                                 self.spacing // 2), (self.window_width // 2 + x, self.window_height // 2 + self.spacing // 2))

        # Draw horizontal lines
        for y in range(0, self.window_height // 2, self.spacing):
            pygame.draw.line(window, color, (0, self.window_height // 2 + y),
                             (self.window_width, self.window_height // 2 + y))
            pygame.draw.line(window, color, (0, self.window_height // 2 - y),
                             (self.window_width, self.window_height // 2 - y))

            # Draw small horizontal axes every 10 cells
            if ((y/self.spacing) % 10) == 0:
                pygame.draw.line(window, color_axis, (self.window_width // 2 - self.spacing // 2, self.window_width //
                                 2 - y), (self.window_width // 2 + self.spacing // 2, self.window_height // 2 - y))
                pygame.draw.line(window, color_axis, (self.window_width // 2 - self.spacing // 2, self.window_width //
                                 2 + y), (self.window_width // 2 + self.spacing // 2, self.window_height // 2 + y))

        # Draw center lines
        pygame.draw.line(window, color_axis, (self.window_width //
                         2, 0), (self.window_width // 2, self.window_height))
        pygame.draw.line(window, color_axis, (0, self.window_height // 2),
                         (self.window_width, self.window_height // 2))


class PyGameRenderer(RendererInterface, SimulationObserver):
    def __init__(self, width, height, map=None, simulation_steps={}, obstacles=[], goals={}, agents=[], display_caption='Simulador de Navegación de Agentes', font_size=36, font_color=(0, 0, 0), font_name='arial', cell_size=50):
        self.font_name = font_name
        self.font_size = font_size
        self.font_color = font_color
        self.map = map
        self.obstacles = obstacles
        self.goals = goals
        self.clock = pygame.time.Clock()
        self.agents = agents
        self.simulation_steps = simulation_steps
        self.cell_size = cell_size        
        self.grid = Grid(width, height, cell_size)   

        # Window settings
        self.window = None
        self.window_width, self.window_height = width, height
        self.display_caption = display_caption
        # Colors
        self.agent_color = (0, 255, 0)  # Verde
        self.obstacle_color = (255, 0, 0)  # Rojo
        self.background_color = (195, 215, 224)  # Blanco
        self._rendering_is_active = False

        # UI Manager

        # self.ui_manager = pygame_gui.UIManager(
        #     (self.window_width, self.window_height))
        # self.ui_manager.ui_theme.load_fonts()
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

    def draw_text(self, text, x, y):
        font = pygame.font.Font(pygame.font.match_font(
            self.font_name), self.font_size)
        text_surface = font.render(text, True, self.font_color)
        text_rect = text_surface.get_rect()
        text_rect.topright = (x, y)
        self.window.blit(text_surface, text_rect)

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
                self.window, self.obstacle_color, vertices_transformed, 3)

    def draw_agents(self, step):
        if step in self.simulation_steps:
            for agent_id, x, y in self.simulation_steps[step]:
                x, y = self.transform_coordinates(x, y)
                pygame.draw.circle(self.window, self.agent_color, (x, y), 10)

    def draw_goals(self):
        goals = self.goals.values()
        for goal in goals:
            x, y = self.transform_coordinates(*goal)
            pygame.draw.circle(self.window, self.obstacle_color, (x, y), 10)

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
        self.draw_text(f"step: {step}", self.window_width - 10, 10)

    def render_step_with_agents(self, agents, step):
        self._pygame_event_manager()
        if not self._rendering_is_active:
            return
        self.window.fill(self.background_color)
        self.draw_grid()
        self.draw_obstacles()
        for agent_id, x, y in agents:
            x, y = self.transform_coordinates(x, y)
            pygame.draw.circle(self.window, self.agent_color, (x, y), 10)
        self.draw_text(f"step: {step}", self.window_width - 10, 10)
        self.draw_goals()
        self.update_display()
        pygame.time.delay(int(self.delay))

    def update_display(self):
        pygame.display.flip()
        self.clock.tick(60)

    def dispose(self):
        pygame.quit()

    # Métodos de SimulationObserver
    def update(self, message):
        """
        Método que será llamado cuando el `subject` notifique a sus observadores.
        """
        if isinstance(message, AgentPositionsUpdateMessage):
            self.render_step_with_agents(message.agent_positions, message.step)
        elif isinstance(message, SimulationInitializedMessage):
            print("Simulación inicializada.")
        elif isinstance(message, ObstaclesProcessedMessage):
            self.obstacles_processed(message.obstacles)
        elif isinstance(message, GoalsProcessedMessage):
            self.goals_processed(message.goals)
        elif isinstance(message, GoalPositionUpdatedMessage):
            self.goal_position_updated(message.goal_id, message.new_position)
        elif isinstance(message, NewObstacleAddedMessage):
            self.new_obstacle_added(message.obstacle)

    def obstacles_processed(self, obstacles: list):
        self.obstacles = obstacles

    def goals_processed(self, goals: dict):
        self.goals = goals

    def goal_position_updated(self, goal_id: int, new_position: tuple):
        if goal_id in self.goals:
            self.goals[goal_id] = new_position

    def new_obstacle_added(self, obstacle: list):
        self.obstacles.append(obstacle)

if __name__ == '__main__':
    # Configuración básica para pruebas
    grid = Grid(1000, 1000, 20)
    renderer = PyGameRenderer(1000, 1000, grid=grid, cell_size=grid.spacing)
    renderer.setup()

    # Bucle simple para mantener la ventana abierta
    try:
        renderer.game_loop()
    except KeyboardInterrupt:
        renderer.dispose()
