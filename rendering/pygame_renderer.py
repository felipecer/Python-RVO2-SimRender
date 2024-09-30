#!/usr/bin/env python
import math
from pprint import pprint
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

from rendering.drawing_utils import draw_text, draw_arrow, draw_detection_radius, draw_distance_to_goal, Grid
from rendering.color_scheme_parser import ColorScheme

# Importar el parser para cargar esquemas de color
from rendering.color_scheme_parser import load_color_schemes


class PyGameRenderer(RendererInterface, SimulationObserver):
    def __init__(self, width, height, color_scheme_file='./rendering/color_schemes.yaml', color_scheme_name='orca-behaviors',
                 map=None, simulation_steps={}, obstacles=[], goals={}, agents=[], display_caption='Simulador de Navegación de Agentes',
                 font_size=36, font_name='arial', cell_size=50):
        # Cargar el esquema de colores
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
            # Usar el mismo radio del agente
            radius = self.agent_radii.get(agent_id, 10)
            # Dibujar el círculo de la meta con el mismo radio que el agente
            pygame.draw.circle(self.window, self.color_scheme.goal_color, (x, y),
                               int(radius * self.cell_size))

            # Agregar texto dentro del círculo de la meta
            draw_text(self.window, f"G_{agent_id}", x, y)

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
            velocity = agent_data[3]  # Velocidad actual
            pref_velocity = agent_data[4]  # Velocidad preferida
            distance_to_goal = agent_data[5]  # Distancia a la meta
            radius = self.agent_radii.get(agent_id, 10)

            x_screen, y_screen = self.transform_coordinates(x, y)
            pygame.draw.circle(self.window, self.color_scheme.agent_color,
                               (x_screen, y_screen), int(radius * self.cell_size))

            # Agregar texto dentro del círculo del agente
            draw_text(self.window, f"A_{agent_id}", x_screen, y_screen)

            # Dibujar el radio de detección del agente (neighbor_dist)
            detection_radius = self.agent_neighbour_dist.get(agent_id, 10)
            draw_detection_radius(self.window,
                                  (x_screen, y_screen), detection_radius, cell_size=self.cell_size, color=self.color_scheme.detection_radius_color, border_width=2
                                  )

            # Dibujar la flecha de la velocidad actual (rojo) con ancho mayor
            draw_arrow(self.window, (x_screen, y_screen), velocity, self.color_scheme.velocity_color,
                       scale=100, width=16)

            # Dibujar la flecha de la velocidad preferida (azul) con ancho menor
            draw_arrow(self.window, (x_screen, y_screen), pref_velocity, self.color_scheme.pref_velocity_color,
                       scale=100, width=6)

            # Obtener la posición de la meta
            goal_x, goal_y = self.transform_coordinates(*self.goals[agent_id])

            # Dibujar la línea de distancia a la meta con marcadores perpendiculares
            draw_distance_to_goal(self.window,
                                  (x_screen, y_screen), (goal_x,
                                                         goal_y), color=self.color_scheme.distance_line_color, line_width=4
                                  )

        draw_text(self.window, f"step: {step}", self.window_width - 150, 50)
        self.draw_goals()
        self.update_display()
        pygame.time.delay(int(self.delay))

    def update_display(self):
        pygame.display.flip()
        self.clock.tick(60)

    def dispose(self):
        pygame.quit()

    def update(self, message):
        """
        Método que será llamado cuando el `subject` notifique a sus observadores.
        """
        if isinstance(message, SimulationInitializedMessage):
            print("Simulación inicializada.")
            # Guardar los radios de los agentes y neighbourDist
            self.agent_radii = {agent_data["agent_id"]: agent_data["radius"]
                                for agent_data in message.agent_initialization_data}
            self.agent_neighbour_dist = {agent_data["agent_id"]: agent_data["neighbor_dist"]
                                         for agent_data in message.agent_initialization_data}
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

    def obstacles_processed(self, obstacles: list):
        self.obstacles = obstacles

    def goals_processed(self, goals: dict):
        # Asegurarte de que las metas se procesen correctamente
        self.goals = goals
        print(f"Processed {len(goals)} goals.")

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
