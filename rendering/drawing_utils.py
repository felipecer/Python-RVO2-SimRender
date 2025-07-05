# drawing_utils.py

import pygame
import math


def desaturate_color(color, factor=0.5):
    """ Reduces the saturation of an RGB color by a factor """
    gray = sum(color) // 3  # Average of the RGB values to get a gray
    return tuple(int(gray + (c - gray) * factor) for c in color)


class Grid:
    def __init__(self, window_width, window_height, spacing, sim_scale=1.0, font_size=18, font_color=(0, 0, 0), origin_radius=10):
        self.window_width = window_width
        self.window_height = window_height
        self.spacing = spacing
        # Simulation scale (e.g. meters per cell)
        self.sim_scale = sim_scale
        self.font_size = font_size
        self.font_color = font_color
        self.origin_radius = origin_radius  # Radius of the point at the origin

    def draw_text(self, window, text, x, y):
        """Draws text on the window."""
        font = pygame.font.Font(
            pygame.font.match_font('arial'), self.font_size)
        text_surface = font.render(text, True, self.font_color)
        text_rect = text_surface.get_rect(center=(x, y))
        window.blit(text_surface, text_rect)

    def draw_origin(self, window):
        """Draws a black point at the origin of the axes."""
        color_origin = (0, 0, 0)  # Black for the origin
        center_x = self.window_width // 2
        center_y = self.window_height // 2
        pygame.draw.circle(window, color_origin,
                           (center_x, center_y), self.origin_radius)

    def draw(self, window):
        color = (180, 203, 211)
        color_axis = (0, 0, 0)

        # Length of the full markers (for cell_size)
        marker_length_full = 10
        # Length of the intermediate markers (for half of cell_size)
        marker_length_half = 5

        # Calculate the center of the screen
        center_x = self.window_width // 2
        center_y = self.window_height // 2

        # 1. Draw the point at the origin
        self.draw_origin(window)

        # 2. Draw long lines on the grid every cell_size
        for x in range(0, self.window_width // 2, self.spacing):
            pygame.draw.line(window, color, (center_x + x, 0),
                             (center_x + x, self.window_height))
            pygame.draw.line(window, color, (center_x - x, 0),
                             (center_x - x, self.window_height))

        for y in range(0, self.window_height // 2, self.spacing):
            pygame.draw.line(window, color, (0, center_y + y),
                             (self.window_width, center_y + y))
            pygame.draw.line(window, color, (0, center_y - y),
                             (self.window_width, center_y - y))

        # 3. Draw the axis markers (rulers) with lines and text
        for x in range(0, self.window_width // 2, self.spacing // 2):
            marker_length = marker_length_half if x % self.spacing != 0 else marker_length_full

            # Calculate the real-world coordinates for the X axis
            label_x = int((x / self.spacing) * self.sim_scale)

            # Draw markers on the X axis (above and below the central line)
            if x > 0:  # Avoid the origin
                if x % self.spacing == 0:
                    pygame.draw.line(window, color_axis,
                                     (center_x + x, center_y - marker_length),
                                     (center_x + x, center_y + marker_length), 2)
                    self.draw_text(
                        window, f"{label_x}", center_x + x, center_y + marker_length + 15)

                    pygame.draw.line(window, color_axis,
                                     (center_x - x, center_y - marker_length),
                                     (center_x - x, center_y + marker_length), 2)
                    self.draw_text(
                        window, f"{-label_x}", center_x - x, center_y + marker_length + 15)

                else:
                    pygame.draw.line(window, color_axis,
                                     (center_x + x, center_y - marker_length_half),
                                     (center_x + x, center_y + marker_length_half), 1)

                    pygame.draw.line(window, color_axis,
                                     (center_x - x, center_y - marker_length_half),
                                     (center_x - x, center_y + marker_length_half), 1)

        for y in range(0, self.window_height // 2, self.spacing // 2):
            marker_length = marker_length_half if y % self.spacing != 0 else marker_length_full

            # Calculate the real-world coordinates for the Y axis
            label_y = int((y / self.spacing) * self.sim_scale)

            # Draw markers on the Y axis (left and right of the central line)
            if y > 0:  # Avoid the origin
                if y % self.spacing == 0:
                    pygame.draw.line(window, color_axis,
                                     (center_x - marker_length, center_y + y),
                                     (center_x + marker_length, center_y + y), 2)
                    self.draw_text(
                        window, f"{-label_y}", center_x - marker_length - 20, center_y + y)

                    pygame.draw.line(window, color_axis,
                                     (center_x - marker_length, center_y - y),
                                     (center_x + marker_length, center_y - y), 2)
                    self.draw_text(
                        window, f"{label_y}", center_x - marker_length - 20, center_y - y)

                else:
                    pygame.draw.line(window, color_axis,
                                     (center_x - marker_length_half, center_y + y),
                                     (center_x + marker_length_half, center_y + y), 1)

                    pygame.draw.line(window, color_axis,
                                     (center_x - marker_length_half, center_y - y),
                                     (center_x + marker_length_half, center_y - y), 1)
        # 1. Draw the point at the origin
        self.draw_origin(window)


def draw_text(window, text, x, y, font_name='arial', font_size=8, font_color=(0, 0, 0)):
    font = pygame.font.Font(pygame.font.match_font(font_name), font_size)
    text_surface = font.render(text, True, font_color)
    text_rect = text_surface.get_rect(center=(x, y))
    window.blit(text_surface, text_rect)


def draw_arrow(window, start_pos, velocity, color, scale=50, width=2):
    """Draws an arrow from the `start_pos` in the direction of the `velocity` vector."""
    end_pos = (
        start_pos[0] + velocity[0] * scale,
        start_pos[1] - velocity[1] * scale
    )

    pygame.draw.line(window, color, start_pos, end_pos, width)

    velocity_magnitude = math.sqrt(velocity[0]**2 + velocity[1]**2)
    arrow_head_length = max(10, width * 1.5)
    angle = math.atan2(start_pos[1] - end_pos[1], end_pos[0] - start_pos[0])

    arrow_left = (
        end_pos[0] - arrow_head_length * math.cos(angle + math.pi / 6),
        end_pos[1] + arrow_head_length * math.sin(angle + math.pi / 6)
    )
    arrow_right = (
        end_pos[0] - arrow_head_length * math.cos(angle - math.pi / 6),
        end_pos[1] + arrow_head_length * math.sin(angle - math.pi / 6)
    )

    pygame.draw.line(window, color, end_pos, arrow_left, width)
    pygame.draw.line(window, color, end_pos, arrow_right, width)


def draw_distance_to_goal(window, agent_position, goal_position, color=(0, 0, 0), line_width=2, marker_length=10):
    """Draws a line from the agent's position to its goal with perpendicular markers at the ends."""
    pygame.draw.line(window, color, agent_position, goal_position, line_width)

    angle = math.atan2(
        goal_position[1] - agent_position[1], goal_position[0] - agent_position[0])

    agent_marker_start = (
        agent_position[0] + marker_length * math.cos(angle + math.pi / 2),
        agent_position[1] + marker_length * math.sin(angle + math.pi / 2)
    )
    agent_marker_end = (
        agent_position[0] + marker_length * math.cos(angle - math.pi / 2),
        agent_position[1] + marker_length * math.sin(angle - math.pi / 2)
    )
    pygame.draw.line(window, color, agent_marker_start,
                     agent_marker_end, line_width)

    goal_marker_start = (
        goal_position[0] + marker_length * math.cos(angle + math.pi / 2),
        goal_position[1] + marker_length * math.sin(angle + math.pi / 2)
    )
    goal_marker_end = (
        goal_position[0] + marker_length * math.cos(angle - math.pi / 2),
        goal_position[1] + marker_length * math.sin(angle - math.pi / 2)
    )
    pygame.draw.line(window, color, goal_marker_start,
                     goal_marker_end, line_width)


def draw_detection_radius(window, position, radius, cell_size, color, border_width):
    """Draws a circle representing an agent's detection radius."""
    scaled_radius = int(radius * cell_size)
    interior_color = tuple(min(255, int(c * 1.5)) for c in color)

    surface = pygame.Surface(
        (scaled_radius * 2, scaled_radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(surface, (*interior_color, 80),
                       (scaled_radius, scaled_radius), scaled_radius)

    window.blit(
        surface, (position[0] - scaled_radius, position[1] - scaled_radius))
    pygame.draw.circle(window, color, position, scaled_radius, border_width)
