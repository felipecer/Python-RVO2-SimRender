#!/usr/bin/env python
import pygame
import sys

class PyGameRenderer:
    def __init__(self, width, height, grid_scale= 100, map = None, simulation_steps = {}, obstacles = [], goals = {}, agents=[], display_caption = 'Simulador de Navegación de Agentes', font_size=36, font_color=(0, 0, 0), font_name='arial'):
        self.font_name = font_name
        self.font_size = font_size
        self.font_color = font_color
        self.map = map
        self.obstacles = obstacles
        self.goals = goals
        self.clock = pygame.time.Clock()
        self.agents = agents
        self.simulation_steps = simulation_steps 
        self.grid_scale = grid_scale              
        # Window settings
        self.window = None
        self.window_width, self.window_height = width, height        
        self.display_caption = display_caption
        # Colors
        self.agent_color = (0, 255, 0)  # Verde
        self.obstacle_color = (255, 0, 0)  # Rojo
        self.background_color = (255, 255, 255) # Blanco    
        self.rendering_is_active = False

    def _pygame_event_manager(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.rendering_is_active = False 
                break    

    def init_window(self):
        # Pygame Initialization
        pygame.init()
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption(self.display_caption)
        self.rendering_is_active = True

    def load_simulation_steps_file(self, file):
        simulation_steps = {}
        with open(file, 'r') as f:
            for line in f:
                line_split = line.strip().split(',')    
                step = int(line_split[0])
                agent_id = int(line_split[1])
                x, y = map(float, line_split[2:])
                if step not in simulation_steps:
                    simulation_steps[step] = []
                simulation_steps[step].append((agent_id, x, y))
        self.simulation_steps = simulation_steps

    def load_obstacles_file(self, file):
        obstacles = []
        with open(file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                obstacle_id = int(parts[0])
                vertices = []
                for i in range(1, len(parts), 2):
                    x_str = parts[i].strip("() ")
                    y_str = parts[i + 1].strip("() ")
                    x, y = float(x_str), float(y_str)
                    vertices.append((x, y))
                obstacles.append(vertices)
        self.obstacles = obstacles

    def load_goals_file(self, file):
        goals = {}
        with open(file, 'r') as f:
            i = 0
            for line in f:
                x, y = map(float, line.strip().split(','))
                goals[i] = (x, y)
                i += 1
        self.goals = goals

    def draw_text(self, text, x, y):
        font = pygame.font.Font(pygame.font.match_font(self.font_name), self.font_size)
        text_surface = font.render(text, True, self.font_color)
        text_rect = text_surface.get_rect()
        text_rect.topright = (x, y)
        self.window.blit(text_surface, text_rect)  

    def transform_coordinates(self, x, y):        
        scale = self.grid_scale
        x_new = self.window_width / 2 + x * scale
        y_new = self.window_height / 2 - y * scale
        return int(x_new), int(y_new)
    
    def draw_grid(self, spacing):
        color = (200, 200, 200)
        for x in range(0, self.window_width // 2, spacing):
            pygame.draw.line(self.window, color, (self.window_width // 2 + x, 0), (self.window_width // 2 + x, self.window_height))
            pygame.draw.line(self.window, color, (self.window_width // 2 - x, 0), (self.window_width // 2 - x, self.window_height))
        
        for y in range(0, self.window_height // 2, spacing):
            pygame.draw.line(self.window, color, (0, self.window_height // 2 + y), (self.window_width, self.window_height // 2 + y))
            pygame.draw.line(self.window, color, (0, self.window_height // 2 - y), (self.window_width, self.window_height // 2 - y))

        pygame.draw.line(self.window, color, (self.window_width // 2, 0), (self.window_width // 2, self.window_height))
        pygame.draw.line(self.window, color, (0, self.window_height // 2), (self.window_width, self.window_height // 2))

    def draw_terrain(self):
        pass
    
    def draw_obstacles(self):
        for obstacle in self.obstacles:
            vertices_transformed = [self.transform_coordinates(x, y) for x, y in obstacle]
            pygame.draw.polygon(self.window, self.obstacle_color, vertices_transformed, 3)    

    def draw_agents(self, step):
        if step in self.simulation_steps:
            for agent_id, x, y in self.simulation_steps[step]:
                x, y = self.transform_coordinates(x, y)
                pygame.draw.circle(self.window, self.agent_color, (x, y), 10)        
    
    def draw_goals(self):        
        for goal in self.goals.values():            
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
            pygame.display.flip()
            self.clock.tick(60)

    def render_step(self, step):
        self.window.fill(self.background_color)
        self.draw_grid(100)
        self.draw_obstacles()
        self.draw_agents(step)
        self.draw_goals()
        self.draw_text(f"step: {step}", self.window_width - 10, 10)

    def render_step_with_agents(self, agents, step):
        self._pygame_event_manager()
        if not self.rendering_is_active:
            return
        self.window.fill(self.background_color)
        self.draw_grid(100)
        self.draw_obstacles()        
        for agent_id, x, y in agents:
            x, y = self.transform_coordinates(x, y)
            pygame.draw.circle(self.window, self.agent_color, (x, y), 10)       
        self.draw_text(f"step: {step}", self.window_width - 10, 10)
        self.draw_goals()
        # self.draw_text(f"step: {step}", self.window_width - 10, 10)
        pygame.display.flip()
        self.clock.tick(60)

    def update_display(self):
        pygame.display.flip()
        self.clock.tick(60)
    
    def finish_simulation(self):
        pygame.quit()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python renderer.py <obstacles_file> <goals_file> <agents_file>")
        sys.exit(1)
    
    obstacles_file = sys.argv[1]
    goals_file = sys.argv[2]
    agents_file = sys.argv[3]

    renderer = PyGameRenderer(1000, 1000)
    renderer.load_obstacles_file(obstacles_file)
    renderer.load_goals_file(goals_file)
    renderer.load_simulation_steps_file(agents_file)
    renderer.init_window()
    renderer.game_loop()   
