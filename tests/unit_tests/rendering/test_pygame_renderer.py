import unittest
from unittest.mock import patch, Mock
from rendering.pygame_renderer import PyGameRenderer
import os


class PyGameRendererTestCase(unittest.TestCase):
    @patch('pygame.display.set_mode')
    def setUp(self, mock_set_mode):
        # Disable sound to avoid ALSA errors during tests
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        self.mock_window = mock_set_mode.return_value
        self.renderer = PyGameRenderer(1000, 1000)
        self.renderer.setup()

    def tearDown(self):
        self.renderer.dispose()

    @patch('pygame.display.set_mode')
    def test_setup(self, mock_set_mode):
        self.renderer.setup()
        mock_set_mode.assert_called_once_with((1000, 1000))
        self.assertTrue(self.renderer.rendering_is_active)

    def test_load_simulation_steps_file(self):
        test_file_content = "0,1,100,100\n1,2,200,200"
        with patch("builtins.open", unittest.mock.mock_open(read_data=test_file_content)) as mock_file:
            self.renderer.load_simulation_steps_file("dummy_path")
            self.assertIn(0, self.renderer.simulation_steps)
            self.assertIn(1, self.renderer.simulation_steps)
            self.assertEqual(len(self.renderer.simulation_steps[0]), 1)
            self.assertEqual(
                self.renderer.simulation_steps[0][0], (1, 100.0, 100.0))
            self.assertEqual(
                self.renderer.simulation_steps[1][0], (2, 200.0, 200.0))
            mock_file.assert_called_with("dummy_path", 'r')

    def test_load_obstacles_file(self):
        test_obstacles_content = "1,0,0,100,100\n2,200,200,300,300"
        with patch("builtins.open", unittest.mock.mock_open(read_data=test_obstacles_content)) as mock_file:
            self.renderer.load_obstacles_file("dummy_path")
            self.assertEqual(len(self.renderer.obstacles), 2)
            self.assertEqual(self.renderer.obstacles[0], [
                             (0.0, 0.0), (100.0, 100.0)])
            self.assertEqual(self.renderer.obstacles[1], [
                             (200.0, 200.0), (300.0, 300.0)])
            mock_file.assert_called_with("dummy_path", 'r')

    def test_load_goals_file(self):
        test_goals_content = "100,100\n200,200"
        with patch("builtins.open", unittest.mock.mock_open(read_data=test_goals_content)) as mock_file:
            self.renderer.load_goals_file("dummy_path")
            self.assertEqual(self.renderer.goals[0], (100, 100))
            self.assertEqual(self.renderer.goals[1], (200, 200))
            mock_file.assert_called_with("dummy_path", 'r')

    @patch('pygame.draw.polygon')
    def test_draw_obstacles(self, mock_draw_polygon):
        self.renderer.obstacles = [[(0, 0), (100, 100)], [
            (200, 200), (300, 300)]]
        self.renderer.window = True  # Assuming the window is initialized
        self.renderer.draw_obstacles()
        self.assertEqual(mock_draw_polygon.call_count, 2)

    @patch('pygame.draw.circle')
    def test_draw_goals(self, mock_draw_circle):
        self.renderer.goals = {0: (0, 0), 1: (100, 100)}
        self.renderer.window = True  # Assuming the window is initialized
        self.renderer.draw_goals()
        self.assertEqual(mock_draw_circle.call_count, 2)

    def test_draw_text(self):
        # Assume setup has already created self.mock_window via setUp
        text = "Test"
        x = 500
        y = 500
        self.renderer.draw_text(text, x, y)
        self.mock_window.blit.assert_called()

    @patch('pygame.draw.line')
    def test_draw_grid(self, mock_draw_line):
        spacing = 100
        self.renderer.draw_grid(spacing)
        expected_lines_count = (self.renderer.window_width // spacing + 1) + \
            (self.renderer.window_height // spacing + 1)

        self.assertEqual(mock_draw_line.call_count, expected_lines_count)

    def test_transform_coordinates_center(self):
        x, y = self.renderer.transform_coordinates(0, 0)
        self.assertEqual(x, 500)
        self.assertEqual(y, 500)

    def test_transform_coordinates_positive(self):
        x, y = self.renderer.transform_coordinates(1, 1)
        self.assertEqual(x, 600)  # 500 + 1*100
        self.assertEqual(y, 400)  # 500 - 1*100

    def test_transform_coordinates_negative(self):
        x, y = self.renderer.transform_coordinates(-1, -1)
        self.assertEqual(x, 400)  # 500 - 1*100
        self.assertEqual(y, 600)  # 500 + 1*100

    def test_dispose(self):
        with patch('pygame.quit') as mock_pygame_quit:
            self.renderer.dispose()
            mock_pygame_quit.assert_called_once()

    @patch('pygame.display.flip')
    def test_update_display(self, mock_flip):
        self.renderer.update_display()
        mock_flip.assert_called_once()

    def test_render_step(self):
        with patch.object(self.renderer, 'draw_grid') as mock_draw_grid, \
                patch.object(self.renderer, 'draw_obstacles') as mock_draw_obstacles, \
                patch.object(self.renderer, 'draw_agents') as mock_draw_agents, \
                patch.object(self.renderer, 'draw_goals') as mock_draw_goals, \
                patch.object(self.renderer, 'draw_text') as mock_draw_text:
            self.renderer.render_step(1)
            mock_draw_grid.assert_called_once()
            mock_draw_obstacles.assert_called_once()
            mock_draw_agents.assert_called_once_with(1)
            mock_draw_goals.assert_called_once()
            mock_draw_text.assert_called_once()
