import unittest
from unittest.mock import patch
from rendering.pygame_renderer import PyGameRenderer
import os


class PyGameRendererTestCase(unittest.TestCase):
    def setUp(self):
        # Disable sound to avoid ALSA errors during tests
        # Use the 'dummy' audio driver to avoid initializing the sound system
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        self.renderer = PyGameRenderer(1000, 1000)

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
