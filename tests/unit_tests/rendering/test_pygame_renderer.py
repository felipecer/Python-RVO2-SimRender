import unittest
from rendering.pygame_renderer import PyGameRenderer

class PyGameRendererTestCase(unittest.TestCase):    
    def setUp(self):
        self.pygame_renderer = PyGameRenderer(1000,1000)

    def tearDown(self):
        pass

    def test_transform_coordinates_center(self):      
      x, y = self.pygame_renderer.transform_coordinates(0, 0)
      self.assertEqual(x, 500)
      self.assertEqual(y, 500)

    def test_transform_coordinates_positive(self):      
      x, y = self.pygame_renderer.transform_coordinates(1, 1)
      self.assertEqual(x, 600)  # 500 + 1*100
      self.assertEqual(y, 400)  # 500 - 1*100

    def test_transform_coordinates_negative(self):      
      x, y = self.pygame_renderer.transform_coordinates(-1, -1)
      self.assertEqual(x, 400)  # 500 - 1*100
      self.assertEqual(y, 600)  # 500 + 1*100             