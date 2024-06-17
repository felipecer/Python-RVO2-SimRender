import numpy as np


class GoalSpawner:
    def __init__(self, seed, num_iterations=20, max_radius=5, step_radius=1, empty_radius=1.5):
        self.seed = seed
        self.num_iterations = num_iterations
        self.max_radius = max_radius
        self.step_radius = step_radius
        self.empty_radius = empty_radius
        self.rng = np.random.default_rng(seed)
        self.generated_points = []
        self._generate_points()

    def _generate_points_in_annulus(self, inner_radius, outer_radius, num_points):
        points = []
        while len(points) < num_points:
            r = np.sqrt(self.rng.uniform(inner_radius**2, outer_radius**2))
            theta = self.rng.uniform(0, 2 * np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append((x, y))
        return points

    def _generate_points(self):
        self.generated_points = []
        current_radius = self.empty_radius
        while current_radius < self.max_radius:
            inner_radius = current_radius
            outer_radius = current_radius + self.step_radius
            annulus_area = np.pi * (outer_radius**2 - inner_radius**2)
            base_area = np.pi * self.step_radius**2
            num_points = int(self.num_iterations * (annulus_area / base_area))
            points = self._generate_points_in_annulus(
                inner_radius, outer_radius, num_points)
            self.generated_points.extend(points)
            current_radius += self.step_radius

    def reset(self, seed=None):
        if seed is None:
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = np.random.default_rng(seed=seed)
        self.generated_points = []
        self._generate_points()

    def get_next_goal(self):
        if not self.generated_points:
            self._generate_points()
        return self.generated_points.pop(0)
