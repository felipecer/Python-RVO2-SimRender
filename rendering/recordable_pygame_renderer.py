# recordable_renderer.py
import pygame
import numpy as np
from rendering.pygame_renderer import PyGameRenderer

class RecordablePyGameRenderer(PyGameRenderer):
    """
    Inherits from PyGameRenderer and extends it so that after drawing,
    we capture the frame as a NumPy array (rgb array).
    """

    def __init__(self, *args, record_all=False, **kwargs):
        """
        :param record_all: If True, store every frame in self.all_frames.
                           Otherwise, we just keep the most recent in self.last_frame.
        """
        super().__init__(*args, **kwargs)
        self.record_all = record_all
        blank = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        self.last_frame = blank
        self.all_frames = []  # Optional: store all frames
        

    def render_step_with_agents(self, agents, step):
        """
        Override the parent method. After the parent draws, capture the frame.
        """
        super().render_step_with_agents(agents, step)
        self._capture_frame()

    def _capture_frame(self):
        """Internal helper to capture the current PyGame display to a NumPy array."""
        surface = pygame.display.get_surface()
        frame_array = pygame.surfarray.array3d(surface)
        # Convert from (width, height, 3) to (height, width, 3)
        frame_array = np.transpose(frame_array, (1, 0, 2))

        self.last_frame = frame_array
        if self.record_all:
            # Store a copy if we want to keep all frames
            self.all_frames.append(frame_array.copy())

    def get_rgb_array(self):
        """
        Returns the latest captured frame (NumPy array of shape [height, width, 3])
        or None if no frame has been rendered yet.
        """
        return self.last_frame
