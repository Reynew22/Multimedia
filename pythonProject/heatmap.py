import cv2
import numpy as np

class MotionHeatmap:
    def __init__(self, shape):
        self.accumulator = np.zeros(shape[:2], dtype=np.float32)
        self.prev_gray = None

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        # Frame difference
        diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 1.0, cv2.THRESH_BINARY)
        self.accumulator += thresh
        self.prev_gray = gray

        return self.get_overlay()

    def get_overlay(self):
        norm = cv2.normalize(self.accumulator, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = np.uint8(norm)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap
