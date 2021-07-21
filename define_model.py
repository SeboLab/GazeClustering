#! /usr/bin/python3

import numpy as np

class GazeAngle:
    def __init__(self, direction=[0,0,0], angle_span=0):
        self.direction = np.array(direction)
        self.angle_span = angle_span

    def within_region(coordinates):
        dot_product = np.dot()
        angle_between = np.arccos()

GAZE_3 = {"participant1": GazeAngle([], 500), "participant3": GazeAngle()}