#! /usr/bin/python3

import numpy as np

# class GazeAngle:
#     def __init__(self, direction=[0,0,0], angle_span=0):
#         self.direction = np.array(direction)
#         self.angle_span = angle_span

#     def within_region(coordinates):
#         dot_product = np.dot()
#         angle_between = np.arccos()

class GazeRegion:
    def __init__(self, points):
        self.points = points
        self.left = np.amin(points[:,0])
        self.top = np.amin(points[:,1])
        self.right = np.amax(points[:,0])
        self.bottom = np.amax(points[:,1])

    def within_region(point):
        pass

#GAZE_3 = {"participant1": GazeAngle([], 500), "participant3": GazeAngle()}