import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GAZE_ANGLE_X = ' gaze_angle_x'
GAZE_ANGLE_Y = ' gaze_angle_y'
GAZE_0_X = ' gaze_0_x'
GAZE_0_Y = ' gaze_0_y'
GAZE_0_Z = ' gaze_0_z'
GAZE_1_X = ' gaze_1_x'
GAZE_1_Y = ' gaze_1_y'
GAZE_1_Z = ' gaze_1_z'

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

def edge_projection(row):
    # frame shape is (width, height)
    x_plane_normal = np.array([1, 0, 0])
    if row[GAZE_0_X] > 0:
        # participant is looking right
        x_plane_point = np.array([FRAME_WIDTH, 0, 0])
    else:
        # participant is looking left
        x_plane_point = np.array([0, 0, 0])
    y_plane_normal = np.array([0, 1, 0])
    if row[GAZE_0_Y] > 0:
        # participant is looking down
        y_plane_point = np.array([0, FRAME_HEIGHT, 0])
    else:
        # participant is looking up
        y_plane_point = np.array([0, 0, 0])
    line_point = np.array([row[' eye_lmk_x_0'], row[' eye_lmk_y_0'], 0])
    line_vector = np.array([row[GAZE_0_X], row[GAZE_0_Y], row[GAZE_0_Z]])
    x_t = np.dot((x_plane_point - line_point), x_plane_normal) / np.dot(line_vector, x_plane_normal)
    y_t = np.dot((y_plane_point - line_point), y_plane_normal) / np.dot(line_vector, y_plane_normal)
    if np.abs(x_t) < np.abs(y_t):
        # gaze falls on the left/ri, row[' eye_lmk_z_0'ower edge
        return (line_point + line_vector * y_t).astype(int)[[0,1]]

def screen_projection(row):
    '''
    Uses 3D mm values for eye positions
    '''
        # frame shape is (width, height)
    z_plane_normal = np.array([0, 0, 1])
    z_plane_point = np.array([0, 0, 0])

    line_point = np.array([row[' eye_lmk_X_0'], row[' eye_lmk_Y_0'], row[' eye_lmk_Z_0']])
    line_vector = np.array([row[GAZE_0_X], row[GAZE_0_Y], row[GAZE_0_Z]])
    z_t = np.dot((z_plane_point - line_point), z_plane_normal) / np.dot(line_vector, z_plane_normal)
    return (line_point + line_vector *z_t).astype(int)[[0, 1]]

def sphere_projection(row):
    
 
def multi_projection(row):
    r1 = edge_projection(row)
    r2 = screen_projection(row)
    return np.array([r1[0],r1[1],r2[0],r2[1]]).astype(int)

def no_projection(row):
    return np.array([row[GAZE_ANGLE_X], row[GAZE_ANGLE_Y], row[' eye_lmk_x_0'], row[' eye_lmk_y_0']])

################################################ Variables to set
CAMERA = "camera2"
#If you want to cluster for a single group, otherwise set to none
GROUP_NAME=None

EVAL_GROUP = 'CA'
#Export title
MODEL_TITLE = "KMEANS_projection_3D_"
#Projection Function either edge_projection (2D) or screen_projection (3D), mult_projection (both), no_projection (None)
PROJECTION = screen_projection
#number of features, 4 for multi_projection, 2 for the rest
N_FEATURES = 5
#number of CLUSTERS
N_CLUSTERS = 10

#Evaluation features used
def vec(row):
    return PROJECTION(row)
################################################


PICKLE_TITLE = MODEL_TITLE+CAMERA+"_clustering.pickle"
FILE_NAME = "~/Desktop/jibo-survival-trimming/Data_Classification/shrink_data_"+CAMERA+".csv"

USED_COLS = [GAZE_ANGLE_X,GAZE_ANGLE_Y,GAZE_0_X,GAZE_0_Y,GAZE_0_Z,GAZE_1_X,GAZE_1_Y,GAZE_1_Z,' eye_lmk_x_0',' eye_lmk_y_0',' eye_lmk_X_0',' eye_lmk_Y_0',' eye_lmk_Z_0']


CSV_FILE = "/media/sebo-hri-lab/DATA/OpenFace/group_"+EVAL_GROUP+"_"+CAMERA+"_trim.csv"
VIDEO_FILE = "/media/sebo-hri-lab/DATA/Trimmed_Videos/group_"+EVAL_GROUP+"_"+CAMERA+"_trim.mp4"

