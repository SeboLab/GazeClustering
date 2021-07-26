from os import EX_CANTCREAT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

'''
Predefined Variables

1) Column names
2) Video Frame Height
3) Projection Settings
'''
#1
GAZE_ANGLE_X = ' gaze_angle_x'
GAZE_ANGLE_Y = ' gaze_angle_y'
GAZE_0_X = ' gaze_0_x'
GAZE_0_Y = ' gaze_0_y'
GAZE_0_Z = ' gaze_0_z'
GAZE_1_X = ' gaze_1_x'
GAZE_1_Y = ' gaze_1_y'
GAZE_1_Z = ' gaze_1_z'
EYE_LMK_X_0 = ' eye_lmk_x_0'
EYE_LMK_Y_0 = ' eye_lmk_y_0'
EYE_LMK_3D_0_X = ' eye_lmk_X_0'
EYE_LMK_3D_0_Y = ' eye_lmk_Y_0'
EYE_LMK_3D_0_Z = ' eye_lmk_Z_0'
#2
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

#3
MAX_PROJ_SIZE = 3500
SPHERE_RADIUS = 1000

'''
Several Gaze Projection Methods compared
'''

#2D projection to the edge
def edge_projection(row):
    # frame shape is (width, height)
    x_plane_normal = np.array([1, 0, 0])
    x_plane_point = None
    y_plane_point = None
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
    line_point = np.array([row[EYE_LMK_X_0], row[EYE_LMK_Y_0], 0])
    line_vector = np.array([row[GAZE_0_X], row[GAZE_0_Y], row[GAZE_0_Z]])
    x_t = np.dot((x_plane_point - line_point), x_plane_normal) / np.dot(line_vector, x_plane_normal)
    y_t = np.dot((y_plane_point - line_point), y_plane_normal) / np.dot(line_vector, y_plane_normal)
    if np.abs(x_t) < np.abs(y_t):
        # gaze falls on the left/ri, row[EYE_LMK_Y_0] over edge
        return (line_point + line_vector * y_t).astype(int)[[0,1]]

#3D projection to the camera plane
def screen_projection(row):
    '''
    Uses 3D mm values for eye positions
    '''
        # frame shape is (width, height)
    z_plane_normal = np.array([0, 0, 1])
    z_plane_point = np.array([0, 0, 0])

    line_point = np.array([row[EYE_LMK_3D_0_X], row[EYE_LMK_3D_0_Y], row[EYE_LMK_3D_0_Z]])
    line_vector = np.array([row[GAZE_0_X], row[GAZE_0_Y], row[GAZE_0_Z]])
    z_t = np.dot((z_plane_point - line_point), z_plane_normal) / np.dot(line_vector, z_plane_normal)

    val =(line_point + line_vector *z_t).astype(int)[[0, 1]]
    
    if(val[0]>=MAX_PROJ_SIZE):
        val[0]=MAX_PROJ_SIZE
    if(val[1]>=MAX_PROJ_SIZE):
        val[1]=MAX_PROJ_SIZE
    if(val[0]<=-MAX_PROJ_SIZE):
        val[0]=-MAX_PROJ_SIZE
    if(val[1]<=-MAX_PROJ_SIZE):
        val[1]=-MAX_PROJ_SIZE

    return val

#3D projection on a sphere roughly centered around participant head
def sphere_projection(row):
    # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection#Calculation_using_vectors_in_3D
    c = np.array([0, 0, SPHERE_RADIUS])
    o = np.array([row[EYE_LMK_3D_0_X], row[EYE_LMK_3D_0_Y], row[EYE_LMK_3D_0_Z]])
    u = np.array([row[GAZE_0_X], row[GAZE_0_Y], row[GAZE_0_Z]])
    
    b24ac = (np.dot(u, o - c) ** 2) - (np.dot(o - c, o - c) - SPHERE_RADIUS ** 2)

    if b24ac < 0:
        return np.array(np.zeros(3))

    else:
        first_term = -(np.dot(u, o-c))
        d_plus = first_term + b24ac**0.5
        d_minus = first_term - b24ac**0.5
        
        proj_plus = (d_plus * u) + o
        proj_minus = (d_minus * u) + o

        if proj_minus[2] < proj_plus[2]:
            return proj_minus
        else:
            return proj_plus

#Combining edge and screen projection 
def multi_projection(row):
    r1 = edge_projection(row)
    r2 = screen_projection(row)
    return np.array([r1[0],r1[1],r2[0],r2[1]]).astype(int)

#Just return a subset of the used data without any changes
def no_projection(row):
    return np.array([row[GAZE_ANGLE_X], row[GAZE_ANGLE_Y], row[EYE_LMK_3D_0_X], row[EYE_LMK_3D_0_Y],row[EYE_LMK_3D_0_Z]])

#Function for plotting the clusters
def save_plots(clusters, vectorList, eval=False, x_col=0, y_col=1, point_size=1, n_bins=150):

    predictions = clusters.predict(vectorList)

    
    title = f"{MODEL_TITLE}_{N_CLUSTERS}_clusters"

    plt.scatter(vectorList[:, x_col], vectorList[:, y_col], c=RGB_COLORS[predictions], s=point_size)
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.savefig(f"plots/class_scatter_{title}.png")

    plt.hist2d(vectorList[:, x_col], vectorList[:, y_col], bins=(n_bins, n_bins), norm=LogNorm())
    plt.colorbar()
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.savefig(f"plots/class_hist_{title}.png")

    plt.close()

#Evaluation features used
def vec(row):
    return PROJECTION(row)


def project(df):
    vectorFrame = df.apply(lambda x: PROJECTION(x),axis=1)
    vectorList = np.vstack(vectorFrame.to_numpy())
    return vectorList

################################################ Plots

#Defining 10 Colors in RGB (for plots) and BGR (for opencv visualisation) format

RGB_BLUE,BGR_BLUE=(0,0,1),(255,0,0)
RGB_GREEN,BGR_GREEN=(0,1,0),(0,255,0)
RGB_RED,BGR_RED=(1,0,0),(0,0,255)
RGB_YELLOW,BGR_YELLOW=(1,1,0),(0,255,255)
RGB_PURPLE,BGR_PURPLE = (1,0,1),(255,0,255)
RGB_CYAN,BGR_CYAN = (0,1,1),(255,255,0)
RGB_ORANGE,BGR_ORANGE = (1,0.5,0),(0,140,255)
RGB_GRAY,BGR_GRAY = (0.5,0.5,0.5),(150,150,150)
RGB_BLACK,BGR_BLACK = (0,0,0),(0,0,0)
RGB_WINE,BGR_WINE = (0.5,0,0.25),(75,0,127)

RGB_COLORS = np.array([RGB_BLUE,RGB_GREEN,RGB_RED,RGB_YELLOW,RGB_PURPLE,RGB_CYAN,RGB_ORANGE,RGB_GRAY,RGB_BLACK,RGB_WINE])

BGR_COLORS = [BGR_BLUE,BGR_GREEN,BGR_RED,BGR_YELLOW,BGR_PURPLE,BGR_CYAN,BGR_ORANGE,BGR_GRAY,BGR_BLACK,BGR_WINE]

################################################

MODEL_TITLE = "KMEANS_projection_screen_filtered"
#Projection Function either edge_projection (2D) or screen_projection (3D), mult_projection (both), no_projection (None)
PROJECTION = screen_projection
#number of features
N_FEATURES = 5
#number of CLUSTERS
N_CLUSTERS = 10

PICKLE_TITLE = f"models/{MODEL_TITLE}_clustering.pickle"

USED_COLS = [GAZE_ANGLE_X,GAZE_ANGLE_Y,GAZE_0_X,GAZE_0_Y,GAZE_0_Z,GAZE_1_X,GAZE_1_Y,GAZE_1_Z,EYE_LMK_X_0,EYE_LMK_Y_0,EYE_LMK_3D_0_X,EYE_LMK_3D_0_Y,EYE_LMK_3D_0_Z]

#CSV file returned by OpenFace
CSV_FILE = "data/123OpenFaceDemo.csv"
#Show with OPENFACE overlay or without
DISPLAY_OPENFACE = True


FRAME_RATE = 25


VIDEO_FILE = None

if(DISPLAY_OPENFACE):
    VIDEO_FILE = "./data/123OpenFaceDemo.avi"
else:
    VIDEO_FILE = "./data/123OpenFaceDemo_original.mp4"

