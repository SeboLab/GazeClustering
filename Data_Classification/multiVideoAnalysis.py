# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import rand

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

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



# %%


# %%
def edge_projection(row, frame_shape):
    # frame shape is (width, height)

    x_plane_normal = np.array([1, 0, 0])

    if row[GAZE_0_X] > 0:
        # participant is looking right
        x_plane_point = np.array([frame_shape[0], 0, 0])
    else:
        # participant is looking left
        x_plane_point = np.array([0, 0, 0])

    y_plane_normal = np.array([0, 1, 0])
    
    if row[GAZE_0_Y] > 0:
        # participant is looking down
        y_plane_point = np.array([0, frame_shape[1], 0])
    else:
        # participant is looking up
        y_plane_point = np.array([0, 0, 0])

    line_point = np.array([row[' eye_lmk_x_0'], row[' eye_lmk_y_0'], 0])

    line_vector = np.array([row[GAZE_0_X], row[GAZE_0_Y], row[GAZE_0_Z]])

    x_t = np.dot((x_plane_point - line_point), x_plane_normal) / np.dot(line_vector, x_plane_normal)

    y_t = np.dot((y_plane_point - line_point), y_plane_normal) / np.dot(line_vector, y_plane_normal)

    if np.abs(x_t) < np.abs(y_t):
        # gaze falls on the left/right edge
        return (line_point + line_vector * x_t).astype(int)
    else:
        # gaze falls on the upper/lower edge
        return (line_point + line_vector * y_t).astype(int)


# %%
import glob

MAX_NUMS = 0
CAMERA = "camera2"

path = r"/media/sebo-hri-lab/DATA/OpenFace/" # use your path
all_files = glob.glob(path + "/*.csv")

li = []
i = 0

for filename in all_files:
    if(CAMERA in filename):
        print('reading '+filename)
        if i>MAX_NUMS:
            break
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
        i+=1

frame = pd.concat(li, axis=0, ignore_index=True)


# %%
vectorList = np.zeros((len(frame),2))
for index, row in frame.iterrows():
    if(row[GAZE_ANGLE_X] or row[GAZE_ANGLE_Y]):
        res = edge_projection(row,(FRAME_WIDTH,FRAME_HEIGHT))
        vectorList[index] = res[0],res[1]#[row[GAZE_0_X], row[GAZE_0_Y], row[GAZE_0_Z], row[GAZE_1_X], row[GAZE_1_Y], row[GAZE_1_Z]]


# %%
kmeans = KMeans(n_clusters=6,random_state=0).fit(vectorList)
predictions = kmeans.predict(vectorList)

colors = np.array([(0,0,1),(0,1,0),(1,0,1),(0,1,1),(1,0,0),(0,0,0)])

plt.scatter(vectorList[:, 0], vectorList[:, 1], c=colors[predictions], s=1)
plt.title("Incorrect Number of Blobs")
plt.gca().invert_yaxis()
plt.show()
plt.savefig('class.png')


# %%



