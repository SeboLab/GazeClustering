# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pickle
import cv2
import pandas as pd 
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import RectangleSelector
import config

# %%
with open(config.PICKLE_TITLE, 'rb') as f:
    clusters = pickle.load(f)


# %%
df = pd.read_csv(config.CSV_FILE, usecols=config.USED_COLS)
df = df.loc[(df!=0).all(1)]
print("reading "+config.CSV_FILE)


# %%
vectorList = config.project(df)


# %%
config.save_plots(clusters, vectorList, eval=True)

predictions = clusters.predict(vectorList)

# show all gaze points in the video
title = f"select your desired clusters"

fig, current_ax = plt.subplots() 
plt.scatter(vectorList[:, 0], vectorList[:, 1], c=config.RGB_COLORS[predictions], s=1)
plt.title(title)
plt.gca().invert_yaxis()

def rectangle_callback(eclick, erelease):
    '''
    A callback function that is executed when a rectangle is drawn on the figure of clusters.
    '''

    # get rectangle corners
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    # get all points that fall within the rectangle
    x_mask = (vectorList[:,0] > min(x1, x2)) & (vectorList[:,0] < max(x1, x2))
    y_mask = (vectorList[:,1] > min(y1, y2)) & (vectorList[:,1] < max(y1, y2))

    indices = np.nonzero(x_mask & y_mask)[0]

    # sort points by color
    indices = indices[np.argsort(predictions[indices])]

    # read in the video
    cap = cv2.VideoCapture(config.VIDEO_FILE)

    # define debug box boundaries
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    BOX_RADIUS = 150
    box_upper_left = (frame_width//2 - BOX_RADIUS, frame_height//2 - BOX_RADIUS)
    box_lower_right = (frame_width//2 + BOX_RADIUS, frame_height//2 + BOX_RADIUS)

    # set up the video writer
    out = cv2.VideoWriter(f'videos/eval_extract_{config.MODEL_TITLE}_{config.N_CLUSTERS}_clusters_pick.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))

    for i in indices:

        # set the video position to the current frame in the list
        accepted = cap.set(cv2.CAP_PROP_POS_FRAMES, i)

        # if the frame set was successful:
        if accepted:

            # read in the frame, draw the rectangle, and write it to the output file
            _, frame = cap.read()
            cv2.rectangle(frame,box_upper_left,box_lower_right,config.BGR_COLORS[predictions[i]],2)
            out.write(frame)
            cv2.imshow('my image', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def toggle_color(event):
    print(event.key)

toggle_color.RS = RectangleSelector(current_ax,
                                    rectangle_callback, 
                                    drawtype='box',
                                    useblit=True,
                                    button=[1, 3],
                                    minspanx=5,
                                    minspany=5,
                                    spancoords='pixels',
                                    interactive=True)
plt.connect('key_press_event', toggle_color)
plt.show()


# %%
# cap = cv2.VideoCapture(config.VIDEO_FILE)
# i = 0

# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# BOX_RADIUS = 30
# out = cv2.VideoWriter(f'videos/eval_extract_{config.MODEL_TITLE}_{config.N_CLUSTERS}_clusters.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))
# while i<len(df):
#     _, frame = cap.read()
#     row = df.iloc[i]
#     if(row[config.GAZE_ANGLE_X] or row[config.GAZE_ANGLE_Y]):
#         gaze_target = config.screen_projection(row)
#         vec = [vectorList[i]]
#         clus = clusters.predict(vec)[0]
#         #box_center = [gaze_target[0] + (frame_width//2), gaze_target[1] + (frame_height//2)]
#         cv2.rectangle(frame,(100,100),(200,200),config.BGR_COLORS[clus],2)
#         out.write(frame)
#     cv2.imshow('my image', frame)
#     i+=1
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# out.release()
# cv2.destroyAllWindows()


# %%
if(config.GROUP_NAME!=None):
    df = pd.read_csv(f"/media/sebo-hri-lab/DATA/OpenFace/group_{config.GROUP_NAME}_{config.CAMERA}_trim.csv")
else:
    df = pd.read_csv(config.FILE_NAME)