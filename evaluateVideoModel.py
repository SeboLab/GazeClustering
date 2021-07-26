# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pickle
import cv2
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
import config

# %%
with open(config.PICKLE_TITLE, 'rb') as f:
    clusters = pickle.load(f)

# %%
df = pd.read_csv(config.CSV_FILE, usecols=config.USED_COLS)
df_mask = (df!=0).all(1)
df_indices = np.arange(len(df_mask))[df_mask]
df = df.loc[df_mask]
print("reading "+config.CSV_FILE)


# %%
vectorList = config.project(df)


# %%
config.save_plots(clusters, vectorList, eval=True)
predictions = clusters.predict(vectorList)

# show all gaze points in the video
title = f"{config.MODEL_TITLE}\nselect your desired clusters"

fig, current_ax = plt.subplots() 
plt.scatter(vectorList[:, 0], vectorList[:, 1], c=config.RGB_COLORS[predictions], s=1)
plt.title(title)
plt.gca().invert_yaxis()
plt.draw()

def rectangle_callback(eclick, erelease):
    '''
    A callback function that is executed when a rectangle is drawn on the figure of clusters. 
    Select the region you desire to view
    '''

    # get rectangle corners
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    # get all points that fall within the rectangle
    x_mask = (vectorList[:,0] > min(x1, x2)) & (vectorList[:,0] < max(x1, x2))
    y_mask = (vectorList[:,1] > min(y1, y2)) & (vectorList[:,1] < max(y1, y2))

    indices = np.nonzero(x_mask & y_mask)[0]

    display_video(indices)

def display_video(indices):

    # read in the video
    cap = cv2.VideoCapture(config.VIDEO_FILE)
    print(config.VIDEO_FILE)
    # define debug box boundaries
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    BOX_RADIUS = 150
    box_upper_left = (frame_width//2 - BOX_RADIUS, frame_height//2 - BOX_RADIUS)
    box_lower_right = (frame_width//2 + BOX_RADIUS, frame_height//2 + BOX_RADIUS)

    # set up the video writer
    out = cv2.VideoWriter(f'videos/eval_extract_{config.MODEL_TITLE}_{config.N_CLUSTERS}_clusters_pick.avi',cv2.VideoWriter_fourcc('M','J','P','G'), config.FRAME_RATE, (frame_width,frame_height))

    for i in indices:

        # get the index of the frame in the video
        frame_index = df_indices[i]

        # set the video position to the current frame in the list
        accepted = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # if the frame set was successful:
        if accepted:

            # read in the frame, draw the rectangle, and write it to the output file
            _, frame = cap.read()
            cv2.rectangle(frame,box_upper_left,box_lower_right,config.BGR_COLORS[predictions[i]],2)
            out.write(frame)
            cv2.imshow('my image', frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
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

