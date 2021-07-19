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
import config


# %%
with open(config.PICKLE_TITLE, 'rb') as f:
    clusters = pickle.load(f)


# %%
df = pd.read_csv(config.CSV_FILE, usecols=config.USED_COLS)
df = df.loc[(df!=0).all(1)]

print("reading "+config.CSV_FILE)


# %%
df['occurence'] = df.apply(lambda x: config.projection(x),axis=1)


# %%
vectorList = np.zeros((len(df),config.N_FEATURES))
vectorList = np.vstack(df['occurence'].to_numpy())


# %%
#Color titles:
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


# %%
predictions = clusters.predict(vectorList)

colors = np.array([RGB_BLUE,RGB_GREEN,RGB_RED,RGB_YELLOW,RGB_PURPLE,RGB_CYAN,RGB_ORANGE,RGB_GRAY,RGB_BLACK,RGB_WINE])
            
plt.scatter(vectorList[:, 0], vectorList[:, 1], c=colors[predictions], s=1)
plt.title(f"{config.N_CLUSTERS} Clusters")
plt.gca().invert_yaxis()
plt.savefig('class_scatter.png')
plt.show()
plt.hist2d(vectorList[:, 0], vectorList[:, 1], bins=(150, 150), norm=LogNorm())
plt.title(f"{config.N_CLUSTERS} Clusters")
plt.gca().invert_yaxis()
plt.savefig('class_hist.png')
plt.show()


# %%
cap = cv2.VideoCapture(config.VIDEO_FILE)
i = 0
colors = [BGR_BLUE,BGR_GREEN,BGR_RED,BGR_YELLOW,BGR_PURPLE,BGR_CYAN,BGR_ORANGE,BGR_GRAY,BGR_BLACK,BGR_WINE]
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
BOX_RADIUS = 30
out = cv2.VideoWriter('classificationExtract.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))
while i<len(df):
    _, frame = cap.read()
    row = df.iloc[i]
    if(row[config.GAZE_ANGLE_X] or row[config.GAZE_ANGLE_Y]):
        gaze_target = config.screen_projection(row)
        vec = [config.vec(row)]
        clus = clusters.predict(vec)[0]
        box_center = [gaze_target[0] + (frame_width//2), gaze_target[1] + (frame_height//2)]
        cv2.rectangle(frame,(100,100),(200,200),colors[clus],2)
        out.write(frame)
    cv2.imshow('my image', frame)
    i+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()


# %%



# %%



