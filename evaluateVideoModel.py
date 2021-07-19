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
df['occurence'] = df.apply(lambda x: config.PROJECTION(x),axis=1)


# %%
vectorList = np.zeros((len(df),config.N_FEATURES))
vectorList = np.vstack(df['occurence'].to_numpy())


# %%
predictions = clusters.predict(vectorList)

colors = config.RGB_COLORS
            
plt.scatter(vectorList[:, 0], vectorList[:, 1], c=colors[predictions], s=1)
plt.title(f"{config.N_CLUSTERS} Clusters")
plt.gca().invert_yaxis()
plt.savefig('plots/eval_class_scatter.png')
plt.show()
plt.hist2d(vectorList[:, 0], vectorList[:, 1], bins=(150, 150), norm=LogNorm())
plt.title(f"{config.N_CLUSTERS} Clusters")
plt.gca().invert_yaxis()
plt.savefig('plots/eval_class_hist.png')
plt.show()


# %%
cap = cv2.VideoCapture(config.VIDEO_FILE)
i = 0
colors = config.BGR_COLORS
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



