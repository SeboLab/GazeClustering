# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle
import config

print(config.MODEL_TITLE)


# %%
if(config.GROUP_NAME!=None):
    df = pd.read_csv(f"/media/sebo-hri-lab/DATA/OpenFace/group_{config.GROUP_NAME}_{config.CAMERA}_trim.csv")
else:
    df = pd.read_csv(config.FILE_NAME)


# %%
vectorList = config.project(df)
print(f"{len(vectorList)} training frames")

# %%
clusters = KMeans(n_clusters=config.N_CLUSTERS).fit(vectorList)
#clusters = Birch(threshold=0.5, branching_factor=3,n_clusters=config.N_CLUSTERS).fit(vectorList)

#clusters = OPTICS().fit(vectorList)

config.save_plots(clusters, vectorList)
            


# %%

with open(config.PICKLE_TITLE, 'wb') as handle:
    pickle.dump(clusters, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("finished writing model")


