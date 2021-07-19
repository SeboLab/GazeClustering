# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pickle
import config

print(config.MODEL_TITLE)


# %%
if(config.GROUP_NAME!=None):
    df = pd.read_csv(f"/media/sebo-hri-lab/DATA/OpenFace/group_{config.GROUP_NAME}_{config.CAMERA}_trim.csv")
else:
    df = pd.read_csv(config.FILE_NAME)


# %%
vectorList = np.zeros((len(df),config.N_FEATURES))
df['occurrence'] = df.apply(lambda x: config.PROJECTION(x),axis=1)
vectorList = np.vstack(df['occurrence'].to_numpy())
print(f"{len(vectorList)} training frames")

# %%
clusters = KMeans(n_clusters=config.N_CLUSTERS).fit(vectorList)

predictions = clusters.predict(vectorList)

colors = config.RGB_COLORS
            
plt.scatter(vectorList[:, 0], vectorList[:, 1], c=colors[predictions], s=1)
plt.title(f"{config.N_CLUSTERS} Clusters")
plt.gca().invert_yaxis()
plt.savefig('plots/class_scatter_sphere.png')
plt.show()

# %%

with open(config.PICKLE_TITLE, 'wb') as handle:
    pickle.dump(clusters, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("finished writing model")


