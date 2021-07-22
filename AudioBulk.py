# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Read data and create Dataframe

# %%
import os
import pandas as pd
import numpy as np
import glob
import pandas as pd
import config


INPUT_PATH = r"/media/sebo-hri-lab/DATA/OpenSmile/" # use your path
THRESH = 0.35
OUTPUT_PATH = r"/media-sebo-hri-lab/DATA/WhoIsSpeaking/"

RANGE = 10


all_files = glob.glob(INPUT_PATH + "/*.csv")
audio_chunks = []
i = 0
prefixes = []
while i < len(all_files):
    prefix = all_files[i][35:37]
    prefixes.append(prefix)
    csv_files = [ file for file in os.listdir(INPUT_PATH) if ((".csv" in file) and prefix in file)]
    audio_chunks.append(csv_files)
    i+=4

# Get audio files
#audio_files = [ file for file in os.listdir(INPUT_PATH) if ((".wav" in file) & (file not in csv_files)) ]


# %%
# Loudness of each mic

df = pd.DataFrame()

for predex, csv_files in enumerate(audio_chunks):
    filename = prefixes[predex]+'_who_is_talking.csv'
    print("preparing "+filename)
    for i, file in enumerate(csv_files):
    
        if "_conf" in file:
            ext = "conf"
        else:
            ext = "pid" + file[file.find("pid") + 3]
        
        temp_df = pd.read_csv(os.path.join(INPUT_PATH, file))
    
        # start and end time stamps -- files are synced
        if i == 0:
            df['start'] = temp_df['start']
            df['end'] = temp_df['end']
    
        df[f"Loudness_{ext}"] = temp_df['Loudness_sma3']
    ordered_df = df[['Loudness_pid1', 'Loudness_pid2', 'Loudness_pid3', 'Loudness_conf']]
    for col in ["Loudness_pid1", "Loudness_pid2", "Loudness_pid3", "Loudness_conf"]:
        df[col + "_norm"] = df[col] * RANGE / max(1, np.abs(df[col].max()))

    for id in ["pid1", "pid2", "pid3", "conf"]:
        norm = "Loudness_%s_norm" % (id)
        df[id + "_speaking"] = df[norm] > THRESH
    df["jibo_speaking"] = df.apply(lambda x: x["conf_speaking"] and not (x["pid1_speaking"] or x["pid2_speaking"] or x["pid3_speaking"]) , axis=1)
    df.to_csv('WhoIsSpeaking/'+filename)
