#! /usr/bin/python3

import glob
import pandas as pd
import config

MAX_NUMS = 50
CAMERA = "camera3"
USED_COLS = [config.GAZE_ANGLE_X,config.GAZE_ANGLE_Y,config.GAZE_0_X,config.GAZE_0_Y,config.GAZE_0_Z,config.GAZE_1_X,config.GAZE_1_Y,config.GAZE_1_Z,' eye_lmk_x_0',' eye_lmk_y_0',' eye_lmk_X_0',' eye_lmk_Y_0',' eye_lmk_Z_0']
FILE_NAME = "data/shrink_data_"+CAMERA+".csv"

path = r"/media/sebo-hri-lab/DATA/OpenFace/" # use your path
all_files = glob.glob(path + "/*.csv")

li = []
i = 0
df = pd.DataFrame(columns=USED_COLS)
df.to_csv(FILE_NAME)
for filename in all_files:
    #print(all_files)
    if(CAMERA in filename):
        print('reading '+filename)
        if i>MAX_NUMS:
            break
        df = pd.read_csv(filename, index_col=None, usecols=USED_COLS)
        print(df)
        ndf = df.loc[(df!=0).all(1)]
        ndf.to_csv(FILE_NAME, columns=USED_COLS, mode='a', header=None)
        i+=1

print('finished reading lines')

