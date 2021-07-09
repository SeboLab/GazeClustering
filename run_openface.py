#! /usr/bin/python3

import os
import pandas as pd

DATA_PATH = "/media/sebo-hri-lab/DATA/Trimmed_Videos"
OUTPUT_PATH = "/media/sebo-hri-lab/DATA/OpenFace"

COMMAND_TEMPLATE = "~/Documents/OpenFace-OpenFace_2.2.0/build/bin/FeatureExtraction %s-out_dir %s" # input filenames, output folder

video_names = os.listdir(DATA_PATH)

cmd_string = ""
last_complete_video = "group_BI_camera1_trim.avi"
for name in video_names:
    if(name>=last_complete_video):
        print("Extraction of video "+name)
        filepath = os.path.join(DATA_PATH, name)
        string = "-f " + filepath + " "
        cmd_string = cmd_string + string

command = COMMAND_TEMPLATE % (cmd_string, OUTPUT_PATH)

os.system(command)