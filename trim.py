#! /usr/bin/python3

import os
import pandas as pd
import datetime

DATA_PATH = "/media/sebo-hri-lab/YaleJiboDat/Jibo Survival 2018-2019 Data"
OUTPUT_PATH = "/media/sebo-hri-lab/DATA/Trimmed_Videos"
CSV_PATH = "/home/sebo-hri-lab/Desktop/ROS_time.csv"

ros_sheet = pd.read_csv(CSV_PATH)

video_data = ros_sheet[["Group", "part2_start_time", "Audio File Offset (Audio - IRS)"]]


print(video_data["part2_start_time"].dtype)