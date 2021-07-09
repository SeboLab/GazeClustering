#! /usr/bin/python3

import os
import pandas as pd
import datetime

DATA_PATH = "/media/sebo-hri-lab/YaleJiboDat/Jibo Survival 2018-2019 Data"
OUTPUT_PATH = "/media/sebo-hri-lab/DATA/Trimmed_Videos"
CSV_PATH = "ROS_time.csv"

COMMAND_TEMPLATE = "ffmpeg -ss %s -i %s -c copy %s" # start time, input file name, output file name

EXTENSION = "_trim"

ros_sheet = pd.read_csv(CSV_PATH)

ros_info = ros_sheet[["Group", "part2_start_time", "Audio File Offset (Audio - IRS)"]]

video_names = os.listdir(os.path.join(DATA_PATH, "Experiment Intel Realsense Videos - New"))

for i, group in enumerate(ros_info["Group"]):
    group_videos = [video for video in video_names if group in video]

    for input_filename in group_videos:
        output_filename = input_filename.split(".")[0] + EXTENSION + ".mp4"

        input_filepath = "'" + os.path.join(DATA_PATH, "Experiment Intel Realsense Videos - New", input_filename) + "'"
        output_filepath = "'" + os.path.join(OUTPUT_PATH, output_filename) + "'"

        command = COMMAND_TEMPLATE % (ros_info["part2_start_time"][i], input_filepath, output_filepath)
        print(command)
        os.system(command)