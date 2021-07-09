#! /usr/bin/python3

import os
import pandas as pd
from datetime import datetime as dt
import datetime

DATA_PATH = "/media/sebo-hri-lab/YaleJiboDat/Jibo Survival 2018-2019 Data"
OUTPUT_PATH = "/media/sebo-hri-lab/DATA/Trimmed_Audio"
CSV_PATH = "ROS_time.csv"

COMMAND_TEMPLATE = "ffmpeg -ss %s -i %s -c copy %s" # start time, input file name, output file name

EXTENSION = "_trim"

ros_sheet = pd.read_csv(CSV_PATH)

ros_info = ros_sheet[["Group", "part2_start_time", "Audio File Offset (Audio - IRS)"]]


def getStartTime(x,offset):
    print(x,offset)
    x = dt.strptime(x,"%M:%S.%f")
    zero = dt.strptime('0',"%M")
    x = x-zero
    #offset = datetime.strptime(str(offset),"%S.%f")
    time_change = datetime.timedelta(seconds=offset)

    print(x,time_change)
    res = x+time_change
    return str(res)


print(ros_info["part2_start_time"])
ros_info["part2_audio_start_time"] = ros_info.apply(lambda row : getStartTime(row["part2_start_time"],row["Audio File Offset (Audio - IRS)"]), axis = 1)
print(ros_info)


#ros_info['part2_audio_start_time'] = audio_start_time

audio_files = os.listdir(os.path.join(DATA_PATH, "Experiment Audio Files"))

for i, group in enumerate(ros_info["Group"]):
    group_audio = [video for video in audio_files if group in video]

    for input_filename in group_audio:
        output_filename = input_filename.split(".")[0] + EXTENSION + ".wav"

        input_filepath = "'" + os.path.join(DATA_PATH, "Experiment Audio Files", input_filename) + "'"
        output_filepath = "'" + os.path.join(OUTPUT_PATH, output_filename) + "'"

        command = COMMAND_TEMPLATE % (ros_info["part2_audio_start_time"][i], input_filepath, output_filepath)
        print(command)
        #os.system(command)
    