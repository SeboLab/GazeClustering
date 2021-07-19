#! /usr/bin/python3

import os
import sys
import pandas as pd
import datetime

VIDEO_PATH = "/media/sebo-hri-lab/DATA/Trimmed_Videos"
OPENFACE_PATH = '/media/sebo-hri-lab/DATA/OpenFace'
AUDIO_PATH = '/media/sebo-hri-lab/DATA/Trimmed_Audio'

VIDEO_OUT_PATH = "/media/sebo-hri-lab/DATA/Spliced_Videos"
OPENFACE_OUT_PATH = "/media/sebo-hri-lab/DATA/Spliced_Openface"
AUDIO_OUT_PATH = "/media/sebo-hri-lab/DATA/Spliced_Audio"

EXTENSION = "_spliced"

COMMAND_TEMPLATE = "ffmpeg -ss %s -i %s -to %s -c copy %s" # start time, input file name, end time,  output file name

# converts timestamp of format hh:mm:ss or mm:ss to seconds
def to_sec(a):
    arr = a.split(":")
    return sum([ float(num) * (60 ** (len(arr) - i - 1 )) for i, num in enumerate(arr) ])

if __name__ == "__main__":
    # input is start_timestamp end_timestamp video_name
    [start, end, video_name] = sys.argv[1:4]
    
    # write new spliced openface csv files
    csv_names  = [ csv for csv in os.listdir(OPENFACE_PATH) if video_name in csv  ]
    for csv in csv_names:
        df = pd.read_csv(csv)
        
        csv_filename = csv.split(".")[0] # remove extension

        df_spliced = (df.loc[ (start_timestamp <= df[' timestamp']) & (df[' timestamp'] <= end_timestamp) ])
        df_spliced.to_csv(os.path.join(OPENFACE_OUT_PATH, csv_filename + EXTENSION + ".csv"))


    # write new spliced audio csv files
    csv_names  = [ csv for csv in os.listdir(AUDIO_PATH) if video_name in csv  ]
    for csv in csv_names:
        df = pd.read_csv(csv)

        csv_filename = csv.split(".")[0] # remove extension
        
        # new columns with timestamps in seconds -- Days ignored
        df['start_sec'] = df.apply(lambda x: to_sec(x['start'].split(" ")[2]), axis=1)        
        df['end_sec'] = df.apply(lambda x: to_sec(x['end'].split(" ")[2]), axis=1)

        df_spliced = (df.loc[ (start_timestamp <= df['start_sec']) & (df['end_sec'] <= end_timestamp) ])
        df_spliced.to_csv(os.path.join(AUDIO_OUT_PATH, csv_filename + EXTENSION + ".csv"))

    # write new spliced video files 
    start_timestamp = to_sec(start)
    end_timestamp = to_sec(end)
    
    video_names = [ video for video in os.listdir(VIDEO_PATH) if video_name in video ]
    
    for input_filename in video_names:
        output_filename = input_filename.split(".")[0] + EXTENSION + ".mp4"
        
        input_filepath = "'" + os.path.join(VIDEO_PATH, input_filename) + "'"
        output_filepath = "'" + os.path.join(OUTPUT_PATH, output_filename) + "'"
        command = COMMAND_TEMPLATE % (start_timestamp, input_filename, end_timestamp, output_filename)
        os.system(command)
