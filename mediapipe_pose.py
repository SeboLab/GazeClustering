import cv2
import numpy as np
import pandas as pd
import config

import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 

cap = cv2.VideoCapture(config.VIDEO_FILE)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
BOX_RADIUS = 30
out = cv2.VideoWriter('poseExtract.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))

with mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5) as pose:

    i = 0

    while True:
        _, frame = cap.read()
        
        # Convert the BGR image to RGB and process it with MediaPipe Pose.
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Print nose landmark.
        image_hight, image_width, _ = frame.shape
        if not results.pose_landmarks:
            continue
        print(
            f'Nose coordinates: ('
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_hight})'
        )

        # Draw pose landmarks.
        print(f'Pose landmarks of frame {i}:')
        annotated_image = frame.copy()
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS)

        cv2.imshow("frame", annotated_image)

        i = i + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
out.release()
cv2.destroyAllWindows()