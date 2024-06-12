# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import os
import cv2
import numpy as np
import pandas as pd

model = YOLO('best.pt')
video_path = "c:/Users/Sarah saeed/Desktop/Gurdian angel_videoToFrames/kids_nersery_videos/playing1.mp4"
cap = cv2.VideoCapture(video_path)
names = model.names
output_folder_cropped_bodies="C:/Users/Sarah saeed/Desktop/gurndian angel vscode env/cropped_for_bodies"
# Get the video's frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
output_path = 'output_video2.mp4'  # Path to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
output_video = cv2.VideoWriter(output_path, fourcc, 3.0, (frame_width, frame_height))

i=0
# Check if the video file was successfully opened
if not cap.isOpened():
    print("Error opening video file")
    exit()
# Iterate over each frame in the video
while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break
    # Convert the frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = model(frame_rgb)
    result = output[0]

    for (row,c) in zip(result.boxes.xyxy,result.boxes.cls):
        x, y, w, h = int(row[0]), int(row[1]), int(row[2]), int(row[3])

        frame_with_rectangle=frame.copy()
        cv2.rectangle(frame_with_rectangle, (x, y), (w, h), (0, 255, 0), 2) 
        cv2.putText(frame_with_rectangle,names[int(c)], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) 

        cropped_frame=frame[y:h,x:w]
        cv2.imwrite(os.path.join(output_folder_cropped_bodies,f"frame{i}.jpg"),cropped_frame)
    # Display the frame with bounding boxes
    #output_video.write(frame)
    cv2.imshow('action Detection', frame_with_rectangle)
    i=i+1
    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release the video file and close any open windows
cap.release()
cv2.destroyAllWindows()