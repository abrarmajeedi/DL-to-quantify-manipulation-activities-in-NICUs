import cv2
import os
import pandas as pd 

videos = []
num_frames = []

for vid in os.listdir("tray"):
    if vid.endswith(".mp4"):
        videos.append(vid.split(".mp4")[0])
        cap = cv2.VideoCapture(os.path.join("tray",vid))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames.append(length)
        print(vid,length)

df = pd.DataFrame({"Video": videos,
                    "Frames": num_frames})
df.to_csv("tray.csv", header =  None, index = None)

