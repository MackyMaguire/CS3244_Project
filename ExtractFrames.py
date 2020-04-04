'''
pip install opencv-python
'''

import os
from os.path import join
import cv2

def extractFrames(frames):
    for folder in ['Real', 'Fake']:       
        for video in sorted(os.listdir(folder)):
            src = join(folder, video)
            dst = join(folder, video[0])
            
            #Placing the frames in a new folder for each video
            os.mkdir(dst)
            
            reader = cv2.VideoCapture(src)
            
            frame_num = 1
            
            while reader.isOpened():
                running, frame = reader.read()
                if not running:
                    break
                if frame_num > frames:
                    break
                cv2.imwrite(join(dst, '%d.jpg' % frame_num), frame)
                frame_num += 1

            reader.release()

def renameVideos():
    for folder in ['Real', 'Fake']:       
        counter = 1
        for video in sorted(os.listdir(folder)):
            src = join(folder, video)
            dst = join(folder, '%d.mp4' % counter)
            os.rename(src, dst)
            counter += 1

if __name__ == "__main__":
    frames = int(input("Enter desired number of frames to extract: "))
    renameVideos()
    extractFrames(frames)