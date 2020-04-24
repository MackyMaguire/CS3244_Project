
import face_recognition
import random
import os
from os.path import join
from shutil import copy
from cv2 import VideoCapture, imread, imwrite, imshow, rectangle, cvtColor, COLOR_BGR2RGB
from functools import cmp_to_key

def processVideo(frames, src, realfake, counter):
    #Split train:test 3:1
    rand = random.random()
    traintest = 'Train'
    if rand > 0.75:
        traintest = 'Test'

    #Extract face frames from each video
    #Store frames in an individual folder for each video
    frames_dst = join(traintest, realfake, 'Frames', str(counter))
    os.mkdir(frames_dst)
    extractFrames(frames, src, frames_dst)
    
    #Rename each video based on counter, which is in ascending numerical order
    #Copy to new destination
    video_dst = join(traintest, realfake, 'Videos', '%d.mp4' % counter)
    copy(src, video_dst)

def extractFrames(frames, src, dst):
    reader = VideoCapture(src)
            
    frame_num = 1

    while reader.isOpened():
        running, frame = reader.read()
        if not running:
            break
        if frame_num > frames:
            break

        #Extract face, with 25 pixels margin
        loc = face_recognition.face_locations(frame)

        if (len(loc) == 0):
            face = frame
        else:
            loc = sorted(loc, key=cmp_to_key(lambda x, y: (y[2] - y[0]) * (y[1] - y[3]) - (x[2] - x[0]) * (x[1] - x[3])))
            face = frame[loc[0][0] - 25 : loc[0][2] + 25, loc[0][3] - 25 : loc[0][1] + 25]
        
        imwrite(join(dst, '%d.jpg' % frame_num), face)
        frame_num += 1
        
    reader.release()

def main(): 
    frames = int(input("Enter desired number of frames to extract: "))

    #Refer to README.md for outputted folder structure
    for x in ['Train', 'Test']:
        os.mkdir(x)
        for y in ['Real', 'Fake']:
            os.mkdir('%s/%s' % (x, y))
            for z in ['Frames', 'Videos']:
                os.mkdir('%s/%s/%s' % (x, y, z))

    REAL_PATHS = ['data/original_sequences/actors/c23/videos',
                  'data/original_sequences/youtube/c23/videos']

    FAKE_PATHS = ['data/manipulated_sequences/DeepFakeDetection/c23/videos',
                  'data/manipulated_sequences/Deepfakes/c23/videos',
                  'data/manipulated_sequences/Face2Face/c23/videos',
                  'data/manipulated_sequences/FaceSwap/c23/videos',
                  'data/manipulated_sequences/NeuralTextures/c23/videos',]

    fake_counter = 1
    real_counter = 1
            
    for path in REAL_PATHS:
        for video in os.listdir(path):
            src = join(path, video)
            processVideo(frames, src, 'Real', real_counter)
            real_counter += 1
        
    for path in FAKE_PATHS: 
        for video in os.listdir(path):
            src = join(path, video)
            processVideo(frames, src, 'Fake', fake_counter)
            fake_counter += 1

if __name__ == "__main__":
    main()
