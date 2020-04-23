
'''
Folder Strucuture:
/root
    ExtractFrames.py
    /data
        /train
            /frames
                /Real
                    /055_frames
                        1.jpg
                        ...
                /Fake
                    /033_097_frames
                        1.jpg
                        ...
            /videos
                /Real
                    055.mp4
                /Fake
                    033_097.mp4
        /test
            /frames
                /Real
                    /055_frames
                        1.jpg
                        ...
                /Fake
                    /033_097_frames
                        1.jpg
                        ...
            /videos
                /Real
                    055.mp4
                /Fake
                    033_097.mp4
'''

import os

import face_recognition
from os.path import join
from cv2 import VideoCapture, imread, imwrite, imshow, rectangle, cvtColor, COLOR_BGR2RGB

BASE_VIDEO_FOLDER = 'data/train/videos'
BASE_IMAGES_FOLDER = 'data/train/frames'

def extractFrames(frames):
    print('Extracting frames...')
    for folder in ['Real', 'Fake']:
        for video in os.listdir(join(BASE_VIDEO_FOLDER, folder)):
            src = join(join(BASE_VIDEO_FOLDER, folder), video)
            dst = join(join(BASE_IMAGES_FOLDER, folder), os.path.splitext(video)[0] + '_frames')
            
            os.mkdir(dst)
            
            reader = VideoCapture(src)
            
            frame_num = 1
            
            while reader.isOpened():
                running, frame = reader.read()
                if not running:
                    break
                if frame_num > frames:
                    break
                imwrite(join(dst, '%d.jpg' % frame_num), frame)
                frame_num += 1

            reader.release()
            extractFace(dst)

# Replace all images inside a path with images of the extracted face from the frame
def extractFace(path):
    print('Extracting faces from frames for %s...' % path)
    for img in os.listdir(path):
        pixels = imread(join(path, img))
        loc = face_recognition.face_locations(pixels)

        for (top, right, bottom, left) in loc:
            rectangle(pixels, (left - 25, top - 25), (right + 25, bottom + 25), (0, 0, 255), 2)
            # rectangle(pixels, (left, top), (right, bottom), (0, 0, 255), 2)

        extracted_img = pixels[top - 25 : bottom + 25, left - 25 : right + 25]
        imwrite(join(path, img), extracted_img)

def renameVideos():
    print('Renaming videos...')
    for folder in [join(BASE_VIDEO_FOLDER, 'Real'), join(BASE_VIDEO_FOLDER,'Fake')]:
        counter = 1
        for video in os.listdir(folder):
            src = join(folder, video)
            dst = join(folder, '%d.mp4' % counter)
            os.rename(src, dst)
            counter += 1

if __name__ == "__main__":
    frames = int(input("Enter desired number of frames to extract: "))
    # renameVideos()
    extractFrames(frames)
