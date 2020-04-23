import os
import face_recognition
from os.path import join
from cv2 import VideoCapture, imread, imwrite, imshow, rectangle, cvtColor, COLOR_BGR2RGB

def extractFrames(frames):
    for folder in ['Real', 'Fake']:   
        for video in sorted(os.listdir(folder)):
            src = join(folder, video)
            dst = join(folder, video[0])
            
            #Placing the frames in a new folder for each video
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
    for img in os.listdir(path):
        pixels = imread(join(path, img))
        test = face_recognition.face_locations(pixels)

        for (top, right, bottom, left) in test:
            rectangle(pixels, (left, top), (right, bottom), (0, 0, 255), 2)

        extracted_img = pixels[top:bottom, left:right]
        imwrite(join(path, img), extracted_img)

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