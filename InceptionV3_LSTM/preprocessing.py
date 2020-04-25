import os
from os.path import join
import cv2
import numpy as np
import random
from keras.preprocessing.image import ImageDataGenerator

def video_generator(real_path, fake_path, img_size = 299, 
                    frame = 40, batch_size = 10):
    # real_path/fake_path: path to the folder of real/fake videos
    # each video is a folder containing ordered images labelled by
    # a number starting from 1
    
    # frame, img_size: the size of video - an array of 
    # img_size X img_size X 3 image with length frame, note that the
    # frame here must align with how many images are contained in each
    # video folder
    
    # batch_size: how many records for each batch, preferably even
    
    # start_idx: a tuple that specifies the labels of the first videos
    # for training among the real videos and fake ones
    
    # placeholder for inputs and targets
    X = np.zeros((batch_size, frame, img_size, img_size, 3))
    y = np.zeros(batch_size, dtype=int)
    
    # we use this object to preprocess video
    train_datagen = ImageDataGenerator(samplewise_center=True)
    fitted = False
    
    real_lst = []
    fake_lst = []
    
    for folder, lst, label in [(real_path,real_lst,False), (fake_path,fake_lst,True)]:
        for video in os.listdir(folder):
            src = join(folder,video)
            if frame <= len(os.listdir(src)):
                lst.append((src,label))
    
    lst = real_lst[:min(len(real_lst),len(fake_lst))]
    lst.extend(fake_lst[:min(len(real_lst),len(fake_lst))])
    random.shuffle(lst)
    total = 0
    
    while True:       
        counter = 0
        
        # placeholder for each video
        sequence = np.zeros((frame, img_size, img_size, 3))
        
        for i in range(total,total+batch_size):
            if i >= len(lst):
                continue
            src, label = lst[i]
            counter_frame = 0 

            for img in os.listdir(src):
                img_src = join(src, img)
                img_array = cv2.imread(img_src)
                img_array = cv2.resize(img_array, (img_size, img_size), 
                                      interpolation = cv2.INTER_CUBIC)

                sequence[counter_frame] = img_array
                counter_frame += 1
                if counter_frame >= frame:
                    break

            if not fitted:
                train_datagen.fit(sequence)
                fitted = True

            X[counter] = train_datagen.standardize(sequence)
            y[counter] = label
            counter += 1
            i += 1
            
        total += batch_size

        
        # use yield here instead of return so the function will only
        # return values unless being called again by the model
        yield (X[:counter], y[:counter])
        

def video_generator_test(real_path, fake_path, img_size = 299, 
                    frame = 40, batch_size = 10):

    X = np.zeros((batch_size, frame, img_size, img_size, 3))
    y = np.zeros(batch_size, dtype=int)
    
    # we use this object to preprocess video
    train_datagen = ImageDataGenerator(samplewise_center=True)
    fitted = False
    
    lst = []
    
    for folder, label in [(real_path,False), (fake_path,True)]:
        for video in os.listdir(folder):
            src = join(folder,video)
            if frame <= len(os.listdir(src)):
                lst.append((src,label))
     
    total = 0
    
    while True:       
        counter = 0
        
        # placeholder for each video
        sequence = np.zeros((frame, img_size, img_size, 3))
        
        for i in range(total,total+batch_size):
            if i >= len(lst):
                continue
            src, label = lst[i]
            counter_frame = 0 

            for img in os.listdir(src):
                img_src = join(src, img)
                img_array = cv2.imread(img_src)
                img_array = cv2.resize(img_array, (img_size, img_size), 
                                      interpolation = cv2.INTER_CUBIC)

                sequence[counter_frame] = img_array
                counter_frame += 1
                if counter_frame >= frame:
                    break

            if not fitted:
                train_datagen.fit(sequence)
                fitted = True

            X[counter] = train_datagen.standardize(sequence)
            y[counter] = label
            counter += 1
            i += 1
            
        total += batch_size

        
        # use yield here instead of return so the function will only
        # return values unless being called again by the model
        yield (X[:counter], y[:counter])
    