import os
from os.path import join
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def load_data(real_path, fake_path, num_sample = (100,100), 
              img_size = 299, frame = 40):
    # output X_train, X_test, y_train, y_test
    X = np.zeros(sum(num_sample), frame, img_size, img_size, 3)
    y = np.concatenate((np.zeros(num_sample[0], dtype=int),
                        np.ones(num_sample[1], dtype=int))) 
    
    # creat an image generator to preprocess the data
    train_datagen = ImageDataGenerator(samplewise_center=True)
    
    counter = 0
    # placeholder for a video sample
    sequence = np.zeros(frame, img_size, img_size, 3)
    
    for folder, num in [(real_path, num_sample[0]), 
                        (fake_path, num_sample[1])]:
        
        for i in range(1, num+1):
            src = join(folder, str(i)) 
            counter_frame = 0
            
            for img in os.listdir(src):
                img_src = join(src, img)
                img_array = cv2.imread(img_src)
                img_array = cv2.resize(img_array, (img_size, img_size), 
                                     interpolation = cv2.INTER_CUBIC)
                
                sequence[counter_frame] = img_array
                counter_frame += 1
            
            if not counter:
                # fit the generator in order to transform data (only once)
                train_datagen.fit(sequence)
                
            X[counter] = train_datagen.standardize(sequence)
            counter += 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test


def video_generator(real_path, fake_path, img_size = 299, 
                    frame = 40, batch_size = 10, start_idx=(1,1)):
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
    
    real_num, fake_num = batch_size//2, batch_size-batch_size//2
    
    # placeholder for inputs and targets
    X = np.zeros(batch_size, frame, img_size, img_size, 3)
    y = np.concatenate((np.zeros(real_num, dtype=int),
                        np.ones(fake_num, dtype=int))) 
    
    # we use this object to preprocess video
    train_datagen = ImageDataGenerator(samplewise_center=True)
    
    total_real, total_fake = start_idx
    
    # set index array for reshuffle
    index = np.arange(batch_size)
    while True:
        # same structure of the load_data function
        # it's a generator function so don't worry about the infinite loop
        counter = 0
        
        # placeholder for each video
        sequence = np.zeros(frame, img_size, img_size, 3)
        for folder,num,total in [(real_path, real_num, total_real), 
                            (fake_path, fake_num, total_fake)]:
            
            for i in range(total,num+total):
                src = join(folder, str(i))
                counter_frame = 0
                
                # must ensure frame <= len(os.listdir(src))
                for img in os.listdir(src):
                    img_src = join(src, img)
                    img_array = cv2.imread(img_src)
                    img_array = cv2.resize(img_array, (img_size, img_size), 
                                         interpolation = cv2.INTER_CUBIC)
                    
                    sequence[counter_frame] = img_array
                    counter_frame += 1
                    if counter_frame >= frame:
                        break
                
                if not (total_real - start_idx[0]):
                    train_datagen.fit(sequence)
                    
                X[counter] = train_datagen.standardize(sequence)
                counter += 1
            
            total_real += real_num 
            total_fake += fake_num 
            
            # reshuffle the batch to make trained model more robust
            np.random.shuffle(index)
        
        # use yield here instead of return so the function will only
        # return values unless being called again by the model
        yield (X[index], y[index])
        
# To use the generator function:
# model.fit_generator(
#   video_generator("~\real", "~\fake", img_size = 299, 
#   frame = 40, batch_size = 10, start_idx=(1,1)),
#   steps_per_epoch = 20)

# Here we have 10 videos per batch (5 real, 5 fake),
# and 200 videos in total (20 batches per epoch)
        
        
    