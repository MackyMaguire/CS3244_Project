import os
from os.path import join
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def load_data(real_path, fake_path, num_sample = (100,100), 
              img_size = 299, frame = 40):
    X = np.zeros(sum(num_sample), frame, img_size, img_size, 3)
    y = np.concatenate((np.zeros(num_sample[0], dtype=int),
                        np.ones(num_sample[1], dtype=int))) 
    
    train_datagen = ImageDataGenerator(samplewise_center=True)
    
    counter = 0
    for folder, num in [(real_path, num_sample[0]), 
                        (fake_path, num_sample[1])]:
        for i in range(1, num+1):
            src = join(folder, str(i))
            sequence = np.zeros(frame, img_size, img_size, 3)
            
            counter_frame = 0
            for img in os.listdir(src):
                img_src = join(src, img)
                img_array = cv2.imread(img_src)
                img_array = cv2.resize(img_array, (img_size, img_size), 
                                     interpolation = cv2.INTER_CUBIC)
                
                sequence[counter_frame] = img_array
                counter_frame += 1
            
            if not counter:
                train_datagen.fit(sequence)
                
            X[counter] = train_datagen.standardize(sequence)
            counter += 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test
    