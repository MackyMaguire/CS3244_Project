import os
from os.path import join
import cv2
import numpy as np

def load_data(real_path, fake_path, num_sample = (1,1), 
              img_size = 256, frame = 40):

    X = np.zeros((sum(num_sample)*frame, img_size, img_size, 3))
    y = np.concatenate((np.zeros(num_sample[0]*frame, dtype=int),
                        np.ones(num_sample[1]*frame, dtype=int))) 
    
    counter = 0

    sequence = np.zeros((frame, img_size, img_size, 3))
    
    for folder, num in [(real_path, num_sample[0]), 
                        (fake_path, num_sample[1])]:
        
        # folder name starts from 1
        for i in range(1, num+1):
            src = join(folder, str(i)) # e.g ../Train/Real/1
            counter_frame = 0
            
            for img in os.listdir(src):
                img_src = join(src, img)
                img_array = cv2.imread(img_src)
                img_array = cv2.resize(img_array, (img_size, img_size), 
                                     interpolation = cv2.INTER_CUBIC)
                
                sequence[counter_frame] = img_array
                counter_frame += 1
                
            # X[counter:counter+frame] = sequence
            for i in range(len(sequence)):
              X[counter+i] = sequence[i]
            counter += frame
    
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]

    return X, y