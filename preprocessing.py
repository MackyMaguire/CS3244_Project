import os
from os.path import join
import cv2
import numpy as np

def load_data_generator(real_path, fake_path,
              img_size = 256, frame = 40):
    '''
    X = np.zeros((sum(num_sample)*frame, img_size, img_size, 3))
    y = np.concatenate((np.zeros(num_sample[0]*frame, dtype=int),
                        np.ones(num_sample[1]*frame, dtype=int))) 
    '''
    x_sequence = np.zeros((frame, img_size, img_size, 3))
    
    for frame_root in [real_path, fake_path]:
        
        path_arr = frame_root.split("/")
        is_real_path = path_arr[-2] == 'Real'
        y_sequence = np.ones(frame,dtype=int) if is_real_path else np.zeros(frame,dtype=int)

        for folder_path in os.listdir(frame_root):
            src = join(frame_root, folder_path)
            seq_counter = 0        
            for img in os.listdir(src):
                img_src = join(src, img)
                img_array = cv2.imread(img_src)
                img_array = cv2.resize(img_array, (img_size, img_size), 
                                     interpolation = cv2.INTER_CUBIC)
                
                x_sequence[seq_counter] = img_array
                seq_counter += 1

            yield (x_sequence, y_sequence)
    '''
            for i in range(len(sequence)):
              X[counter+i] = sequence[i]
            counter += frame
    
    return X, y
    '''