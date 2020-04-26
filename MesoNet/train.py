import numpy as np
import os
import cv2
from os.path import join
from keras.preprocessing.image import ImageDataGenerator

from models import *

# Generator for model
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
    
    return X, y

def train_model(model, dataGenerator, x_train, y_train, epochs):
    model_name = model.name
    try:
        for e in range(epochs):
            print('Epoch', e)
            batches = 0
            for x_batch, y_batch in dataGenerator.flow(x_train, y_train, batch_size=32):
                loss = model.fit(x_batch, y_batch)
                print(loss)
                batches += 1
                if batches >= len(x_train) / 32:
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    break
    except KeyboardInterrupt:
        # Save model on interrupt
        model.save("{}.h5".format(model_name))
        print('Output saved to: "{}.h5"'.format(model_name))

    # save model
    model.save("{}.h5".format(model_name))
    print('Output saved to: "{}.h5"'.format(model_name))   

def main():
    # To change to num_sample = (number_of_real_vids, number_of_fake_vids)
    x_train, y_train = load_data("../Train/Real/Frames","../Train/Fake/Frames", num_sample = (1,1))
    
    dataGenerator = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        rotation_range=15,
        brightness_range=(0.8, 1.2),
        channel_shift_range=30,
        horizontal_flip=True,
        validation_split=0.15)
    dataGenerator.fit(x_train)

    # Initial training
    meso4 = Meso4()
    meso4.load('weights/Meso4_DF')
    # meso4.freeze()
    train_model(meso4, dataGenerator, x_train, y_train, 4)
    # meso4.unfreeze()
    # meso4.set_lr(1e-4))
    # train_model(meso4, dataGenerator, x_train, y_train, 16)
    # meso4.set_lr(1e-3))
    # train_model(meso4, dataGenerator, x_train, y_train, x) x can be a big number so we know when model overfits

    '''
    # Initial training
    mesoInception4 = MesoInception4()
    mesoInception4.load('weights/MesoInception4_DF')
    train_model(mesoInception4, dataGenerator, x_train, y_train, 4)
    '''

if __name__ == '__main__':
    main()