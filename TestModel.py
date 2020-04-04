import os
from os.path import join

import cv2

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, Conv2D, MaxPooling2D, TimeDistributed, BatchNormalization

# Initialising training set
# 8 input videos used
x_train = np.empty([8, 10, 28, 28, 3], dtype=int)
l1 = np.zeros((4), dtype=int)
l2 = np.ones((4), dtype=int)
y_train = np.concatenate((l1, l2), axis=None)

# Representing the sequence of frames for each input video as an (10, 28, 28, 3) array
# Each input video generated 10 frames
# Frames resized to 28*28 because my Com exploded for the original 720*1280
counter = 0
for folder in ['Real', 'Fake']:
    for i in range(1, 5):
        src = join(folder, str(i))
        
        sequence = []
        
        for img in os.listdir(src):
            img_src = join(src, img)
            img_array = cv2.imread(img_src)
            resized = cv2.resize(img_array, (28, 28), interpolation = cv2.INTER_CUBIC)
            sequence.append(resized)
        
        x_train[counter] = sequence
        counter += 1

# Test Model
model = Sequential()

model.add(TimeDistributed(Conv2D(32, (3, 3)), input_shape=(10, 28, 28, 3)))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))

model.add(TimeDistributed(Conv2D(32, (3,3))))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))

model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(TimeDistributed(Flatten()))
model.add(LSTM(64, return_sequences=False, dropout=0.5))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])
            
model.fit(x_train, y_train, epochs=5)