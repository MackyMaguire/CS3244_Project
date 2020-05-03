import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from statistics import mean
from os.path import join
from keras.preprocessing.image import ImageDataGenerator

from models import *
from preprocessing import *

def train_model(model, dataGenerator, real_path, fake_path, epochs):
    model_name = model.name
    loss = []
    accuracy = []
    try:
        for e in range(epochs):
            print('Epoch', e)
            batches = 0
            batch_loss = []
            batch_accuracy = []
            for x_train, y_train in load_data_generator(real_path, fake_path):
              for x_batch, y_batch in dataGenerator.flow(x_train, y_train, batch_size=40):
                  hist = model.fit(x_batch, y_batch)
                  batch_loss.append(hist[0])
                  batch_accuracy.append(hist[1])
                  batches += 1
                  if batches >= len(x_train) / 40:
                      # we need to break the loop by hand because
                      # the generator loops indefinitely
                      break
            loss.append(mean(batch_loss))
            accuracy.append(mean(batch_accuracy))
    except KeyboardInterrupt:
        # Save model on interrupt
        model.save("{}.h5".format(model_name))
        print('Output saved to: "{}.h5"'.format(model_name))

    # save model
    model.save("{}.h5".format(model_name))
    print('Output saved to: "{}.h5"'.format(model_name))
    
    return loss, accuracy

def train_and_plot(model, dataGenerator, real_path, fake_path):
    initial_epoch = 1
    slow_epoch = 4
    normal_epoch = 16
    epochs = initial_epoch + slow_epoch + normal_epoch
    
    model.freeze()
    print("Train output layer while preserving other layers...")
    loss, accuracy = train_model(model, dataGenerator, real_path, fake_path, initial_epoch)

    model.unfreeze()
    model.set_lr(1e-4)
    print("Train all layers with slow lr to adjust weights...")
    loss2, accuracy2 = train_model(model, dataGenerator, real_path, fake_path, slow_epoch)
    loss.extend(loss2)
    accuracy.extend(accuracy2)

    model.set_lr(1e-3)
    print("Train all layers with normal lr...")
    loss3, accuracy3 = train_model(model, dataGenerator, real_path, fake_path, normal_epoch)
    loss.extend(loss3)
    accuracy.extend(accuracy3)

    # save graph image
    plt.plot(range(epochs), loss, marker='.', label='loss')
    plt.plot(range(epochs), accuracy, marker='.', label='accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss/accuracy')
    plt.savefig('{}_train.png'.format(model.name))
    plt.show()
    plt.clf()

def main():

    current_directory = os.path.dirname(os.path.abspath(__file__))
    real_path = os.path.join(current_directory,"Train/Real/Frames")
    fake_path = os.path.join(current_directory,"Train/Fake/Frames")
    
    dataGenerator = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        rotation_range=15,
        brightness_range=(0.8, 1.2),
        channel_shift_range=30,
        horizontal_flip=True,
        validation_split=0.15)

    # Meso4
    meso4 = Meso4()
    meso4.load('weights/Meso4_DF')
    print("Now training: meso4")
    train_and_plot(meso4, dataGenerator, real_path, fake_path)

    # MesoInception4
    mesoInception4 = MesoInception4()
    mesoInception4.load('weights/MesoInception4_DF')
    print("Now training: mesoInception4")
    train_and_plot(mesoInception4, dataGenerator, real_path, fake_path)

if __name__ == '__main__':
    main()