import numpy as np
from models import *

from keras.preprocessing.image import ImageDataGenerator

# Initialize model
model = Meso4()

# Generator for model
dataGenerator = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        rotation_range=15,
        brightness_range=(0.8, 1.2),
        channel_shift_range=30,
        horizontal_flip=True,
        validation_split=0.15)

generator = dataGenerator.flow_from_directory(
        "../data/test/frames",
        target_size=(256, 256),
        batch_size=32,
        class_mode="binary",
        subset="training")

# 2 - train and save
try:
    for x_batch, y_batch in generator:
        loss = model.fit(x_batch, y_batch)
        print(loss)
except KeyboardInterrupt:
    # Save model on interrupt
    model.save("model.h5")
    print('Output saved to: "{}."'.format("model.h5"))

# 3 - save model
model.save("model.h5")
print('Output saved to: "{}."'.format("model.h5"))
