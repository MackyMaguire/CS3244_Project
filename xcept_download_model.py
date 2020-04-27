import math
import os
import matplotlib
import imghdr
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.xception import Xception
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

def main():
    # ====================================================
    # Download and build a custom Xception
    # ====================================================
    # instantiate pre-trained Xception model
    base_model = Xception(
        include_top=False,
        weights='imagenet',
        input_shape=(299, 299, 3))

    # save base model
    base_model_json = base_model.to_json()
    name = "xcept_base_model"
    with open(name + ".json", "w") as json_file:
        json_file.write(base_model_json)
        
    base_model.save_weights(name + "_weight.h5")
    print("Saved base model to disk")

    # create a custom top classifier
    num_classes = 2
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.inputs, outputs=predictions)

    # save custom model
    model_json = model.to_json()
    name = "xcept_custom_model"
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)
        
    model.save_weights(name + "_weight.h5")
    print("Saved custom model to disk")

if __name__ == '__main__':
    main()
