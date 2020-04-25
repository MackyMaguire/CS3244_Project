import keras
import numpy as np
from keras.applications import inception_v3
from keras.models import model_from_json, Model
from keras.layers import Input, Dense, Dropout, LSTM, TimeDistributed

def new_model(image_size = 299, video_length = 40, cnn_trainable = False):
    inputs = Input(shape=(video_length, image_size, image_size, 3))
    cnn = inception_v3.InceptionV3(include_top=True, weights='imagenet')
    model = TimeDistributed(cnn)(inputs)
    model.trainable = cnn_trainable
    
    model = LSTM(512)(model)
    model = Dropout(0.5)(model)
    model = Dense(1, activation='softmax')(model)
    model = Model(inputs=inputs, outputs=model)
    
    adam = keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(loss='binary_crossentropy', optimizer=adam, 
                  metrics=['accuracy'])
    
    model.summary()
    return model

def save_model(model,name):
    model_json = model.to_json()
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)
        
    model.save_weights(name + "_weight.h5")
    print("Saved model to disk")
    
def load_model(name):
    json_file = open(name + '.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    
    model.load_weights(name + "_weight.h5")
    print("Loaded model from disk")
 
    adam = keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(loss='binary_crossentropy', optimizer = adam, 
                  metrics=['accuracy'])
    print("Model compiled")
    return model