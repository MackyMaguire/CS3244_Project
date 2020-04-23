import numpy as np
from models import *

from keras.preprocessing.image import ImageDataGenerator

# 1 - Load the model and its pretrained weights
classifier = Meso4()
classifier.load('weights/Meso4_DF')

# 2 - Minimial image generator
dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
        '../data/frames',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary',
        subset='training')

# 3 - Predict
for X, y in generator:
    print('Predicted :', classifier.predict(X), '\nReal class :', y)


# # For videos -WIP
# def predict_faces(generator, classifier, batch_size = 50, output_size = 1):
#     '''
#     Compute predictions for a face batch generator
#     '''
#     n = len(generator.finder.coordinates.items())
#     profile = np.zeros((1, output_size))
#     for epoch in range(n // batch_size + 1):
#         face_batch = generator.next_batch(batch_size = batch_size)
#         prediction = classifier.predict(face_batch)
#         if (len(prediction) > 0):
#             profile = np.concatenate((profile, prediction))
#     return profile[1:]


# def compute_accuracy(classifier, dirname, frame_subsample_count = 30):
#     '''
#     Extraction + Prediction over a video
#     '''
#     filenames = [f for f in listdir(dirname) if isfile(join(dirname, f)) and ((f[-4:] == '.mp4') or (f[-4:] == '.avi') or (f[-4:] == '.mov'))]
#     predictions = {}
    
#     for vid in filenames:
#         print('Dealing with video ', vid)
        
#         # Compute face locations and store them in the face finder
#         face_finder = FaceFinder(join(dirname, vid), load_first_face = False)
#         skipstep = max(floor(face_finder.length / frame_subsample_count), 0)
#         face_finder.find_faces(resize=0.5, skipstep = skipstep)
        
#         print('Predicting ', vid)
#         gen = FaceBatchGenerator(face_finder)
#         p = predict_faces(gen, classifier)
        
#         predictions[vid[:-4]] = (np.mean(p > 0.5), p)
#     return predictions