from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

import statistics
import numpy as np

from models import *
from preprocessing import *

'''
def test_generator(X, y, dataGenerator, batch_size):
  for x_batch, y_batch in dataGenerator.flow(X, y, batch_size):
    yield(x_batch, y_batch)
'''

def evaluate_model(model,real_path,fake_path):
  # num_videos = len(x_test) // 40
  num_videos = 0
  num_correct = 0
  # for i in range(num_videos):
  for x_test, y_test in load_data_generator(real_path,fake_path):
    video = x_test
    num_videos += 1

    pred_array = model.predict(video)
    classification_array = map(lambda x: int(round(x[0])), pred_array)
    arr = np.array(list(classification_array))
    video_mean = np.bincount(arr).argmax()
    classification = 1 if video_mean >= 0.5 else 0
    if classification == y_test[0]:
      num_correct += 1

  with open("{}_result.txt".format(model.name), 'w') as f:
    f.write('correctly classified: {}/{}\n'.format(num_correct,num_videos) + 'accuracy: ' + str(num_correct/num_videos) + '\n')

def main():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    real_path = os.path.join(current_directory,"Test/Real/Frames")
    fake_path = os.path.join(current_directory,"Test/Fake/Frames")
    
    meso4 = Meso4()
    meso4.load(os.path.join(current_directory,'Meso4.h5'))
    evaluate_model(meso4.model, real_path, fake_path)

    mesoInception4 = MesoInception4()
    mesoInception4.load(os.path.join(current_directory,'MesoInception.h5'))
    evaluate_model(mesoInception4.model, real_path, fake_path)

if __name__ == '__main__':
    main()