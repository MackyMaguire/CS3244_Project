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

def evaluate_model(model, x_test, y_test):
  num_videos = len(x_test) // 40
  num_correct = 0
  for i in range(num_videos):
    video = x_test[i*40:(i+1)*40]
    pred_array = model.predict(video)
    classification_array = map(lambda x: int(round(x[0])), pred_array)
    arr = np.array(list(classification_array))
    video_mean = np.bincount(arr).argmax()
    classification = 1 if video_mean >= 0.5 else 0
    if classification == y_test[i*40]:
      num_correct += 1

  with open("{}_result.txt".format(model.name), 'w') as f:
    f.write('correctly classified: {}/{}\n'.format(num_correct,num_videos) + 'accuracy: ' + str(num_correct/num_videos) + '\n')

def main():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    real_path = os.path.join(current_directory,"Test/Real/Frames")
    fake_path = os.path.join(current_directory,"Test/Fake/Frames")

    # To change to num_sample = (number_of_real_vids, number_of_fake_vids)
    x_test, y_test = load_data(real_path,fake_path, num_sample = (2,8))
    
    meso4 = Meso4()
    meso4.load(os.path.join(current_directory,'Meso4.h5'))
    evaluate_model(meso4.model, x_test, y_test)

    meso4 = MesoInception4()
    meso4.load(os.path.join(current_directory,'MesoInception.h5'))
    evaluate_model(meso4.model, x_test, y_test)

if __name__ == '__main__':
    main()
