from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from models import *
from preprocessing import *

def test_generator(X, y, dataGenerator, batch_size):
  for x_batch, y_batch in dataGenerator.flow(X, y, batch_size):
    yield(x_batch, y_batch)

def evaluate_model(model, x_test, y_test):
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        rotation_range=15,
        brightness_range=(0.8, 1.2),
        channel_shift_range=30,
        horizontal_flip=True,
        validation_split=0.15)
    test_datagen.fit(x_test)

    batch_size = 32

    test_gen = test_generator(x_test, y_test, test_datagen, batch_size)

    total_step =  len(x_test) // batch_size

    result = model.evaluate_generator(generator = test_gen, steps = total_step)

    with open("{}_result.txt".format(model.name), 'w') as f:
        f.write('loss: ' + str(result[0]) + '\n' + 'accuracy: ' + str(result[1]) + '\n')

def main():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    real_path = os.path.join(current_directory,"Train/Real/Frames")
    fake_path = os.path.join(current_directory,"Train/Fake/Frames")

    # To change to num_sample = (number_of_real_vids, number_of_fake_vids)
    x_test, y_test = load_data(real_path,fake_path, num_sample = (1,1))
    
    meso4 = load_model("Meso4.h5")
    evaluate_model(meso4, x_test, y_test)

    mesoInception4 = load_model("MesoInception4.h5")
    evaluate_model(mesoInception4, x_test, y_test)

if __name__ == '__main__':
    main()