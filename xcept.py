import math
import os
import matplotlib
import imghdr
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.xception import preprocess_input
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.models import Model, model_from_json
from keras.utils import to_categorical

matplotlib.use('Agg')

def generate_train_from_paths_and_labels(
        input_paths, labels, batch_size, input_size=(299, 299)):
    num_samples = len(input_paths)
    while 1:
        perm = np.random.permutation(num_samples)
        input_paths = input_paths[perm]
        labels = labels[perm]
        for i in range(0, num_samples, batch_size):
            inputs = list(map(
                lambda x: image.load_img(x, target_size=input_size),
                input_paths[i:i+batch_size]
            ))
            inputs = np.array(list(map(
                lambda x: image.img_to_array(x),
                inputs
            )))
            inputs = preprocess_input(inputs)
            yield (inputs, labels[i:i+batch_size])

def main():
    current_directory = os.path.dirname(os.path.abspath(__file__))

    train_set_root = 'Train'
    test_set_root = 'Test'

    train_frame_root = os.path.join(current_directory, train_set_root)
    # /project/Train

    test_frame_root = os.path.join(current_directory, test_set_root)
    # /project/Test

    result_root = 'result_xception'
    result_root = os.path.join(current_directory, result_root)
    # /project/result_xception

    epochs_pre = 1 #5
    epochs_fine = 5 #50
    batch_size_pre = 32
    batch_size_fine = 16
    lr_pre = 1e-3
    lr_fine = 1e-4
    train_split = 0.7/0.85

    # ====================================================
    # Preparation: load training data
    # ====================================================
    # parameters
    epochs = epochs_pre + epochs_fine
    train_frame_root = os.path.expanduser(train_frame_root)
    result_root = os.path.expanduser(result_root)

    classes = ['Fake', 'Real']
    num_classes = 2

    # make input_paths and labels
    input_paths, labels = [], []
    for class_name in ['Fake', 'Real']:
        class_root = os.path.join(train_frame_root, class_name)
        class_id = classes.index(class_name)

        frame_root = os.path.join(class_root, 'Frames')

        for folder_path in os.listdir(frame_root):
            folder_path = os.path.join(frame_root, folder_path)
            for img_path in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_path)
                if imghdr.what(img_path) is None:
                    # this is not an image file
                    continue
                input_paths.append(img_path)
                labels.append(class_id)

    labels = to_categorical(labels, num_classes=num_classes)

    # convert to numpy array
    input_paths = np.array(input_paths)

    # shuffle train dataset
    perm = np.random.permutation(len(input_paths))
    labels = labels[perm]
    input_paths = input_paths[perm]

    # split dataset for training and validation
    train_border = int(len(input_paths) * train_split)
    train_labels = labels[:train_border]
    val_labels = labels[train_border:]
    train_input_paths = input_paths[:train_border]
    val_input_paths = input_paths[train_border:]

    print("======Training on %d images and labels======" % (len(train_input_paths)))
    print("======Validation on %d images and labels======" % (len(val_input_paths)))

    # create a directory where results will be saved (if necessary)
    if os.path.exists(result_root) is False:
        os.makedirs(result_root)

    # ====================================================
    # Load base and custom Xception model
    # ====================================================
    # load base model
    name = "xcept_base_model"
    json_file = open(name + '.json', 'r')
    base_model_json = json_file.read()
    json_file.close()
    base_model = model_from_json(base_model_json)
    
    base_model.load_weights(name + "_weight.h5")
    print("Loaded base model from disk")

    # load custom model
    name = "xcept_custom_model"
    json_file = open(name + '.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    
    model.load_weights(name + "_weight.h5")
    print("Loaded custom model from disk")

    # ====================================================
    # Train only the top classifier
    # ====================================================
    # freeze the body layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=lr_pre),
        metrics=['accuracy']
    )

    # train
    hist_pre = model.fit_generator(
        generator=generate_train_from_paths_and_labels(
            input_paths=train_input_paths,
            labels=train_labels,
            batch_size=batch_size_pre
        ),
        steps_per_epoch=math.ceil(
            len(train_input_paths) / batch_size_pre),
        epochs=epochs_pre,
        validation_data=generate_train_from_paths_and_labels(
            input_paths=val_input_paths,
            labels=val_labels,
            batch_size=batch_size_pre
        ),
        validation_steps=math.ceil(
            len(val_input_paths) / batch_size_pre)
    )

    # ====================================================
    # Train the whole model
    # ====================================================
    # set all the layers to be trainable
    for layer in model.layers:
        layer.trainable = True

    # recompile
    model.compile(
        optimizer=Adam(lr=lr_fine),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    # train
    hist_fine = model.fit_generator(
        generator=generate_train_from_paths_and_labels(
            input_paths=train_input_paths,
            labels=train_labels,
            batch_size=batch_size_fine
        ),
        steps_per_epoch=math.ceil(
            len(train_input_paths) / batch_size_fine),
        epochs=epochs_fine,
        validation_data=generate_train_from_paths_and_labels(
            input_paths=val_input_paths,
            labels=val_labels,
            batch_size=batch_size_fine
        ),
        validation_steps=math.ceil(
            len(val_input_paths) / batch_size_fine)
    )

    model.save(os.path.join(result_root, 'model_xcept.h5'))

    # ====================================================
    # Create & save result graphs
    # ====================================================
    # concatinate plot data
    acc = hist_pre.history['accuracy']
    val_acc = hist_pre.history['val_accuracy']
    loss = hist_pre.history['loss']
    val_loss = hist_pre.history['val_loss']
    acc.extend(hist_fine.history['accuracy'])
    val_acc.extend(hist_fine.history['val_accuracy'])
    loss.extend(hist_fine.history['loss'])
    val_loss.extend(hist_fine.history['val_loss'])

    # save graph image
    plt.plot(range(epochs), acc, marker='.', label='accuracy')
    plt.plot(range(epochs), val_acc, marker='.', label='val_accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(os.path.join(result_root, 'accuracy.png'))
    plt.clf()

    plt.plot(range(epochs), loss, marker='.', label='loss')
    plt.plot(range(epochs), val_loss, marker='.', label='val_loss')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(result_root, 'loss.png'))
    plt.clf()

    # ====================================================
    # Preparation: load testing data
    # ====================================================

    # make test_input_paths and test_labels
    test_input_paths, test_labels = [], []
    for class_name in ['Fake', 'Real']:
        class_root = os.path.join(test_frame_root, class_name)
        class_id = classes.index(class_name)

        frame_root = os.path.join(class_root, 'Frames')

        for folder_path in os.listdir(frame_root):
            folder_path = os.path.join(frame_root, folder_path)
            current_paths = []
            current_labels = []
            for img_path in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_path)
                if imghdr.what(img_path) is None:
                    # this is not an image file
                    continue
                current_paths.append(img_path)
                current_labels.append(class_id)
            current_paths = np.array(current_paths)
            current_labels = to_categorical(current_labels, num_classes=num_classes)
            test_input_paths.append(current_paths)
            test_labels.append(current_labels)

    # ====================================================
    # Test on test set
    # ====================================================
    correct_counter = 0

    for i in range(len(test_input_paths)):
        current_paths = test_input_paths[i]
        current_label = test_labels[i][0]

        inputs = list(map(
            lambda x: image.load_img(x, target_size=(299, 299)),
            current_paths[:]
        ))
        inputs = np.array(list(map(
            lambda x: image.img_to_array(x),
            inputs
        )))

        inputs = preprocess_input(inputs)

        pred = model.predict(inputs)

        real_counter = 0

        for i in range(len(pred)):
            result = pred[i]
            fake_prob = float(result[0])
            real_prob = float(result[1])
            if fake_prob < real_prob:
                real_counter += 1

        real_prob = real_counter / len(pred)

        if real_prob > 0.5:
            pred_label = [0.0, 1.0]
        else:
            pred_label = [1.0, 0.0]

        if pred_label == current_label.tolist():
            correct_counter += 1

    print("==== Predict Correct Video: %d ====" % (correct_counter))
    print("==== Total Test Video: %d ====" % (len(test_input_paths)))
    test_acc = correct_counter / len(test_input_paths)
    print("===== Test Accuracy: %.2f =====" % (test_acc))

if __name__ == '__main__':
    main()
