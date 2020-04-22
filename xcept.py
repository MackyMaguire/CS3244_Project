import math
import os
import matplotlib
import imghdr
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.xception import Xception, preprocess_input
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import to_categorical

matplotlib.use('Agg')

def generate_from_paths_and_labels(
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

    dataset_root = 'videos'
    dataset_root = os.path.join(current_directory, dataset_root)
    result_root = 'result_xception'
    result_root = os.path.join(current_directory, result_root)
    epochs_pre = 5
    epochs_fine = 50
    batch_size_pre = 32
    batch_size_fine = 16
    lr_pre = 1e-3
    lr_fine = 1e-4
    train_split = 0.7
    test_split = 0.15
    # ====================================================
    # Preparation
    # ====================================================
    # parameters
    epochs = epochs_pre + epochs_fine
    dataset_root = os.path.expanduser(dataset_root)
    result_root = os.path.expanduser(result_root)

    classes = ['Fake', 'Real']
    num_classes = 2

    # make input_paths and labels
    input_paths, labels = [], []
    for class_name in ['Fake', 'Real']:
        class_root = os.path.join(dataset_root, class_name)
        class_id = classes.index(class_name)
        for folder_path in os.listdir(class_root):
            # assume all mp4 files has been removed
            folder_path = os.path.join(class_root, folder_path)
            for img_path in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_path)
                if imghdr.what(img_path) is None:
                    # this is not an image file
                    continue
                input_paths.append(img_path)
                labels.append(class_id)

    labels = to_categorical(labels, num_classes=num_classes)
    # labels = np.asarray(labels)

    # convert to numpy array
    input_paths = np.array(input_paths)

    # shuffle dataset
    perm = np.random.permutation(len(input_paths))
    labels = labels[perm]
    input_paths = input_paths[perm]

    # split dataset for training and validation
    train_border = int(len(input_paths) * train_split)
    val_border = int(len(input_paths) * (1 - test_split))
    train_labels = labels[:train_border]
    val_labels = labels[train_border:val_border]
    test_labels = labels[val_border:]
    train_input_paths = input_paths[:train_border]
    val_input_paths = input_paths[train_border:val_border]
    test_input_paths = input_paths[val_border:]

    print("======Training on %d images and labels======" % (len(train_input_paths)))
    print("======Validation on %d images and labels======" % (len(val_input_paths)))
    print("======Testing on %d images and labels======" % (len(test_input_paths)))
    # create a directory where results will be saved (if necessary)
    if os.path.exists(result_root) is False:
        os.makedirs(result_root)

    # ====================================================
    # Build a custom Xception
    # ====================================================
    # instantiate pre-trained Xception model
    # the default input shape is (299, 299, 3)
    # NOTE: the top classifier is not included
    base_model = Xception(
        include_top=False,
        weights='imagenet',
        input_shape=(299, 299, 3))

    # create a custom top classifier
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.inputs, outputs=predictions)

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
        generator=generate_from_paths_and_labels(
            input_paths=train_input_paths,
            labels=train_labels,
            batch_size=batch_size_pre
        ),
        steps_per_epoch=math.ceil(
            len(train_input_paths) / batch_size_pre),
        epochs=epochs_pre,
        validation_data=generate_from_paths_and_labels(
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
        generator=generate_from_paths_and_labels(
            input_paths=train_input_paths,
            labels=train_labels,
            batch_size=batch_size_fine
        ),
        steps_per_epoch=math.ceil(
            len(train_input_paths) / batch_size_fine),
        epochs=epochs_fine,
        validation_data=generate_from_paths_and_labels(
            input_paths=val_input_paths,
            labels=val_labels,
            batch_size=batch_size_fine
        ),
        validation_steps=math.ceil(
            len(val_input_paths) / batch_size_fine)
    )

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
    # Test on test set
    # ====================================================

    inputs = list(map(
        lambda x: image.load_img(x, target_size=(299,299)),
        test_input_paths[:]
    ))
    inputs = np.array(list(map(
        lambda x: image.img_to_array(x),
        inputs
    )))

    inputs = preprocess_input(inputs)
    
    pred = model.predict(inputs)

    correct_counter = 0

    for i in range(len(pred)):
        result = pred[i]
        fake_prob = float(result[0])
        real_prob = float(result[1])
        if fake_prob > real_prob:
            label = [1.0, 0.0]
        else:
            label = [0.0, 1.0]
        if label == test_labels[i].tolist():
            correct_counter += 1

    test_acc = correct_counter / len(pred)
    print("===== Test Accuracy: %.2f =====" % (test_acc))

if __name__ == '__main__':
    main()
