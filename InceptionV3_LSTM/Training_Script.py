import os
from InceptionV3_LSTM import load_model, save_model
from preprocessing import video_generator, video_generator_test

current_directory = os.path.dirname(os.path.abspath(__file__))

real_path = os.path.join(current_directory,"Train/Real/Frames")
fake_path = os.path.join(current_directory,"Train/Fake/Frames")
batch_size = 32
frame = 40

real_lst = []
fake_lst = []

for folder, lst, label in [(real_path,real_lst,False), (fake_path,fake_lst,True)]:
    for video in os.listdir(folder):
        src = os.path.join(folder,video)
        if frame <= len(os.listdir(src)):
            lst.append((src,label))

total_step = min(len(real_lst),len(fake_lst))*2//batch_size+1

model = load_model("InceptionV3_LSTM")

model.fit_generator(
    video_generator(real_path, fake_path, img_size = 299, 
                    frame = 40, batch_size = batch_size),
    steps_per_epoch = total_step)

save_model(model,"InceptionV3_LSTM_Trained")
