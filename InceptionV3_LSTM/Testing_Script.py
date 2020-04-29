import os
from InceptionV3_LSTM import load_model, save_model
from preprocessing import video_generator, video_generator_test


current_directory = os.path.dirname(os.path.abspath(__file__))

real_path = os.path.join(current_directory,"Test/Real/Frames")
fake_path = os.path.join(current_directory,"Test/Fake/Frames")
batch_size = 32
frame = 40


real_lst = []
fake_lst = []

for folder, lst, label in [(real_path,real_lst,False), (fake_path,fake_lst,True)]:
    for video in os.listdir(folder):
        src = join(folder,video)
        if frame <= len(os.listdir(src)):
            lst.append((src,label))

total_step = (len(real_lst)+len(real_lst))//batch_size+1

model = load_model("InceptionV3_LSTM(Trained)")

result = model.evaluate_generator(
    video_generator_test(real_path, fake_path, img_size = 299, 
                    frame = 40, batch_size = batch_size), 
    steps = total_step)

with open("result.txt", 'w') as f:
    f.write('loss: ' + str(result[0]) + '\n' + 'accuracy: ' + str(result[1]) + '\n')
