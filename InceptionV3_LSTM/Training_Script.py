from InceptionV3_LSTM import load_model, save_model
from preprocessing import video_generator, video_generator_test


train_real_path = ""
train_fake_path = ""
batch_size = 0

real_lst = []
fake_lst = []

for folder, lst, label in [(real_path,real_lst,False), (fake_path,fake_lst,True)]:
    for video in os.listdir(folder):
        src = join(folder,video)
        if frame <= len(os.listdir(src)):
            lst.append((src,label))

total_step = min(len(real_lst),len(fake_lst))*2//batch_size+1

model = load_model("InceptionV3_LSTM")

model.fit_generator(
    video_generator(train_real_path, train_fake_path, img_size = 299, 
                    frame = 40, batch_size = batch_size),
    steps_per_epoch = total_step)

save_model(model,"InceptionV3_LSTM(Trained)")
