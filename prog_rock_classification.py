import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import os
import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


#y, sr = librosa.load('./1939_Judy_Garland_Somewhere_Over_The_Rainbow.mp3')

#mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels = 256)
#mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

#librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time')
#plt.show()

not_prog_directory = "./CAP6610sp21_Training_Set/Not_Progressive_Rock"
prog_directory = "./CAP6610sp21_Training_Set/Progressive_Rock_Songs"

not_prog_data = []
prog_data = []

for filename in os.listdir(not_prog_directory):
    print("Reading song: " + filename)
    y, sr = librosa.load(os.path.join(not_prog_directory, filename))
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    not_prog_data.append(mel_spect)

for filename in os.listdir(prog_directory):
    print("Reading progressive rock song: " + filename)
    y, sr = librosa.load(os.path.join(prog_directory, filename))
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    prog_data.append(mel_spect)

X = []
Y = []

for song in not_prog_data:
    for n in range(int(song.shape[1]/128)):
        X.append(song[:,n*128:(n+1)*128])
        Y.append([0,1])

for song in prog_data:
    for n in range(int(song.shape[1]/128)):
        X.append(song[:,n*128:(n+1)*128])
        Y.append([1,0])

X = np.array([X])
X = np.swapaxes(X, 0, 1)
X = np.swapaxes(X, 1, 2)
X = np.swapaxes(X, 2, 3)

Y = np.array(Y)

model = Sequential()

model.add(Conv2D(64, kernel_size=(12, 12), activation='relu', input_shape=(128,128,1)))

model.add(Conv2D(128, (12, 12), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model_log = model.fit(X, Y, batch_size = 50, epochs = 100, verbose = 1)

model.save("first_cnn")
