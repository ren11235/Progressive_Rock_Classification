import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import os
import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import csv
import random
import matplotlib.pyplot as plt


currdir = os.path.dirname(os.path.abspath(__file__))

model = keras.models.load_model(os.path.join(currdir, "first_CNN"))

not_prog_mel_dir = os.path.join(currdir, "Mel_Spect\\Not_Progressive")
prog_mel_dir = os.path.join(currdir, "Mel_Spect\\Progressive")

X = []
Y = []

for filename in os.listdir(not_prog_mel_dir): 
    print("Reading song: " + filename)

    csv_file_path = os.path.join(not_prog_mel_dir, filename)
    
    curr_data = []

    with open(csv_file_path, 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        for row in csvreader:
            curr_data.append([float(i) for i in row])

    curr_data = np.array(curr_data)

    for n in range(int(curr_data.shape[1]/128)):
        X.append(np.array(curr_data[:,n*128:(n+1)*128], dtype = 'float32'))
        Y.append(np.array([0.,1.], dtype='float32'))

    break

for filename in os.listdir(prog_mel_dir):
    print("Reading Progressive Rock song: " + filename)

    csv_file_path = os.path.join(prog_mel_dir, filename)
    
    curr_data = []

    with open(csv_file_path, 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        for row in csvreader:
            curr_data.append([float(i) for i in row])
            
    curr_data = np.array(curr_data)

    for n in range(int(curr_data.shape[1]/128)):
        X.append(np.array(curr_data[:,n*128:(n+1)*128], dtype = 'float32'))
        Y.append(np.array([1.,0.], dtype='float32'))
        
    break
X = np.array([X])/(-80.0)
X = np.swapaxes(X, 0, 1)
X = np.swapaxes(X, 1, 2)
X = np.swapaxes(X, 2, 3)

Y = np.array(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = .8)

y_predict = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_predict)

display_labels = np.array(["Progressive Rock", "Not Progressive Rock"])

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=display_labels)
disp.plot(include_values=True, cmap=plt.cm.Blues, colorbar=True, ax = None, values_format=None, xticks_rotation='horizontal')

plt.show()