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
import sys
from collections import Counter

currdir = os.path.dirname(os.path.abspath(__file__))

model = keras.models.load_model(os.path.join(currdir, "first_CNN"))

not_prog_mel_dir = os.path.join(currdir, "Test_Mel_Spect\\All_Non_Prog")
prog_mel_dir = os.path.join(currdir, "Test_Mel_Spect\\Progressive")

true = []
apredicted = []

for filename in os.listdir(not_prog_mel_dir): 

    X = []
    Y = []

    print("Reading song: " + filename)
    csv_file_path = os.path.join(not_prog_mel_dir, filename)
    
    curr_data = []

    with open(csv_file_path, 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        for row in csvreader:
            curr_data.append([float(i) for i in row])

    curr_data = np.array(curr_data)

    for n in range(int(curr_data.shape[1]/128)):
        X.append(curr_data[:,n*128:(n+1)*128])
        Y.append([0,1])

    X = np.array([X])/(-80.0)
    X = np.swapaxes(X, 0, 1)
    X = np.swapaxes(X, 1, 2)
    X = np.swapaxes(X, 2, 3)

    predict = model.predict(X)

    predicted = np.argmax(model.predict(X), axis = 1)

    onecount = 0
    zerocount = 0

    for j in predicted:
        if j == 1:
            onecount += 1
        else:
            zerocount += 1

    if onecount >= zerocount:
        apredicted.append(1)
    else:
        apredicted.append(0)

    true.append(1)

    print(predicted)
    

for filename in os.listdir(prog_mel_dir):

    X = []
    Y = []

    print("Reading Progressive Rock song: " + filename)

    csv_file_path = os.path.join(prog_mel_dir, filename)
    
    curr_data = []

    with open(csv_file_path, 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        for row in csvreader:
            curr_data.append([float(i) for i in row])
            
    curr_data = np.array(curr_data)

    for n in range(int(curr_data.shape[1]/128)):
        X.append(curr_data[:,n*128:(n+1)*128])
        Y.append([1,0])

    X = np.array([X])/(-80.0)
    X = np.swapaxes(X, 0, 1)
    X = np.swapaxes(X, 1, 2)
    X = np.swapaxes(X, 2, 3)

    predict = model.predict(X)

    predicted = np.argmax(model.predict(X), axis = 1)

    onecount = 0
    zerocount = 0

    for j in predicted:
        if j == 1:
            onecount += 1
        else:
            zerocount += 1

    if onecount >= zerocount:
        apredicted.append(1)
    else:
        apredicted.append(0)

    print(predicted)

    true.append(0)

correct_count = 0

for i in range(len(true)):
    if true[i] == apredicted[i]:
        correct_count += 1

print("Percentage predicted correctly: " + str(correct_count/len(true)))

cm = confusion_matrix(true, apredicted)

display_labels = np.array(["Progressive Rock", "All Not Progressive Rock Sounds"])

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=display_labels)
disp.plot(include_values=True, cmap=plt.cm.Blues, colorbar=True, ax = None, values_format=None, xticks_rotation='horizontal')

plt.show()






