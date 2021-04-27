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

model = keras.models.load_model(os.path.join(currdir, "second_CNN"))

not_prog_chroma_dir = os.path.join(currdir, "Test_Chroma\\Not_Progressive_Rock")
other_chroma_dir = os.path.join(currdir, "Test_Chroma\\Other")
prog_chroma_dir = os.path.join(currdir, "Test_Chroma\\Progressive_Rock")

true_non_prog = []
true_prog = []
true_other = []
apredicted_non_prog = []
apredicted_prog = []
apredicted_other = []

correct_count_non_prog = 0
correct_count_prog = 0
correct_count_other = 0

for filename in os.listdir(not_prog_chroma_dir): 

    X = []
    Y = []

    print("Reading song: " + filename)
    csv_file_path = os.path.join(not_prog_chroma_dir, filename)
    
    curr_data = []

    with open(csv_file_path, 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        for row in csvreader:
            curr_data.append([float(i) for i in row])

    curr_data = np.array(curr_data)

    for n in range(int(curr_data.shape[1]/12)):
        X.append(curr_data[:,n*12:(n+1)*12])
        Y.append([0,1])

    X = np.array([X])
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
        apredicted_non_prog.append(1)
    else:
        apredicted_non_prog.append(0)

    true_non_prog.append(1)

    #print(predicted)
    
for i in range(len(true_non_prog)):
    if true_non_prog[i] == apredicted_non_prog[i]:
        correct_count_non_prog += 1
    
for filename in os.listdir(other_chroma_dir): 

    X = []
    Y = []

    print("Reading song: " + filename)
    csv_file_path = os.path.join(other_chroma_dir, filename)
    
    curr_data = []

    with open(csv_file_path, 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        for row in csvreader:
            curr_data.append([float(i) for i in row])

    curr_data = np.array(curr_data)

    for n in range(int(curr_data.shape[1]/12)):
        X.append(curr_data[:,n*12:(n+1)*12])
        Y.append([0,1])

    X = np.array([X])
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
        apredicted_other.append(1)
    else:
        apredicted_other.append(0)

    true_other.append(1)

    #print(predicted)
    
for i in range(len(true_other)):
    if true_other[i] == apredicted_other[i]:
        correct_count_other += 1
    
for filename in os.listdir(prog_chroma_dir):

    X = []
    Y = []

    print("Reading Progressive Rock song: " + filename)

    csv_file_path = os.path.join(prog_chroma_dir, filename)
    
    curr_data = []

    with open(csv_file_path, 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        for row in csvreader:
            curr_data.append([float(i) for i in row])
            
    curr_data = np.array(curr_data)

    for n in range(int(curr_data.shape[1]/12)):
        X.append(curr_data[:,n*12:(n+1)*12])
        Y.append([1,0])

    X = np.array([X])
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
        apredicted_prog.append(1)
    else:
        apredicted_prog.append(0)

    #print(predicted)

    true_prog.append(0)

for i in range(len(true_prog)):
    if true_prog[i] == apredicted_prog[i]:
        correct_count_prog += 1

print("Percentage predicted correctly: " + str((correct_count_prog + correct_count_non_prog)/(len(true_non_prog) + len(true_prog))))
print("Percentage predicted correctly with other songs: " + str((correct_count_prog + correct_count_other)/(len(true_other) + len(true_prog))))

true = np.concatenate([true_non_prog, true_prog])
apredicted = np.concatenate([apredicted_non_prog, apredicted_prog])

cm = confusion_matrix(true, apredicted)

display_labels = np.array(["Progressive Rock", "Non Progressive Rock"])

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=display_labels)
disp.plot(include_values=True, cmap=plt.cm.Blues, colorbar=True, ax = None, values_format=None, xticks_rotation='horizontal')

plt.show()


true_other_prog = np.concatenate([true_other, true_prog])
apredicted_other_prog = np.concatenate([apredicted_other, apredicted_prog])

cm_other = confusion_matrix(true_other_prog, apredicted_other_prog)

display_labels = np.array(["Progressive Rock", "Others"])

disp = ConfusionMatrixDisplay(confusion_matrix=cm_other,
                                display_labels=display_labels)
disp.plot(include_values=True, cmap=plt.cm.Blues, colorbar=True, ax = None, values_format=None, xticks_rotation='horizontal')

plt.show()