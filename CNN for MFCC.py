import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import os
import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import csv
import random

currdir = os.path.dirname(os.path.abspath(__file__))

not_prog_mfcc_dir = os.path.join(currdir, "MFCC\\Not_Progressive")
prog_mfcc_dir = os.path.join(currdir, "MFCC\\Progressive")

X = []
Y = []

#j = 0

for filename in os.listdir(not_prog_mfcc_dir): 
    print("Reading song: " + filename)

    csv_file_path = os.path.join(not_prog_mfcc_dir, filename)
    
    curr_data = []

    with open(csv_file_path, 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        for row in csvreader:
            curr_data.append([float(i) for i in row])

    curr_data = np.array(curr_data)
    curr_data = (curr_data + np.min(curr_data))/np.abs(np.max(curr_data)-np.min(curr_data))

    for n in range(int(curr_data.shape[1]/20)):
        X.append(np.array(curr_data[:,n*20:(n+1)*20], dtype = 'float32'))
        Y.append(np.array([0.,1.], dtype='float32'))
        
#    if j >= 5:
#        break
#    else:
#        j += 1

#j = 0

for filename in os.listdir(prog_mfcc_dir):
    print("Reading Progressive Rock song: " + filename)

    csv_file_path = os.path.join(prog_mfcc_dir, filename)
    
    curr_data = []

    with open(csv_file_path, 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        for row in csvreader:
            curr_data.append([float(i) for i in row])
            
    curr_data = np.array(curr_data)
    curr_data = (curr_data + np.min(curr_data))/np.abs(np.max(curr_data)-np.min(curr_data))

    for n in range(int(curr_data.shape[1]/20)):
        X.append(np.array(curr_data[:,n*20:(n+1)*20], dtype = 'float32'))
        Y.append(np.array([1.,0.], dtype='float32'))
        
#    if j >= 5:
#        break
#    else:
#        j += 1

          
X = np.array([X])
X = np.swapaxes(X, 0, 1)
X = np.swapaxes(X, 1, 2)
X = np.swapaxes(X, 2, 3)

Y = np.array(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = .8)

X = []
Y = []

model = Sequential()

model.add(Conv2D(64, kernel_size = (3,3), padding = "same", activation='relu', input_shape=(20,20,1)))

model.add(Conv2D(64, kernel_size = (3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(20, activation='relu'))

model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size = 50, epochs = 30, verbose = 1, validation_data=(x_test, y_test))

model.save(os.path.join(currdir, "second_CNN"))

y_predict = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_predict)

display_labels = np.array(["Progressive Rock", "Not Progressive Rock"])

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=display_labels)
disp.plot(include_values=True, cmap=plt.cm.Blues, colorbar=True, ax = None, values_format=None, xticks_rotation='horizontal')

plt.show()

y_predict = np.argmax(model.predict(x_train), axis=1)
y_true = np.argmax(y_train, axis=1)

cm = confusion_matrix(y_true, y_predict)

display_labels = np.array(["Progressive Rock", "Not Progressive Rock"])

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=display_labels)
disp.plot(include_values=True, cmap=plt.cm.Blues, colorbar=True, ax = None, values_format=None, xticks_rotation='horizontal')

plt.show()

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()