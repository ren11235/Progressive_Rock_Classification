import os
import librosa
import numpy as np
import csv

currdir = os.path.dirname(os.path.abspath(__file__))
not_prog_directory = os.path.join(currdir,"CAP6610sp21_Training_Set\\Not_Progressive_Rock")
prog_directory = os.path.join(currdir,"CAP6610sp21_Training_Set\\Progressive_Rock")

not_prog_mfcc_dir = os.path.join(currdir, "MFCC\\Not_Progressive")
prog_mfcc_dir = os.path.join(currdir, "MFCC\\Progressive")

for filename in os.listdir(not_prog_directory):
    print("Reading_song: ", filename)
    y, sr = librosa.load(os.path.join(not_prog_directory,filename))
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    
    csv_file_path = os.path.join(not_prog_mfcc_dir, filename[0:filename.find(".mp3")] + ".csv")
    with open(csv_file_path, 'w+', newline = '') as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerows(mfccs)
        
for filename in os.listdir(prog_directory):
    print("Reading_song: ", filename)
    y, sr = librosa.load(os.path.join(prog_directory,filename))
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    
    csv_file_path = os.path.join(prog_mfcc_dir, filename[0:filename.find(".mp3")] + ".csv")
    with open(csv_file_path, 'w+', newline = '') as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerows(mfccs)
        