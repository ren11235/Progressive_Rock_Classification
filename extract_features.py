import os
import librosa
import numpy as np
import csv

currdir = os.path.dirname(os.path.abspath(__file__))

not_prog_directory = os.path.join(currdir,"CAP6610sp21_Test_Set\\Not_Progressive_Rock")
prog_directory = os.path.join(currdir, "CAP6610sp21_Test_Set\\Progressive_Rock_Songs")

not_prog_mel_dir = os.path.join(currdir, "Test_Mel_Spect\\Not_Progressive")
prog_mel_dir = os.path.join(currdir, "Test_Mel_Spect\\Progressive")

for filename in os.listdir(not_prog_directory):

    print("Reading song: " + filename)
    y, sr = librosa.load(os.path.join(not_prog_directory, filename))
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    
    csv_file_path = os.path.join(not_prog_mel_dir, filename[0:filename.find(".mp3")] + ".csv")
    with open(csv_file_path, 'w+', newline='') as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerows(mel_spect)

for filename in os.listdir(prog_directory):
    print("Reading progressive rock song: " + filename)
    y, sr = librosa.load(os.path.join(prog_directory, filename))
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

    csv_file_path = os.path.join(prog_mel_dir, filename[0:filename.find(".mp3")] + ".csv")
    with open(csv_file_path, 'w+', newline='') as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerows(mel_spect)