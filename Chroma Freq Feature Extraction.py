import os
import librosa
import numpy as np
import csv

import scipy
from scipy import ndimage

currdir = os.path.dirname(os.path.abspath(__file__))
not_prog_directory = os.path.join(currdir,"CAP6610sp21_Training_Set\\Not_Progressive_Rock")
prog_directory = os.path.join(currdir,"CAP6610sp21_Training_Set\\Progressive_Rock")

not_prog_chroma_dir = os.path.join(currdir, "Chroma\\Not_Progressive")
prog_chroma_dir = os.path.join(currdir, "Chroma\\Progressive")

for filename in os.listdir(not_prog_directory):
    print("Reading_song: ", filename)
    
    y, sr = librosa.load(os.path.join(not_prog_directory,filename))
    #Isolating harmonic component
    y_harm = librosa.effects.harmonic(y=y, margin=8)
    chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    #Non-local filtering to remove sparse additive noise
    chroma_filter = np.minimum(chroma_harm,
                           librosa.decompose.nn_filter(chroma_harm,
                                                       aggregate=np.median,
                                                       metric='cosine'))
    #Supression of discontinuities and transients with horizontal median filter
    chroma_smooth = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))

    
    csv_file_path = os.path.join(not_prog_chroma_dir, filename[0:filename.find(".mp3")] + ".csv")
    with open(csv_file_path, 'w+', newline = '') as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerows(chroma_smooth)
        
for filename in os.listdir(prog_directory):
    print("Reading_song: ", filename)
    y, sr = librosa.load(os.path.join(prog_directory,filename))
    #Isolating harmonic component
    y_harm = librosa.effects.harmonic(y=y, margin=8)
    chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    #Non-local filtering to remove sparse additive noise
    chroma_filter = np.minimum(chroma_harm,
                           librosa.decompose.nn_filter(chroma_harm,
                                                       aggregate=np.median,
                                                       metric='cosine'))
    #Supression of discontinuities and transients with horizontal median filter
    chroma_smooth = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))
    
    csv_file_path = os.path.join(prog_chroma_dir, filename[0:filename.find(".mp3")] + ".csv")
    with open(csv_file_path, 'w+', newline = '') as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerows(chroma_smooth)
        