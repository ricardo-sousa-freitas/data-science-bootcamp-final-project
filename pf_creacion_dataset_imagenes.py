# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:01:19 2019

@author: ricar
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import librosa
import librosa.display
import pandas as pd
from os import mkdir


#Ruta carpeta proyecto
dir_proyecto = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/"
os.chdir(dir_proyecto)
os.listdir()

#Ruta muestras de audio por genero. 100 audios .wav por genero para 10 generos musicales
audio_path = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/Datasets/gtzan-genres/audios/"

os.chdir(audio_path)
os.listdir()

#Lista de las 10 clases
clases=[]
for i in os.listdir():
    clases.append(i)


#--------------------------------------------------------------------------------------------------------------------------------------------------#

#CREACION DATASET DE ESPECTROGRAMAS (IMAGENES)

#Ruta muestras de audio por genero. 100 audios .wav por genero para 10 generos musicales
audio_path = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/Datasets/gtzan-genres/audios/"

#Ruta para imagenes de espectrogramas por genero. 100 imagenes .png por genero para 10 generos musicales
spectrogram_path = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/Datasets/gtzan-genres/espectrogramas/"

os.chdir(audio_path)
os.listdir()

#Bucle para cargar los audios, generar su espectrograma de mel y guardar las imagenes en
#las carpetas correspondientes
count=0
for genre in clases:
    os.mkdir(spectrogram_path+genre)
    for file in os.listdir(audio_path+genre):
        song=audio_path+genre+"/"+file
        y, sr = librosa.load(song, mono=True, duration=30)
        #y = librosa.stft(y)
        #Xdb = librosa.amplitude_to_db(abs(y), ref=np.max)
        fig=plt.figure(figsize=(50,5))
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        filename = (audio_path+genre+"/"+file).replace(".wav", ".png").split("/")[-1]
        melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        librosa.display.specshow(librosa.power_to_db(melspectrogram, ref=np.max))
        #librosa.display.specshow(Xdb, cmap='RdBu_r') 
        plt.savefig(spectrogram_path+genre+"/"+filename, dpi=400, bbox_inches="tight", pad_inches=0)
        plt.close("all")
        count=count+1
        print("Espectrograma del genero: ", genre)
        print("Generando espectrograma ", count, "de 1000")



#y, sr = librosa.load("C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/Datasets/gtzan-genres/audios/blues/blues.00000.wav", mono=True, duration=30)
#y = librosa.stft(y)
#Xdb = librosa.amplitude_to_db(abs(y), ref=np.max)
#fig=plt.figure(figsize=(50,5))
#ax = fig.add_subplot(111)
#ax.axes.get_xaxis().set_visible(False)
#ax.axes.get_yaxis().set_visible(False)
#ax.set_frame_on(False)
##filename = (audio_path+genre+"/"+file).replace(".wav", ".png").split("/")[-1]
#melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
#librosa.display.specshow(librosa.power_to_db(melspectrogram, ref=np.max))
#fig=plt.figure(figsize=(50,10))
#y, sr = librosa.load("C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/Datasets/gtzan-genres/audios/metal/metal.00056.wav", mono=True, duration=30)
#melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
#librosa.display.specshow(librosa.power_to_db(melspectrogram, ref=np.max),  x_axis='time', y_axis='hz')