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
import csv
from os import mkdir
#import pf_creacion_dataset_features as dsf


#RUTA CARPETA PROYECTO
dir_proyecto = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/"
os.chdir(dir_proyecto)
os.listdir()

#RUTA MUESTRAS DE AUDIO POR GENERO. 100 AUDIOS .WAV POR GENERO PARA 10 GENEROS MUSICALES
audio_path = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/Datasets/gtzan-genres/audios/"

os.chdir(audio_path)
os.listdir()

#LISTA DE LAS 10 CLASES
clases=[]
for i in os.listdir():
    clases.append(i)


#--------------------------------------------------------------------------------------------------------------------------------------------------#

#CREACION DATASET DE ESPECTROGRAMAS (IMAGENES)

#RUTA MUESTRAS DE AUDIO POR GENERO. 100 AUDIOS .wav POR GENERO PARA 10 GENEROS MUSICALES
audio_path = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/Datasets/gtzan-genres/audios/"

#RUTA PARA IMAGENES DE ESPECTROGRAMAS POR GENERO. 100 IMAGENES .png POR GENERO PARA 10 GENEROS MUSICALES
spectrogram_path = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/Datasets/gtzan-genres/espectrogramas/"

os.chdir(audio_path)
os.listdir()

#BUCLE PARA CARGAR LOS AUDIOS, GENERAR SU ESPECTROGRAMA DE MEL Y GUARDAR LAS IMAGENES EN LAS CARPETAS CORRESPONDIENTES

for genre in clases:
    os.mkdir(spectrogram_path+genre)
    for file in os.listdir(audio_path+genre):
        song=audio_path+genre+"/"+file
        y, sr = librosa.load(song, mono=True, duration=30)
        fig=plt.figure(figsize=(14,5))
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        filename = (audio_path+genre+"/"+file).replace(".wav", ".png").split("/")[-1]
        melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        librosa.display.specshow(librosa.power_to_db(melspectrogram, ref=np.max))
        plt.savefig(spectrogram_path+genre+"/"+filename, dpi=400, bbox_inches="tight", pad_inches=0)
        plt.close("all")

