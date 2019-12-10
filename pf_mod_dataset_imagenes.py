# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:54:56 2019

@author: ricar
"""

import numpy as np
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from keras.preprocessing import image
from keras.models import load_model
import librosa
import librosa.display
import pandas as pd


#RUTA CARPETA PROYECTO
dir_proyecto = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/"
os.chdir(dir_proyecto)
os.listdir()

#RUTA MUESTRAS DE AUDIO POR GENERO. 100 AUDIOS .wav POR GENERO PARA 10 GENEROS MUSICALES
audio_path = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/Datasets/gtzan-genres/audios/"

os.chdir(audio_path)
os.listdir()

#LISTA DE LAS 10 CLASES
clases=[]
for i in os.listdir():
    clases.append(i)

#RUTA MUESTRAS DE AUDIO POR GENERO. 100 AUDIOS .wav POR GENERO PARA 10 GENEROS MUSICALES
audio_path = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/Datasets/gtzan-genres/audios/"

#RUTA PARA IMAGENES DE ESPECTROGRAMAS POR GENERO. 100 IMAGENES .png POR GENERO PARA 10 GENEROS MUSICALES
spectrogram_path = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/Datasets/gtzan-genres/espectrogramas/"


#DICCIONARIO DE CLASES PARA IMAGENES
dic_clases_images=dict.fromkeys(clases)

#SE RELLENA EL DICCIONARIO CON LOS NOMBRES DE LOS ARCHIVOS DE IMAGENES (ESPECTROGRAMAS) PARA CADA CLAVE (CLASE)
for i, j in enumerate(clases):
    dic_clases_images[j] = [f for f in listdir(spectrogram_path+clases[i]) if isfile(join(spectrogram_path+clases[i], f))]

#LISTA DE LOS 1000 ARCHIVOS DE IMAGENES    
onlyfiles_images=[]   
for i in dic_clases_images:
    onlyfile=dic_clases_images[i]
    onlyfiles_images=onlyfiles_images+onlyfile

#--------------------------------------------------------------------------------------------------------------------------------------------------# 

#BUCLE PARA CARGAR LAS IMAGENES DE LOS ESPECTROGRAMAS DE MEL CON SUS RESPECTIVAS ETIQUETAS
        
all_images=[]
all_labels_images=[]
for i in onlyfiles_images:
    print("Cargando imagen espectrograma de mel de: ", i)
    if "blues" in i:
        img=image.load_img(spectrogram_path+"blues"+"/"+i, target_size=(128,332,3), grayscale=False)
        img = image.img_to_array(img)
        img = img/255
        all_images.append(img)
        all_labels_images.append("blues")
    elif "classical" in i:
        img=image.load_img(spectrogram_path+"classical"+"/"+i, target_size=(128,332,3), grayscale=False)
        img = image.img_to_array(img)
        img = img/255
        all_images.append(img)
        all_labels_images.append("classical")
    elif "country" in i:
        img=image.load_img(spectrogram_path+"country"+"/"+i, target_size=(128,332,3), grayscale=False)
        img = image.img_to_array(img)
        img = img/255
        all_images.append(img)
        all_labels_images.append("country")
    elif "disco" in i:
        img=image.load_img(spectrogram_path+"disco"+"/"+i, target_size=(128,332,3), grayscale=False)
        img = image.img_to_array(img)
        img = img/255
        all_images.append(img)
        all_labels_images.append("disco")
    elif "hiphop" in i:
        img=image.load_img(spectrogram_path+"hiphop"+"/"+i, target_size=(128,332,3), grayscale=False)
        img = image.img_to_array(img)
        img = img/255
        all_images.append(img)
        all_labels_images.append("hiphop")
    elif "jazz" in i:
        img=image.load_img(spectrogram_path+"jazz"+"/"+i, target_size=(128,332,3), grayscale=False)
        img = image.img_to_array(img)
        img = img/255
        all_images.append(img)
        all_labels_images.append("jazz")
    elif "metal" in i:
        img=image.load_img(spectrogram_path+"metal"+"/"+i, target_size=(128,332,3), grayscale=False)
        img = image.img_to_array(img)
        img = img/255
        all_images.append(img)
        all_labels_images.append("metal")
    elif "pop" in i:
        img=image.load_img(spectrogram_path+"pop"+"/"+i, target_size=(128,332,3), grayscale=False)
        img = image.img_to_array(img)
        img = img/255
        all_images.append(img)
        all_labels_images.append("pop")
    elif "reggae" in i:
        img=image.load_img(spectrogram_path+"reggae"+"/"+i, target_size=(128,332,3), grayscale=False)
        img = image.img_to_array(img)
        img = img/255
        all_images.append(img)
        all_labels_images.append("reggae")
    else:
        img=image.load_img(spectrogram_path+"rock"+"/"+i, target_size=(128,332,3), grayscale=False)
        img = image.img_to_array(img)
        img = img/255
        all_images.append(img)
        all_labels_images.append("rock")


#BUCLE PARA MOSTRAR LOS ESPECTROGRMAS DE MEL DE 50 CANCIONES ALEATORIAS

for i in (np.random.randint(1,1000,50)):
    plt.figure(figsize=(28,10))
    plt.imshow(all_images[i], cmap=plt.cm.binary)
    print("Genero musical: ", all_labels_images[i])
    print("Espectrograma de mel: ", onlyfiles_images[i])
    plt.show()      
