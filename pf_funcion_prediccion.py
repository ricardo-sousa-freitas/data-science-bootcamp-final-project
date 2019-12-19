# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:42:13 2019

@author: ricar
"""
from __future__ import unicode_literals
import numpy as np
import os
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import librosa
import librosa.display
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from keras.models import load_model
import matplotlib.pyplot as plt
import librosa.display
import pickle
import youtube_dl


def predict_musical_genre(url):
    #Ruta muestras de audio. 100 audios .wav por género para 10 géneros musicales
    audio_path = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/Datasets/gtzan-genres/audios/"
    
    os.chdir(audio_path)
    os.listdir()
    
    #Lista de las 10 clases
    clases=[]
    for i in os.listdir():
        clases.append(i)
    
    dir_proyecto = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/"
    os.chdir(dir_proyecto)
    os.listdir()
    #Download Audio Files from Youtube
    
    #url_youtube_song=input("Introduce la dirección url de Youtube de la canción cuyo género deseas saber: ")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
        
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        result = ydl.extract_info(url, download=False)
        video = result['entries'][0] if 'entries' in result else result
        titulo=video['title']
        idv=video['id']
        song_dir=dir_proyecto+titulo+"-"+idv+".wav"
    
    
    song=song_dir
    y, sr = librosa.load(song, mono=True)
    stft = librosa.stft(y)
    stft_db = librosa.amplitude_to_db(abs(stft))
    spectogram=np.abs(librosa.stft(y))
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    rms = librosa.feature.rms(S=spectogram)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    fourier_tempogram = librosa.feature.fourier_tempogram(y=y, sr=sr)
    
    header_list=['tempo', 'chroma_stft_mean', 'chroma_cqt_mean', 'rms_mean',
           'spectral_centroid_mean', 'spectral_bandwidth_mean',
           'spectral_contrast_mean', 'tonnetz_mean', 'zero_crossing_rate_mean',
           'tempogram_mean', 'fourier_tempogram_mean', 'mfcc_mean', 'mfcc1_mean',
           'mfcc2_mean', 'mfcc3_mean', 'mfcc4_mean', 'mfcc5_mean', 'mfcc6_mean',
           'mfcc7_mean', 'mfcc8_mean', 'mfcc9_mean', 'mfcc10_mean', 'mfcc11_mean',
           'mfcc12_mean', 'mfcc13_mean', 'mfcc14_mean', 'mfcc15_mean',
           'mfcc16_mean', 'mfcc17_mean', 'mfcc18_mean', 'mfcc19_mean',
           'mfcc20_mean']
    
    predict_set = pd.DataFrame(np.nan, index=range(1), columns=header_list)
    id=0
    predict_set.loc[id, predict_set.columns[0]]=tempo
    predict_set.loc[id, predict_set.columns[1]]=np.mean(chroma_stft)
    predict_set.loc[id, predict_set.columns[2]]=np.mean(chroma_cqt)
    predict_set.loc[id, predict_set.columns[3]]=np.mean(rms)
    predict_set.loc[id, predict_set.columns[4]]=np.mean(spectral_centroid)
    predict_set.loc[id, predict_set.columns[5]]=np.mean(spectral_bandwidth)
    predict_set.loc[id, predict_set.columns[6]]=np.mean(spectral_contrast)
    predict_set.loc[id, predict_set.columns[7]]=np.mean(tonnetz)
    predict_set.loc[id, predict_set.columns[8]]=np.mean(zero_crossing_rate)
    predict_set.loc[id, predict_set.columns[9]]=np.mean(tempogram)
    predict_set.loc[id, predict_set.columns[10]]=np.mean(np.abs(fourier_tempogram))
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    predict_set.loc[id, predict_set.columns[11]]=np.mean(mfcc)
    
    col=12
    for number, mfccs in enumerate(mfcc):
        predict_set.loc[id, predict_set.columns[col+number]]=np.mean(mfccs)
        
    X_test= np.array(predict_set)   
    
    modelf=load_model("C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/best_model_feat_10classes.h5")
    scaler_train_10 = pickle.load( open( "scaler_train_10.p", "rb" ) )
    X_test=scaler_train_10.transform(X_test)
    prediccion=modelf.predict_classes(X_test)[0]
    dic_clases={}
    for i , j in enumerate(clases):
        dic_clases[i]=j
    genero=dic_clases[prediccion]
    print("El género de la cancion ", titulo +" es: ", genero)
    return genero, titulo

