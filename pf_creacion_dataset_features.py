# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:58:11 2019

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
import tensorflow as tf

mypath=os.getcwd()

#RUTA MUESTRAS DE AUDIO POR GENERO. 100 AUDIOS .WAV POR GENERO PARA 10 GENEROS MUSICALES
audio_path = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/Datasets/gtzan-genres/audios/"

os.chdir(audio_path)
os.listdir()

#LISTA DE LAS 10 CLASES
clases=[]
for i in os.listdir():
    clases.append(i)

#DICCIONARIO DE CLASES
dic_clases=dict.fromkeys(clases)

#SE RELLENA EL DICCIONARIO CON LOS NOMBRES DE LOS ARCHIVOS PARA CADA CLAVE (CLASE)
for i, j in enumerate(clases):
    dic_clases[j] = [f for f in listdir(audio_path+clases[i]) if isfile(join(audio_path+clases[i], f))]

#LISTA DE LOS 1000 ARCHIVOS DE AUDIO    
onlyfiles=[]   
for i in dic_clases:
    onlyfile=dic_clases[i]
    onlyfiles=onlyfiles+onlyfile

#BUCLE PARA CARGAR LOS AUDIOS CON SUS ETIQUETAS CORRESPONDIENTES    
all_amp_time_series=[]
all_sr=[]
all_labels=[]
for i in onlyfiles:
    if "blues" in i:
        x , sr = librosa.load(audio_path+"blues"+"/"+i, mono=True, duration=30)
        all_amp_time_series.append(x)
        all_sr.append(sr)
        all_labels.append("blues")
    elif "classical" in i:
        x , sr = librosa.load(audio_path+"classical"+"/"+i, mono=True, duration=30)
        all_amp_time_series.append(x)
        all_sr.append(sr)
        all_labels.append("classical")
    elif "country" in i:
        x , sr = librosa.load(audio_path+"country"+"/"+i, mono=True, duration=30)
        all_amp_time_series.append(x)
        all_sr.append(sr)
        all_labels.append("country")
    elif "disco" in i:
        x , sr = librosa.load(audio_path+"disco"+"/"+i, mono=True, duration=30)
        all_amp_time_series.append(x)
        all_sr.append(sr)
        all_labels.append("disco")
    elif "hiphop" in i:
        x , sr = librosa.load(audio_path+"hiphop"+"/"+i, mono=True, duration=30)
        all_amp_time_series.append(x)
        all_sr.append(sr)
        all_labels.append("hiphop")
    elif "jazz" in i:
        x , sr = librosa.load(audio_path+"jazz"+"/"+i, mono=True, duration=30)
        all_amp_time_series.append(x)
        all_sr.append(sr)
        all_labels.append("jazz")
    elif "metal" in i:
        x , sr = librosa.load(audio_path+"metal"+"/"+i, mono=True, duration=30)
        all_amp_time_series.append(x)
        all_sr.append(sr)
        all_labels.append("metal")
    elif "pop" in i:
        x , sr = librosa.load(audio_path+"pop"+"/"+i, mono=True, duration=30)
        all_amp_time_series.append(x)
        all_sr.append(sr)
        all_labels.append("pop")
    elif "reggae" in i:
        x , sr = librosa.load(audio_path+"reggae"+"/"+i, mono=True, duration=30)
        all_amp_time_series.append(x)
        all_sr.append(sr)
        all_labels.append("reggae")
    else:
        x , sr = librosa.load(audio_path+"rock"+"/"+i, mono=True, duration=30)
        all_amp_time_series.append(x)
        all_sr.append(sr)
        all_labels.append("rock")    
                  
#BUCLE PARA MOSTRAR LAS FORMAS DE ONDA DE 50 CANCIONES ALEATORIAS                  
for i in (np.random.randint(1,1000,50)):
    plt.figure(figsize=(28,10))
    librosa.display.waveplot(all_amp_time_series[i], sr=sr) 
    print("Genero musical: ", all_labels[i])
    print("Cancion: ", onlyfiles[i])
    plt.show()      
        
#--------------------------------------------------------------------------------------------------------------------------------------------------#

#CREACION DEL DATAFRAME CON FEATURES EXTRAIDOS DE LOS AUDIOS .WAV

    
#SPECTRAL FEATURES DISPONIBLES EN LIBROSA
    
#chroma_stft([y, sr, S, norm, n_fft, …])	Compute a chromagram from a waveform or power spectrogram.
#chroma_cqt([y, sr, C, hop_length, fmin, …])	Constant-Q chromagram
#chroma_cens([y, sr, C, hop_length, fmin, …])	Computes the chroma variant “Chroma Energy Normalized” (CENS), following [R674badebce0d-1].
#melspectrogram([y, sr, S, n_fft, …])	Compute a mel-scaled spectrogram.
#mfcc([y, sr, S, n_mfcc, dct_type, norm, lifter])	Mel-frequency cepstral coefficients (MFCCs)
#rms([y, S, frame_length, hop_length, …])	Compute root-mean-square (RMS) value for each frame, either from the audio samples y or from a spectrogram S.
#spectral_centroid([y, sr, S, n_fft, …])	Compute the spectral centroid.
#spectral_bandwidth([y, sr, S, n_fft, …])	Compute p’th-order spectral bandwidth.
#spectral_contrast([y, sr, S, n_fft, …])	Compute spectral contrast [R6ffcc01153df-1]
#spectral_flatness([y, S, n_fft, hop_length, …])	Compute spectral flatness
#spectral_rolloff([y, sr, S, n_fft, …])	Compute roll-off frequency.
#poly_features([y, sr, S, n_fft, hop_length, …])	Get coefficients of fitting an nth-order polynomial to the columns of a spectrogram.
#tonnetz([y, sr, chroma])	Computes the tonal centroid features (tonnetz), following the method of [Recf246e5a035-1].
#zero_crossing_rate(y[, frame_length, …])	Compute the zero-crossing rate of an audio time series.
    
features=["song_name", "tempo", "total_beats", "average_beats"]

features_mean_std_median= ["chroma_stft", "chroma_cqt", "chroma_cens", "melspectrogram", "rms",
          "spectral_centroid", "spectral_bandwidth", "spectral_contrast",
          "spectral_flatness", "spectral_rolloff", "poly_features",
          "tonnetz", "zero_crossing_rate", "tempogram", "fourier_tempogram",
          "harmonic", "percussive", "mfcc"]


#RHYTHM FEATURES DISPONIBLES EN LIBROSA

#tempogram([y, sr, onset_envelope, …])	Compute the tempogram: local autocorrelation of the onset strength envelope.
#fourier_tempogram([y, sr, onset_envelope, …])	Compute the Fourier tempogram: the short-time Fourier transform of the onset strength envelope.


#BEAT AND TEMPO

#beat_track([y, sr, onset_envelope, …])	Dynamic programming beat tracker.
#plp([y, sr, onset_envelope, hop_length, …])	Predominant local pulse (PLP) estimation.
#tempo([y, sr, onset_envelope, hop_length, …])	Estimate the tempo (beats per minute)

#EFFECTS

#Harmonic-percussive source separation
#hpss(y, \*\*kwargs)	Decompose an audio time series into harmonic and percussive components.
#harmonic(y, \*\*kwargs)	Extract harmonic elements from an audio time-series.
#percussive(y, \*\*kwargs)	Extract percussive elements from an audio time-series.


#FEATURE MANIPULATION

#delta(data[, width, order, axis, mode])	Compute delta features: local estimate of the derivative of the input data along the selected axis.
#stack_memory(data[, n_steps, delay])	Short-term history embedding: vertically concatenate a data vector or matrix with delayed copies of itself.

#FEATURE INVERSION

#inverse.mel_to_stft(M[, sr, n_fft, power])	Approximate STFT magnitude from a Mel power spectrogram.
#inverse.mel_to_audio(M[, sr, n_fft, …])	Invert a mel power spectrogram to audio using Griffin-Lim.
#inverse.mfcc_to_mel(mfcc[, n_mels, …])	Invert Mel-frequency cepstral coefficients to approximate a Mel power spectrogram.
#inverse.mfcc_to_audio(mfcc[, n_mels, …])	Convert Mel-frequency cepstral coefficients to a time-domain audio signal

dir_proyecto=os.getcwd()

#RUTA CARPETA PROYECTO
dir_proyecto = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/"

os.chdir(dir_proyecto)
os.listdir()

#HEADER DEL DATAFRAME
header = ""
for i in range(len(features)):
    header += features[i] + ","
for i, j in enumerate(features_mean_std_median):
    header+= features_mean_std_median[i]+"_mean" + ","
    header+= features_mean_std_median[i]+"_std" + ","
    header+= features_mean_std_median[i]+"_median"+ ","
    if "mfcc"==j:
        for k in range(1, 21):
            header += f'mfcc{k}'+"_mean" + "," 
            header += f'mfcc{k}'+"_std" + "," 
            header += f'mfcc{k}'+"_median"+ "," 
header += "label"
header_list=header.split(",")

#DATAFRAME VACIO
features_set = pd.DataFrame(np.nan, index=range(len(onlyfiles)), columns=header_list)

#EXTRAEMOS LOS FEATURES Y RELLENAMOS EL DATAFRAME    
id=0
for genre in clases:
    for file in os.listdir(audio_path+genre):
        song=audio_path+genre+"/"+file
        y, sr = librosa.load(song, mono=True, duration=30)
        stft = librosa.stft(y)
        stft_db = librosa.amplitude_to_db(abs(stft))
        spectogram=np.abs(librosa.stft(y))
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        rms = librosa.feature.rms(S=spectogram)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        poly_features = librosa.feature.poly_features(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
        tempogram = librosa.feature.tempogram(y=y, sr=sr)
        fourier_tempogram = librosa.feature.fourier_tempogram(y=y, sr=sr)
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        
        features_set.loc[id, features_set.columns[0]]=file
        features_set.loc[id, features_set.columns[1]]=tempo
        features_set.loc[id, features_set.columns[2]]=sum(beats)
        features_set.loc[id, features_set.columns[3]]=np.average(beats)
        features_set.loc[id, features_set.columns[4]]=np.mean(chroma_stft)
        features_set.loc[id, features_set.columns[5]]=np.std(chroma_stft)
        features_set.loc[id, features_set.columns[6]]=np.median(chroma_stft)
        features_set.loc[id, features_set.columns[7]]=np.mean(chroma_cqt)
        features_set.loc[id, features_set.columns[8]]=np.std(chroma_cqt)
        features_set.loc[id, features_set.columns[9]]=np.median(chroma_cqt)
        features_set.loc[id, features_set.columns[10]]=np.mean(chroma_cens)
        features_set.loc[id, features_set.columns[11]]=np.std(chroma_cens)
        features_set.loc[id, features_set.columns[12]]=np.median(chroma_cens)
        features_set.loc[id, features_set.columns[13]]=np.mean(melspectrogram)
        features_set.loc[id, features_set.columns[14]]=np.std(melspectrogram)
        features_set.loc[id, features_set.columns[15]]=np.median(melspectrogram)
        features_set.loc[id, features_set.columns[16]]=np.mean(rms)
        features_set.loc[id, features_set.columns[17]]=np.std(rms)
        features_set.loc[id, features_set.columns[18]]=np.median(rms)
        features_set.loc[id, features_set.columns[19]]=np.mean(spectral_centroid)
        features_set.loc[id, features_set.columns[20]]=np.std(spectral_centroid)
        features_set.loc[id, features_set.columns[21]]=np.median(spectral_centroid) 
        features_set.loc[id, features_set.columns[22]]=np.mean(spectral_bandwidth)
        features_set.loc[id, features_set.columns[23]]=np.std(spectral_bandwidth)
        features_set.loc[id, features_set.columns[24]]=np.median(spectral_bandwidth)       
        features_set.loc[id, features_set.columns[25]]=np.mean(spectral_contrast)
        features_set.loc[id, features_set.columns[26]]=np.std(spectral_contrast)
        features_set.loc[id, features_set.columns[27]]=np.median(spectral_contrast)
        features_set.loc[id, features_set.columns[28]]=np.mean(spectral_flatness)
        features_set.loc[id, features_set.columns[29]]=np.std(spectral_flatness)
        features_set.loc[id, features_set.columns[30]]=np.median(spectral_flatness)  
        features_set.loc[id, features_set.columns[31]]=np.mean(spectral_rolloff)
        features_set.loc[id, features_set.columns[32]]=np.std(spectral_rolloff)
        features_set.loc[id, features_set.columns[33]]=np.median(spectral_rolloff)  
        features_set.loc[id, features_set.columns[34]]=np.mean(poly_features)
        features_set.loc[id, features_set.columns[35]]=np.std(poly_features)
        features_set.loc[id, features_set.columns[36]]=np.median(poly_features)
        features_set.loc[id, features_set.columns[37]]=np.mean(tonnetz)
        features_set.loc[id, features_set.columns[38]]=np.std(tonnetz)
        features_set.loc[id, features_set.columns[39]]=np.median(tonnetz)   
        features_set.loc[id, features_set.columns[40]]=np.mean(zero_crossing_rate)
        features_set.loc[id, features_set.columns[41]]=np.std(zero_crossing_rate)
        features_set.loc[id, features_set.columns[42]]=np.median(zero_crossing_rate)
        features_set.loc[id, features_set.columns[43]]=np.mean(tempogram)
        features_set.loc[id, features_set.columns[44]]=np.std(tempogram)
        features_set.loc[id, features_set.columns[45]]=np.median(tempogram) 
        features_set.loc[id, features_set.columns[46]]=np.mean(np.abs(fourier_tempogram))
        features_set.loc[id, features_set.columns[47]]=np.std(np.abs(fourier_tempogram))
        features_set.loc[id, features_set.columns[48]]=np.median(np.abs(fourier_tempogram))
        features_set.loc[id, features_set.columns[49]]=np.mean(harmonic)
        features_set.loc[id, features_set.columns[50]]=np.std(harmonic)
        features_set.loc[id, features_set.columns[51]]=np.median(harmonic)
        features_set.loc[id, features_set.columns[52]]=np.mean(percussive)
        features_set.loc[id, features_set.columns[53]]=np.std(percussive)
        features_set.loc[id, features_set.columns[54]]=np.median(percussive)
         
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        features_set.loc[id, features_set.columns[55]]=np.mean(mfcc)
        features_set.loc[id, features_set.columns[56]]=np.std(mfcc)
        features_set.loc[id, features_set.columns[57]]=np.median(mfcc)


        col=57
        for number, mfccs in enumerate(mfcc):
            #print(number)
            #print(mfccs)
            features_set.loc[id, features_set.columns[col+1+number]]=np.mean(mfccs)
            features_set.loc[id, features_set.columns[(col+2+number)]]=np.std(mfccs)
            features_set.loc[id, features_set.columns[(col+3+number)]]=np.median(mfccs)
            #print(col+1+number)
            #print(col+2+number)
            #print(col+3+number)
            col=col+2
            #print(col)
            
        features_set.loc[id, features_set.columns[118]]=genre
         
        print("Extrayendo features de: ",file)
        print("Cancion ", id+1, "de ", len(onlyfiles))
        id = id+1              


               
#EXPORTAR DATAFRAME A FORMATO .csv        
features_set.to_csv(dir_proyecto+'genres_feat_extracted.csv')
