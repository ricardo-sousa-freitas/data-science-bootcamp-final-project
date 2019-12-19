# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:52:42 2019

@author: ricar
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from keras.preprocessing import image
from keras.models import load_model
import librosa
import librosa.display
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


#Ruta carpeta proyecto
dir_proyecto = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/"
os.chdir(dir_proyecto)
os.listdir()

#Importar dataset a dataframe pandas
df_features = pd.read_csv(dir_proyecto+'genres_feat_extracted_kmeans.csv',delimiter=',', index_col=0)

#Ver información, inicio, fin y descripción del dataframe
df_features.info()
df_features.head()
df_features.tail()
df_features.describe()

#Ver tamaño, dimensiones dataframe y Nan's
df_features.size
df_features.shape

df_features.isna().sum()
df_features.isnull().sum()

#Tenemos la columna "Tipo de Música" que agrupa las 10 clases originales en 3 clusters del estudio del K-Means
#Con 3 clases en lugar de 10, podremos hacer un mejor análisis de las variables y eliminar aquellas que peor
#diferencian las clases.

 
#VISUALIZACION DE LOS DATOS

#Hacemos pairplots para ver que variables podrían diferenciar mejor dichas clases
sns.pairplot(df_features,vars=df_features.columns[0:20], hue="Tipo de música")  #primeras 20 variables
sns.pairplot(df_features,vars=df_features.columns[20:41], hue="Tipo de música") #últimas 21 variables (Mel-frequency cepstral coefficients)

features=df_features.columns.to_list()

#Boxplots de las 41 variables, para los tres tipos de música
for i in range(len(features)-3):
    print(i)
    sns.boxplot(x=features[i], y="Tipo de música", data=df_features)
    plt.show()
    
#Histogramas de las 41 variables
for i in range(len(features)-3):
    print(i)
    sns.distplot(df_features.iloc[:,i], bins=10,kde=True,rug=True)
    plt.show()

#Distribución de las 41 variables diferenciando tipo de música
musica_popular = df_features.loc[df_features["Tipo de música"] == "Música popular"]
musica_melodica = df_features.loc[df_features["Tipo de música"] == "Música melódica"]
musica_ritmica = df_features.loc[df_features["Tipo de música"] == "Música rítmica"]

for i in range(len(features)-3):
    print(i)
    sns.kdeplot(musica_popular.iloc[:,i], shade=True, label="Música popular")
    sns.kdeplot(musica_melodica.iloc[:,i], shade=True, label="Música melódica")
    sns.kdeplot(musica_ritmica.iloc[:,i], shade=True, label="Música rítmica")
    plt.xlabel(features[i])
    plt.show()


#LIMPIEZA DE VARIABLES

#Primero vamos a analizar las variables altamente correlacionadas
    
#Matriz de correlación
heatmap = (df_features.corr())
plt.figure(figsize=(35,35))
fig=sns.heatmap(heatmap, annot=True)      
figure = fig.get_figure()
figure.savefig("heatmap_before.png")

#Hacemos las lista de variables que se correlacionan por encima del 90%    
Columnas_altacorrelacion=[]
for i in heatmap:
    for j in heatmap:
        if (heatmap[i][j]>0.9) & (heatmap[i][j]<1):
            Columnas_altacorrelacion.append([i,j])


#Hacemos la visualización conjunta de distribuciones y boxplots de las variables muy correlacionadas
#para analizar que variables eliminar en función de como separan los tipos de música y el número de
#outliers. También se tiene en consideración la importancia del feature para describir la canción
for i in range(len(Columnas_altacorrelacion)):
    print(i)
    sns.kdeplot(musica_popular[Columnas_altacorrelacion[i][0]], shade=True, label="Música popular")
    sns.kdeplot(musica_melodica[Columnas_altacorrelacion[i][0]], shade=True, label="Música melódica")
    sns.kdeplot(musica_ritmica[Columnas_altacorrelacion[i][0]], shade=True, label="Música rítmica")
    plt.xlabel(Columnas_altacorrelacion[i][0])
    plt.show()
    sns.boxplot(x=Columnas_altacorrelacion[i][0], y="Tipo de música", data=df_features)
    plt.show() 

#Variables altamente correlacionadas a eliminar:
#total_beats: peor separación de clases y mas outliers
#chroma_cens_mean: peor separación de clases y mas outliers
#melspectrogram_mean: peor separación de clases, mas outliers e información de escala Mel ya en otras variables
#poly_features_mean: peor separación de clases y mas outliers
#spectral_rolloff_mean: peor separación de clases y altamente correlacionado con spectral_centroid_mean y spectral_bandwidth_mean
    
#spectral_centroid_mean y spectral_bandwidth_mean a pesar de estar correlacionadas al 91 %, no se eliminan ya
#que son variables importantes para la descripción de audio
    
df_features.drop(columns=["total_beats", "chroma_cens_mean", "melspectrogram_mean", "poly_features_mean", "spectral_rolloff_mean"], inplace=True)       

#Matriz de correlación después de limpieza
heatmap_after = (df_features.corr())
plt.figure(figsize=(35,35))
fig=sns.heatmap(heatmap_after, annot=True)      
figure = fig.get_figure()
figure.savefig("heatmap_after.png")

#Actualizamos datos
features=df_features.columns.to_list()

musica_popular = df_features.loc[df_features["Tipo de música"] == "Música popular"]
musica_melodica = df_features.loc[df_features["Tipo de música"] == "Música melódica"]
musica_ritmica = df_features.loc[df_features["Tipo de música"] == "Música rítmica"]


#Vemos otras variables que podrían aportar poco en la discriminación de tipos de música
for i in range(len(features)-3):
    print(i)
    sns.kdeplot(musica_popular.iloc[:,i], shade=True, label="Música popular")
    sns.kdeplot(musica_melodica.iloc[:,i], shade=True, label="Música melódica")
    sns.kdeplot(musica_ritmica.iloc[:,i], shade=True, label="Música rítmica")
    plt.xlabel(features[i])
    plt.show()
#Y las eliminamos    
df_features.drop(columns=["average_beats", "spectral_flatness_mean", "harmonic_mean", "percussive_mean"], inplace=True)    



#Exportamos dataframe a formato .csv        
df_features.to_csv(dir_proyecto+'genres_feat_extracted_cleaned.csv')
