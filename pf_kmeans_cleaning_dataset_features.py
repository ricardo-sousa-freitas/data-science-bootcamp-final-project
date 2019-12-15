# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:50:19 2019

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
df_features = pd.read_csv(dir_proyecto+'genres_feat_extracted_raw.csv',delimiter=',', index_col=0)

#Ver informacion, inicio y fin dataframe
df_features.info()
df_features.head()
df_features.tail()
df_features.describe()

#Ver tamaño, dimensiones dataframe y nan's
df_features.size
df_features.shape

df_features.isna().sum()
df_features.isnull().sum()

#Debido a los pobres resultados de las predicciones de redes convolucionales con
#imagenes de espectrogramas (test accuracy de 45% a 53%) y teniendo en cuenta que solo hay 100 muestras
#por clase para 10 clases diferentes, vamos a agrupar clases utilizando K-Means.

#Para analizarlo visualmente, hacemos un plot bivariante donde podemos observar como es dificil 
#discriminar clases. Generos musicales muy diferentes como el clasico y el metal se discriminan
#bien pero otros, como por ejemplo el reggae, tienen una dispersion mayor, haciendo mas dificil 
#su clasificacion
plt.figure(figsize=(8,8))
sns.scatterplot(df_features["chroma_cqt_mean"],df_features["spectral_contrast_mean"],hue=df_features["label"])

#Para obtener el numero de clases que mejor podemos diferenciar con K-Means, utilizamos los valores
#medios de los features extraidos de las muestras de musica

#Escalamos el dataframe
X=df_features.copy()
target=df_features["label"]  #guardamos el target
X.drop(columns=["song_name", "label"], inplace=True) #eliminamos las columnas categoricas
Lista_columnas=X.columns.to_list()
scaler=StandardScaler().fit(X)
X_Kmeans_sc=scaler.transform(X)
X_Kmeans_sc_df=pd.DataFrame(X_Kmeans_sc, columns=Lista_columnas)

#Verificamos el escalamiento
np.mean(X_Kmeans_sc_df)
np.var(X_Kmeans_sc_df)

#Eliminamos las columnas de standard deviation y median del dataframe escalado
for i in Lista_columnas:
    print(i)
    if "std" in i:
        print(i)
        X_Kmeans_sc_df.drop(columns=i, inplace=True)
    elif "median" in i:
        X_Kmeans_sc_df.drop(columns=i, inplace=True)


#Calculamos las inercias del K-Means para 12 clusters       
inercias=[]
for k in range(1,12):
    kmeans=KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_Kmeans_sc_df)
    inercias.append(kmeans.inertia_)
    
#Hacemos el dibujo del codo para encontrar el numero de clusters (clases)
#que mejor diferencia nuestros datos
plt.plot(range(1,12),inercias)
plt.title('El método del codo')
plt.xlabel('K')
plt.ylabel('Inercia')
plt.annotate('Codo',xy=(3,27000), xytext=(4,27000), arrowprops={'facecolor':'red'})
#La pendiente de la inercia versus el numero de clusters cambia drasticamente (disminuye)
#en k=3. Utilizamos entonces este numero para definir, ajustar y predecir con K-Means

kmeans = KMeans(n_clusters=3,n_init=10, random_state=42) #Defino
kmeans.fit(X_Kmeans_sc_df) # Ajusto
y_pred=pd.Series(kmeans.predict(X_Kmeans_sc_df)) # Predigo
kmeans.inertia_

#Comparamos nuestros 10 generos musicales con las clases predichas
df_group=pd.DataFrame([target,y_pred]).T
df_group.rename(columns= {"Unnamed 0":"cluster"}, inplace = True)
df_group.groupby("label")["cluster"].describe()

#Se establece la relacion clusters-generos. Se agrupan los generos
cluster0=["hiphop", "pop", "reggae"]
cluster1=["blues", "classical", "country", "jazz"]
cluster2=["disco", "metal", "rock"]

#Se añade la columna target pero con generos agrupados
X_Kmeans_sc_df["Tipo de musica"] = target
X_Kmeans_sc_df["Tipo de musica"] = X_Kmeans_sc_df["Tipo de musica"].replace(cluster0, "Musica popular")   
X_Kmeans_sc_df["Tipo de musica"] = X_Kmeans_sc_df["Tipo de musica"].replace(cluster1, "Musica melodica")       
X_Kmeans_sc_df["Tipo de musica"] = X_Kmeans_sc_df["Tipo de musica"].replace(cluster2, "Musica ritmica")


#Hacemos el scatter plot de las mismas variables ("chroma_cqt_mean" y "spectral_contrast_mean") pero
#ahora con los generos agrupados
plt.figure(figsize=(8,8))
sns.scatterplot(X_Kmeans_sc_df.iloc[:,4],X_Kmeans_sc_df.iloc[:,10],hue=X_Kmeans_sc_df["Tipo de musica"])
#añado los centroides
centroides = kmeans.cluster_centers_
sns.scatterplot(centroides[:,4],centroides[:,10], color='red')
#Se diferencian mejor las clases agrupando los 10 generos en 3 clusters
  
features=X_Kmeans_sc_df.columns.to_list()


#VISUALIZACION DE LOS DATOS

#Hacemos pairplots para ver que variables podrian diferenciar mejor dichas clases
sns.pairplot(X_Kmeans_sc_df,vars=X_Kmeans_sc_df.columns[0:20], hue="Tipo de musica")  #primeras 20 variables
sns.pairplot(X_Kmeans_sc_df,vars=X_Kmeans_sc_df.columns[20:41], hue="Tipo de musica") #ultimas 21 variables (Mel-frequency cepstral coefficients)

#Boxplots de las 41 variables, para los tres tipos de musica
for i in range(len(features)-1):
    print(i)
    sns.boxplot(x=features[i], y="Tipo de musica", data=X_Kmeans_sc_df)
    plt.show()
    
#Histogramas de las 41 variables
for i in range(len(features)-1):
    print(i)
    sns.distplot(X_Kmeans_sc_df.iloc[:,i], bins=10,kde=True,rug=True)
    plt.show()

#Distribucion de las 41 variables diferenciando tipo de musica
musica_popular = X_Kmeans_sc_df.loc[X_Kmeans_sc_df["Tipo de musica"] == "Musica popular"]
musica_melodica = X_Kmeans_sc_df.loc[X_Kmeans_sc_df["Tipo de musica"] == "Musica melodica"]
musica_ritmica = X_Kmeans_sc_df.loc[X_Kmeans_sc_df["Tipo de musica"] == "Musica ritmica"]

for i in range(len(features)-1):
    print(i)
    sns.kdeplot(musica_popular.iloc[:,i], shade=True, label="Musica popular")
    sns.kdeplot(musica_melodica.iloc[:,i], shade=True, label="Musica melodica")
    sns.kdeplot(musica_ritmica.iloc[:,i], shade=True, label="Musica ritmica")
    plt.xlabel(features[i])
    plt.show()


#LIMPIEZA DE VARIABLES

#Primero vamos a analizar las variables altamente correlacionadas
    
#Matriz de correlacion
heatmap = (X_Kmeans_sc_df.corr())
plt.figure(figsize=(40,40))
fig=sns.heatmap(heatmap, annot=True)      
figure = fig.get_figure()
figure.savefig("heatmap_before.png")

#Hacemos las lista de variables que se correlacionan por encima del 90%    
Columnas_altacorrelacion=[]
for i in heatmap:
    for j in heatmap:
        if (heatmap[i][j]>0.9) & (heatmap[i][j]<1):
            Columnas_altacorrelacion.append([i,j])


#Hacemos la visualizacion conjunta de distribuciones y boxplots de las variables muy correlacionadas
#para analizar que variables eliminar en funcion de como separan los tipos de musica, el numero de
#outliers. Tambien se tiene en consideracion la importancia del feature para describir la cancion
for i in range(len(Columnas_altacorrelacion)):
    print(i)
    sns.kdeplot(musica_popular[Columnas_altacorrelacion[i][0]], shade=True, label="Musica popular")
    sns.kdeplot(musica_melodica[Columnas_altacorrelacion[i][0]], shade=True, label="Musica melodica")
    sns.kdeplot(musica_ritmica[Columnas_altacorrelacion[i][0]], shade=True, label="Musica ritmica")
    plt.xlabel(Columnas_altacorrelacion[i][0])
    plt.show()
    sns.boxplot(x=Columnas_altacorrelacion[i][0], y="Tipo de musica", data=X_Kmeans_sc_df)
    plt.show() 

#Variables altamente correlacionadas a eliminar:
#total_beats: peor separacion de clases y mas outliers
#chroma_cens_mean: peor separacion de clases y mas outliers
#melspectrogram_mean: peor separacion de clases, mas outliers e informacion de escala Mel ya en otras variables
#poly_features_mean: peor separacion de clases y mas outliers
#spectral_rolloff_mean: peor separacion de clases y altamente correlacionado con spectral_centroid_mean y spectral_bandwidth_mean
    
#spectral_centroid_mean y spectral_bandwidth_mean a pesar de estar correlacionadas al 91 %, no se eliminan ya
#que son variables importantes para la descripcion de audio
    
X_Kmeans_sc_df.drop(columns=["total_beats", "chroma_cens_mean", "melspectrogram_mean", "poly_features_mean", "spectral_rolloff_mean"], inplace=True)       

#Matriz de correlacion despues de limpieza
heatmap_after = (X_Kmeans_sc_df.corr())
plt.figure(figsize=(40,40))
fig=sns.heatmap(heatmap_after, annot=True)      
figure = fig.get_figure()
figure.savefig("heatmap_after.png")

#Actualizamos datos
features=X_Kmeans_sc_df.columns.to_list()

musica_popular = X_Kmeans_sc_df.loc[X_Kmeans_sc_df["Tipo de musica"] == "Musica popular"]
musica_melodica = X_Kmeans_sc_df.loc[X_Kmeans_sc_df["Tipo de musica"] == "Musica melodica"]
musica_ritmica = X_Kmeans_sc_df.loc[X_Kmeans_sc_df["Tipo de musica"] == "Musica ritmica"]


#Vemos otras variables que podrian aportar poco en la discriminacion de tipos de musica
for i in range(len(features)-1):
    print(i)
    sns.kdeplot(musica_popular.iloc[:,i], shade=True, label="Musica popular")
    sns.kdeplot(musica_melodica.iloc[:,i], shade=True, label="Musica melodica")
    sns.kdeplot(musica_ritmica.iloc[:,i], shade=True, label="Musica ritmica")
    plt.xlabel(features[i])
    plt.show()
#Y las eliminamos    
X_Kmeans_sc_df.drop(columns=["average_beats", "spectral_flatness_mean", "harmonic_mean", "percussive_mean"], inplace=True)    


X_Kmeans_sc_df["Genero"]=target
X_Kmeans_sc_df["Cancion"]=df_features["song_name"]

#Exportar dataframe a formato .csv        
X_Kmeans_sc_df.to_csv(dir_proyecto+'genres_feat_extracted_cleaned.csv')
