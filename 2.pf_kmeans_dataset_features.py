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

#Importar dataset a dataframe Pandas
df_features = pd.read_csv(dir_proyecto+'genres_feat_extracted_raw.csv',delimiter=',', index_col=0)

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

#Debido a los pobres resultados de las predicciones de redes convolucionales con
#imágenes de espectrogramas (test accuracy de 45% a 53%) y teniendo en cuenta que solo hay 100 muestras
#por clase para 10 clases diferentes, vamos a agrupar clases utilizando K-Means.

#Para analizarlo visualmente, hacemos un plot bivariante donde podemos observar como es difícil 
#discriminar clases. Géneros musicales muy diferentes como el clásico y el metal se discriminan
#bien pero otros, como por ejemplo el reggae, tienen una dispersión mayor, haciendo mas difícil 
#su clasificación
plt.figure(figsize=(8,8))
sns.scatterplot(df_features["chroma_cqt_mean"],df_features["spectral_contrast_mean"],hue=df_features["label"])

#Para obtener el número de clases que mejor podemos diferenciar con K-Means, utilizamos los valores
#medios de los features extraídos de las muestras de música

#Eliminamos las columnas de standard deviation y median del dataframe
Lista_columnas=df_features.columns.to_list()

for i in Lista_columnas:
    print(i)
    if "std" in i:
        print(i)
        df_features.drop(columns=i, inplace=True)
    elif "median" in i:
        df_features.drop(columns=i, inplace=True)

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

#Calculamos las inercias del K-Means para 12 clusters       
inercias=[]
for k in range(1,12):
    kmeans=KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_Kmeans_sc_df)
    inercias.append(kmeans.inertia_)
    
#Hacemos el dibujo del codo para encontrar el número de clusters (clases)
#que mejor diferencia nuestros datos
plt.plot(range(1,12),inercias)
plt.title('El método del codo')
plt.xlabel('K')
plt.ylabel('Inercia')
plt.annotate('Codo',xy=(3,27000), xytext=(4,27000), arrowprops={'facecolor':'red'})
#La pendiente de la inercia versus el número de clusters cambia drasticamente (disminuye)
#en k=3. Utilizamos entonces este número para definir, ajustar y predecir con K-Means

kmeans = KMeans(n_clusters=3,n_init=10, random_state=42) #Defino
kmeans.fit(X_Kmeans_sc_df) # Ajusto
y_pred=pd.Series(kmeans.predict(X_Kmeans_sc_df)) # Predigo
kmeans.inertia_

#Comparamos nuestros 10 géneros musicales con las clases predichas
df_group=pd.DataFrame([target,y_pred]).T
df_group.rename(columns= {"Unnamed 0":"cluster"}, inplace = True)
df_group.groupby("label")["cluster"].describe()

#Se establece la relación clusters-generos. Se agrupan los géneros
cluster0=["hiphop", "pop", "reggae"]
cluster1=["blues", "classical", "country", "jazz"]
cluster2=["disco", "metal", "rock"]

#Se añade la columna target pero con géneros agrupados
X_Kmeans_sc_df["Tipo de música"] = target
X_Kmeans_sc_df["Tipo de música"] = X_Kmeans_sc_df["Tipo de música"].replace(cluster0, "Música popular")   
X_Kmeans_sc_df["Tipo de música"] = X_Kmeans_sc_df["Tipo de música"].replace(cluster1, "Música melódica")       
X_Kmeans_sc_df["Tipo de música"] = X_Kmeans_sc_df["Tipo de música"].replace(cluster2, "Música rítmica")


#Hacemos el scatter plot de las mismas variables ("chroma_cqt_mean" y "spectral_contrast_mean") pero
#ahora con los géneros agrupados
plt.figure(figsize=(8,8))
sns.scatterplot(X_Kmeans_sc_df.iloc[:,4],X_Kmeans_sc_df.iloc[:,10],hue=X_Kmeans_sc_df["Tipo de música"])
#añado los centroides
centroides = kmeans.cluster_centers_
sns.scatterplot(centroides[:,4],centroides[:,10], color='red')
#Se diferencian mejor las clases agrupando los 10 géneros en 3 clusters

#Ahora añadimos al dataframe sin escalar, la columna de clases agrupadas por tipo de música
#del análisis del K-Means
df_features["Tipo de música"]=X_Kmeans_sc_df["Tipo de música"]
df_features["Género"]=target
df_features["Canción"]=df_features["song_name"]

df_features.drop(columns=["song_name", "label"], inplace=True)

#Exportamos dataframe a formato .csv        
df_features.to_csv(dir_proyecto+'genres_feat_extracted_kmeans.csv')  

