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
from keras.models import load_model
import librosa
import librosa.display
import pandas as pd
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras import layers
from keras import models
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import svm
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import keras
from keras import regularizers
from keras import optimizers
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


#Ruta carpeta Proyecto
dir_proyecto = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/"
os.chdir(dir_proyecto)
os.listdir()

from pf_funciones import plot_confusion_matrix  

#Ruta muestras de audio. 100 audios .wav por género para 10 géneros musicales
audio_path = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/Datasets/gtzan-genres/audios/"

os.chdir(audio_path)
os.listdir()

#Lista de las 10 clases
clases=[]
for i in os.listdir():
    clases.append(i)

#Ruta para imágenes de espectrogramas. 100 imágenes .png por género para 10 géneros musicales
spectrogram_path = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/Datasets/gtzan-genres/espectrogramas/"

#Diccionario de clases para imágenes
dic_clases_images=dict.fromkeys(clases)

#Se rellena el diccionario con los nombres de los archivos de imágenes (espectrogramas) para cada clave (clase)
for i, j in enumerate(clases):
    dic_clases_images[j] = [f for f in listdir(spectrogram_path+clases[i]) if isfile(join(spectrogram_path+clases[i], f))]

#Lista de los 1000 archivos de imágenes    
onlyfiles_images=[]   
for i in dic_clases_images:
    onlyfile=dic_clases_images[i]
    onlyfiles_images=onlyfiles_images+onlyfile

#--------------------------------------------------------------------------------------------------------------------------------------------------# 

#Bucle para cargar las imágenes de los espectrogramas de Mel con sus respectivas etiquetas
        
all_images=[]
all_labels_images=[]
for i in onlyfiles_images:
    print("Cargando imágen espectrograma de: ", i)
    img=image.load_img(spectrogram_path+i.split(".")[0]+"/"+i, target_size=(100,800,3), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    all_images.append(img)
    all_labels_images.append(i.split(".")[0])
    

#Bucle para mostrar los espectrogramas de 50 canciones aleatorias
for i in (np.random.randint(1,1000,50)):
    plt.figure(figsize=(28,10))
    plt.imshow(all_images[i], cmap=plt.cm.binary)
    print("Género: ", list(all_labels_images)[i])
    print("Espectrograma de Mel: ", onlyfiles_images[i])
    plt.show()      

#--------------------------------------------------------------------------------------------------------------------------------------------------# 

#Hacemos una Red Neuronal Convolucional de prueba para predecir para los 10 géneros
    
#Proporción 70% train, 15% validation, 15% test         
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels_images, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

X_train = np.array(X_train)
X_test= np.array(X_test)
X_val= np.array(X_val)
y_train=np.asarray(pd.get_dummies(y_train))
y_val=np.asarray(pd.get_dummies(y_val))
y_test=np.asarray(pd.get_dummies(y_test))

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(100, 800, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

history = model.fit(X_train,
                    y_train,
                    epochs=40,
                    batch_size=150, 
                    validation_data=(X_val, y_val))

pred_mejor_prueba_10=model.predict(X_test)
pred_clases_mejor_prueba_10=model.predict_classes(X_test)
test_loss_trained_net, test_acc_trained_net = model.evaluate(X_test, y_test)
print('test_acc:', test_acc_trained_net)
#Resultado entre el 45% y el 53% de acuraccy sobre el test

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc=history.history['acc']
epochs = range(1, len(history.epoch) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()        

model.save('test_best_imag_10classes.h5')

#--------------------------------------------------------------------------------------------------------------------------------------------------# 

#Repetimos la arquitectura e hiperparametros de la Red Neuronal Convolucional de prueba anterior pero ahora
#para los géneros agrupados en 3 clusters del análisis del K-Means

#Clases del K-Means
cluster0=["hiphop", "pop", "reggae"]
cluster1=["blues", "classical", "country", "jazz"]
cluster2=["disco", "metal", "rock"]
    
all_labels_images_grouped=pd.Series(all_labels_images)    
all_labels_images_grouped = all_labels_images_grouped.replace(cluster0, "Musica popular")   
all_labels_images_grouped = all_labels_images_grouped.replace(cluster1, "Musica melodica")       
all_labels_images_grouped = all_labels_images_grouped.replace(cluster2, "Musica ritmica")  


#Proporción 70% train, 15% validation, 15% test         
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels_images_grouped, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

X_train = np.array(X_train)
X_test= np.array(X_test)
X_val= np.array(X_val)
y_train=np.asarray(pd.get_dummies(y_train))
y_val=np.asarray(pd.get_dummies(y_val))
y_test=np.asarray(pd.get_dummies(y_test))

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(100, 800, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

history = model.fit(X_train,
                    y_train,
                    epochs=40,
                    batch_size=150, 
                    validation_data=(X_val, y_val))

pred_mejor_prueba_3=model.predict(X_test)
pred_clases_mejor_prueba_3=model.predict_classes(X_test)
test_loss_trained_net, test_acc_trained_net = model.evaluate(X_test, y_test)
print('test_acc:', test_acc_trained_net)
#Resultado entre el 75% y el 79% de acuraccy sobre el test. Agrupando en 3 clases el porcentaje de acierto sube en un 25%

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc=history.history['acc']
epochs = range(1, len(history.epoch) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()        

model.save('test_best_imag_3classes.h5')

#--------------------------------------------------------------------------------------------------------------------------------------------------# 
#Ahora usando GridSearchCV, repetimos la Red Neuronal Convolucional cambiando la arquitectura e hiperparametros
#para los géneros agrupados en 3 clusters del análisis del K-Means

#Clases del K-Means
cluster0=["hiphop", "pop", "reggae"]
cluster1=["blues", "classical", "country", "jazz"]
cluster2=["disco", "metal", "rock"]
    
all_labels_images_grouped=pd.Series(all_labels_images)    
all_labels_images_grouped = all_labels_images_grouped.replace(cluster0, "Musica popular")   
all_labels_images_grouped = all_labels_images_grouped.replace(cluster1, "Musica melodica")       
all_labels_images_grouped = all_labels_images_grouped.replace(cluster2, "Musica ritmica")  


#Proporción 70% train, 15% validation, 15% test         
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels_images_grouped, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

X_train = np.asarray(X_train)
X_test= np.asarray(X_test)
X_val= np.asarray(X_val)
y_train=np.asarray(pd.get_dummies(y_train))
y_val=np.asarray(pd.get_dummies(y_val))
y_test=np.asarray(pd.get_dummies(y_test))


def conv_model(nro_capas, filters,neurons, dropout):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(100, 800, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    for i in range(nro_capas):
        print(i)
        model.add(layers.Conv2D(filtros[i], (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(neurons, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.summary()
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=conv_model)
nro_capas=[2,3]
filtros=[64,128,128]
neuronas=[128,512]
drop=[0.3,0.7]
param_grid = dict(nro_capas=nro_capas,filters=filtros, neurons=neuronas,dropout=drop)
grid = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    verbose = 2,
                    cv=2)

grid_results=grid.fit(X_train,y_train, epochs=5, batch_size=150, verbose=2, validation_data=(X_val,y_val))
#Debido a las limitaciones de memoria del ordenador, los parámetros del grid tuvieron que limitarse.
#Sus resultados se utilizaron solo de referencia para el modelo final, el cual se definió
#combinando esta información con la de las pruebas manuales

Model_1_grid=grid_results.best_estimator_

history=Model_1_grid.fit(X_train,y_train, epochs=32, batch_size=150, validation_data=(X_val,y_val))

Y_pred=Model_1_grid.predict(X_test)
Y_pred_proba=Model_1_grid.predict_proba(X_test)

test_acc_grid = Model_1_grid.score(X_test, y_test)
print('test_acc_grid:', test_acc_grid)

results=pd.DataFrame(grid_results.cv_results_)
results.columns

#Exportamos resultados del GridSearchCV a formato .csv        
results.to_csv(dir_proyecto+'results_grid_search_images.csv')

#--------------------------------------------------------------------------------------------------------------------------------------------------# 

#Después de las pruebas, elegimos la arquitectura y parámetros que mejor han predicho

#Para 3 clases

#Clases del K-Means
cluster0=["hiphop", "pop", "reggae"]
cluster1=["blues", "classical", "country", "jazz"]
cluster2=["disco", "metal", "rock"]
    
all_labels_images_grouped=pd.Series(all_labels_images)    
all_labels_images_grouped = all_labels_images_grouped.replace(cluster0, "Musica popular")   
all_labels_images_grouped = all_labels_images_grouped.replace(cluster1, "Musica melodica")       
all_labels_images_grouped = all_labels_images_grouped.replace(cluster2, "Musica ritmica")  


#Proporción 70% train, 15% validation, 15% test         
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels_images_grouped, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

X_train = np.asarray(X_train)
X_test= np.asarray(X_test)
X_val= np.asarray(X_val)
y_train=np.asarray(pd.get_dummies(y_train))
y_val=np.asarray(pd.get_dummies(y_val))
y_test=np.asarray(pd.get_dummies(y_test))


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(100, 800, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

history = model.fit(X_train,
                    y_train,
                    epochs=35,
                    batch_size=150, 
                    validation_data=(X_val, y_val))

model.save('best_model_imag_3clases.h5')

Y_pred_proba=model.predict(X_test)
Y_pred=model.predict_classes(X_test)
test_loss_trained_net, test_acc_trained_net = model.evaluate(X_test, y_test)
print('test_acc:', test_acc_trained_net)
#Resultado 75% de acuraccy sobre el test

matrix = confusion_matrix(y_test.argmax(axis=1), Y_pred)

labels_grouped=np.array(["Musica popular", "Musica melodica", "Musica ritmica"])
plot_confusion_matrix(y_test.argmax(axis=1), Y_pred, classes=labels_grouped,
                      title='Confusion matrix, without normalization')


#Graficamos el Loss (error) en función de los Epochs
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc=history.history['acc']
epochs = range(1, len(history.epoch) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#El Loss del validation a partir del Epoch 25 se vuelve mas inestable, lo
#que podría indicar que el modelo empieza a sobreajustar, aunque la tendencia
#es que el accuracy del validation siga disminuyendo hasta el Epoch 35

#--------------------------------------------------------------------------------------------------------------------------------------------------# 

#Después de las pruebas, elegimos la aquitectura y parámetros que mejor han predicho

#Para 10 clases

#Proporción 70% train, 15% validation, 15% test         
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels_images, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

X_train = np.asarray(X_train)
X_test= np.asarray(X_test)
X_val= np.asarray(X_val)
y_train=np.asarray(pd.get_dummies(y_train))
y_val=np.asarray(pd.get_dummies(y_val))
y_test=np.asarray(pd.get_dummies(y_test))


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(100, 800, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

history = model.fit(X_train,
                    y_train,
                    epochs=35,
                    batch_size=150, 
                    validation_data=(X_val, y_val))

model.save('best_model_imag_10clases.h5')

Y_pred_proba=model.predict(X_test)
Y_pred=model.predict_classes(X_test)
test_loss_trained_net, test_acc_trained_net = model.evaluate(X_test, y_test)
print('test_acc:', test_acc_trained_net)
#Resultado 49% de acuraccy sobre el test. Para 10 clases el porcentaje de acierto disminuye en un 25%

matrix = confusion_matrix(y_test.argmax(axis=1), Y_pred)

labels=np.array(clases)
plot_confusion_matrix(y_test.argmax(axis=1), Y_pred, classes=labels,
                      title='Confusion matrix, without normalization')


#Graficamos el Loss (error) en función de los Epochs
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc=history.history['acc']
epochs = range(1, len(history.epoch) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#El Loss del validation a partir del Epoch 25 se vuelve muy inestable, lo
#que podría indicar que el modelo empieza a sobreajustar, aunque la tendencia
#es que el accuracy del validation siga disminuyendo hasta el Epoch 35

