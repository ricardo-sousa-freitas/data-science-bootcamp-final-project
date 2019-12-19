# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:41:59 2019

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
import scipy.stats as stats
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras import layers
from keras import models
from sklearn import preprocessing
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import keras
from keras import regularizers
from keras import optimizers
from sklearn.utils.multiclass import unique_labels
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import pickle
  
#Ruta carpeta Proyecto
dir_proyecto = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/"
os.chdir(dir_proyecto)
os.listdir()

from pf_funciones import plot_confusion_matrix

#Importar dataset a dataframe Pandas
df_features = pd.read_csv(dir_proyecto+'genres_feat_extracted_cleaned.csv',delimiter=',', index_col=0)

#Ruta muestras de audio. 100 audios .wav por género para 10 géneros musicales
audio_path = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/Datasets/gtzan-genres/audios/"

os.chdir(audio_path)
os.listdir()

#Lista de las 10 clases
clases=[]
for i in os.listdir():
    clases.append(i)

#--------------------------------------------------------------------------------------------------------------------------------------------------# 

#Usamos GridSearchCV, para definir la arquitectura e hiperparametros que den mejores resultados

#Hacemos una Red Neuronal Densa para predecir para los 10 géneros
    
#Proporción 70% train, 15% validation, 15% test         
X_train, X_test, y_train, y_test = train_test_split(df_features.iloc[:,:-3], df_features.iloc[:,33], test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


X_train = np.array(X_train)
X_test= np.array(X_test)
X_val= np.array(X_val)
y_train=np.asarray(pd.get_dummies(y_train))
y_val=np.asarray(pd.get_dummies(y_val))
y_test=np.asarray(pd.get_dummies(y_test))

def weigths_model(nro_capas, neurons1, neurons2, dropout, regulari):
    model = models.Sequential()
    model.add(layers.Dense(neurons1, activation='relu', input_shape=(32,)))
    for i in range(nro_capas):
        print(i)
        model.add(layers.Dense(neuronas2[(np.random.choice(len(neuronas2), size=1, replace=False))[0]], kernel_regularizer=regularizers.l2(regulari), activation='relu'))
        model.add(layers.Dropout(drop[(np.random.choice(len(drop), size=1, replace=False))[0]]))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=weigths_model)
nro_capas=[2,3]
neuronas1=[128,256]
neuronas2=[64,128,256]
drop=[0,0.5]
regu=[0.0001, 0.001]
param_grid = dict(nro_capas=nro_capas,neurons1=neuronas1, neurons2=neuronas2, dropout=drop, regulari=regu)

grid = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    verbose = 2,
                    cv=2)

grid_results=grid.fit(X_train,y_train, epochs=50, batch_size=150, verbose=2, validation_data=(X_val,y_val))
#Debido a las limitaciones de memoria del ordenador, los parámetros del grid tuvieron que limitarse.
#Sus resultados se utilizaron solo de referencia para el modelo final, el cual se definió
#sobretodo a partir de pruebas manuales

Model_2_grid=grid_results.best_estimator_

history=Model_2_grid.fit(X_train,y_train, epochs=200, batch_size=150, validation_data=(X_val,y_val))

Y_pred=Model_2_grid.predict(X_test)
Y_pred_proba=Model_2_grid.predict_proba(X_test)

test_acc_grid = Model_2_grid.score(X_test, y_test)
print('test_acc_grid:', test_acc_grid)

results=pd.DataFrame(grid_results.cv_results_)
results.columns

#Exportamos resultados del GridSearchCV a formato .csv        
results.to_csv(dir_proyecto+'results_grid_search_features.csv')


#--------------------------------------------------------------------------------------------------------------------------------------------------# 

#Después de las pruebas, elegimos la arquitectura y parámetros que mejor han predicho

#Para 10 clases

#Proporción 70% train, 15% validation, 15% test         
X_train, X_test, y_train, y_test = train_test_split(df_features.iloc[:,:-3], df_features.iloc[:,33], test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

scaler_train_10=StandardScaler().fit(X_train)
X_train=scaler_train_10.transform(X_train)
pickle.dump( scaler_train_10, open( "scaler_train_10.p", "wb" ) )

scaler_val_10=StandardScaler().fit(X_val)
X_val=scaler_val_10.transform(X_val)
pickle.dump( scaler_val_10, open( "scaler_val_10.p", "wb" ) )

scaler_test_10=StandardScaler().fit(X_test)
X_test=scaler_test_10.transform(X_test)
pickle.dump( scaler_test_10, open( "scaler_test_10.p", "wb" ) )


X_train = np.array(X_train)
X_test= np.array(X_test)
X_val= np.array(X_val)
y_train=np.asarray(pd.get_dummies(y_train))
y_val=np.asarray(pd.get_dummies(y_val))
y_test=np.asarray(pd.get_dummies(y_test))


model = models.Sequential()
model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(32,)))
model.add(layers.Dense(128, activation='relu', ))
model.add(layers.Dense(64, activation='relu',))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()

history = model.fit(X_train,
                    y_train,
                    epochs=35,
                    batch_size=120, 
                    validation_data=(X_val, y_val))

model.save('best_model_feat_10classes.h5')

Y_pred_proba=model.predict(X_test)
Y_pred=model.predict_classes(X_test)
test_loss_trained_net, test_acc_trained_net = model.evaluate(X_test, y_test)
print('test_acc:', test_acc_trained_net)
#Resultado 69% de acuraccy sobre el test

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

#--------------------------------------------------------------------------------------------------------------------------------------------------# 

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


#--------------------------------------------------------------------------------------------------------------------------------------------------# 

#Después de las pruebas, elegimos la arquitectura y parámetros que mejor han predicho

#Para 3 clases

#Recordatorio clases del K-Means
cluster0=["hiphop", "pop", "reggae"]
cluster1=["blues", "classical", "country", "jazz"]
cluster2=["disco", "metal", "rock"]
    

#Proporción 70% train, 15% validation, 15% test         
X_train, X_test, y_train, y_test = train_test_split(df_features.iloc[:,:-3], df_features.iloc[:,32], test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

scaler_train_3=StandardScaler().fit(X_train)
X_train=scaler_train_3.transform(X_train)
pickle.dump( scaler_train_3, open( "scaler_train_3.p", "wb" ) )

scaler_val_3=StandardScaler().fit(X_val)
X_val=scaler_val_3.transform(X_val)
pickle.dump( scaler_val_3, open( "scaler_val_3.p", "wb" ) )

scaler_test_3=StandardScaler().fit(X_test)
X_test=scaler_test_3.transform(X_test)
pickle.dump( scaler_test_3, open( "scaler_test_3.p", "wb" ) )

X_train = np.array(X_train)
X_test= np.array(X_test)
X_val= np.array(X_val)
y_train=np.asarray(pd.get_dummies(y_train))
y_val=np.asarray(pd.get_dummies(y_val))
y_test=np.asarray(pd.get_dummies(y_test))


model = models.Sequential()
model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(32,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()

history = model.fit(X_train,
                    y_train,
                    epochs=25,
                    batch_size=120, 
                    validation_data=(X_val, y_val))

model.save('best_model_feat_3classes.h5')

Y_pred_proba=model.predict(X_test)
Y_pred=model.predict_classes(X_test)
test_loss_trained_net, test_acc_trained_net = model.evaluate(X_test, y_test)
print('test_acc:', test_acc_trained_net)
#Resultado 78% de acuraccy sobre el test

matrix = confusion_matrix(y_test.argmax(axis=1), Y_pred)

labels_grouped=np.array(["Música popular", "Música melódica", "Musica rítmica"])
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




