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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


#Ruta carpeta proyecto
dir_proyecto = "C:/Users/ricar/OneDrive/Documentos/Bootcamp_Data_Science/Machile_Learning/Proyecto/"
os.chdir(dir_proyecto)
os.listdir()

#Importar dataset a dataframe pandas
df_features = pd.read_csv(dir_proyecto+'genres_feat_extracted_cleaned.csv',delimiter=',', index_col=0)






























#def redneuronal(n_capas, dropout, optimizer, loss):
#    model = models.Sequential()
##    model.add(layers.Conv2D(32, (3,3), kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(200,200,1)))
#    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,1)))
#    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding = 'same'))
#    for i in range(n_capas):
#        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#        model.add(layers.MaxPooling2D(pool_size=(2, 2), padding = 'same'))
#    model.add(layers.Flatten())
#    model.add(layers.Dense(512, activation='relu'))
#    model.add(layers.Dropout(dropout))
#    model.add(layers.Dense(4, activation='softmax'))
#    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
#    return model
##Distintos dropout, capas conv, nro de filtros, nro de epochs, learning rate? y de bachs.
##probar cpn resnet, inception googlenet etc.
#model = KerasClassifier(build_fn=redneuronal)
#params={'optimizer': ["SGD"],
#        'loss':['categorical_crossentropy'],
#        'n_capas': [3],
#        'dropout': [0.5]}
#grid_solver = GridSearchCV(estimator = model,
#                   param_grid = params,
#                   cv = 5,
#                   verbose = 0)
#history=grid_solver.fit(X_train,y_train,epochs=3, batch_size=600, validation_data=(X_val,y_val))
       
#
##podemos ver qué combinación de cuántas neuronas en cada capa nos da el mejor score
#results[['params','mean_test_score']]
#
#
#def create_model(neurons1=1,neurons2=1,neurons3=1,dropout=0.5):
#    # create model
#    model = models.Sequential()
#    model.add(layers.Dense(neurons1, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(10000,)))
#    model.add(layers.Dropout(dropout))   
#    model.add(layers.Dense(neurons2, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
#    model.add(layers.Dropout(dropout))
#    model.add(layers.Dense(neurons3, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
#    model.add(layers.Dropout(dropout))
#    model.add(layers.Dense(1, activation='sigmoid'))
#    model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#    return model
#
#
#model = KerasClassifier(build_fn=create_model, epochs=15, batch_size=512, verbose=2)
#neurons1 = [2]
#neurons2 = [4,5]
#neurons3 = [3,5]
#param_grid = dict(neurons1=neurons1,neurons2=neurons2,neurons3=neurons3)
## https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
#grid = GridSearchCV(estimator=model,
#                    param_grid=param_grid,
#                    verbose = 2,
#                    cv=2)
#
## si cv (folds en cross validations) no se especifica, se utilizaran 3
#
#
#grid_results=grid.fit(x_train,y_train)
#
#MODEL1=grid_results.best_estimator_
#MODEL1.fit(x_train,y_train)
#predicciones=MODEL1.predict(x_test)
#predicciones_proba=MODEL1.predict_proba(x_test)
#
#test_acc_grid = MODEL1.score(x_test, y_test)
#print('test_acc_grid:', test_acc_grid)
#
#
#results=pd.DataFrame(grid_results.cv_results_)
#results.columns
#
#
##podemos ver qué combinación de cuántas neuronas en cada capa nos da el mejor score
#results[['params','mean_test_score']]
#
#
###a. Haz una grid search del mismo modelo cambiando el coeficiente de regulación l2 en cada hidden layer? Poniendo el valor en 0.0001, 0.001 y 0.01. Cual da mejor resultado?
#
#def create_model(neurons=16,l2s=0.01):
#    # create model
#    model = models.Sequential()
#    model.add(layers.Dense(neurons, kernel_regularizer=regularizers.l2(l2s), activation='relu', input_shape=(10000,)))
#    model.add(layers.Dense(neurons, kernel_regularizer=regularizers.l2(l2s), activation='relu'))
#    model.add(layers.Dense(1, activation='sigmoid'))
#    model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#    return model
#
#
#model = KerasClassifier(build_fn=create_model, epochs=15, batch_size=512, verbose=1)
#l2s = [0.0001, 0.001, 0.01]
#param_grid = dict(l2s=l2s)
#grid = GridSearchCV(estimator=model,
#                    param_grid=param_grid,
#                    cv=2)
#
#
#grid_results=grid.fit(x_train,y_train)
#
#test_acc = grid_results.score(x_test, y_test)
#print('test_acc:', test_acc)
#
#results=pd.DataFrame(grid_results.cv_results_)
#results[['params','mean_test_score']].sort_values(by=['mean_test_score'],ascending=False)
#
###b. Para la mejor red neuronal anterior, haz una haz dos modelos (GridSearchCv de scikit) con un dropout de 0.4 y 0.6. Quedate con el mejor modelo. Que significa el drop out?
#
#
#def create_model(neurons=16,l2s=0.001,dropout=0.4):
#    # create model
#    model = models.Sequential()
#    model.add(layers.Dense(neurons, kernel_regularizer=regularizers.l2(l2s), activation='relu', input_shape=(10000,)))
#    model.add(layers.Dropout(dropout))
#    model.add(layers.Dense(neurons, kernel_regularizer=regularizers.l2(l2s), activation='relu'))
#    model.add(layers.Dropout(dropout))
#    model.add(layers.Dense(1, activation='sigmoid'))
#    model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#    return model
#
##con dropout estamos quitando una serie de nodos de cada capa para evitar overfitting
##according to keras documentation "Dropout consists in randomly setting a fraction rate of input units to 0 at each update"
#    
#model = KerasClassifier(build_fn=create_model, epochs=15, batch_size=512, verbose=2)
#dropouts = [0.4, 0.6]
#param_grid = dict(dropout=dropouts)
#grid = GridSearchCV(estimator=model,
#                    param_grid=param_grid,
#                    verbose = 2,
#                    cv=2)
#
#grid_results=grid.fit(x_train,y_train)
#
#MODEL2=grid_results.best_estimator_
#MODEL2.fit(x_train,y_train)
#predicciones_model2=MODEL2.predict(x_test)
#predicciones_proba_model2=MODEL2.predict_proba(x_test)
#
#test_acc_grid_model_2 = MODEL2.score(x_test, y_test)
#print('test_acc_grid_model2:', test_acc_grid_model_2)
#
#
#
#results=pd.DataFrame(grid_results.cv_results_)
#results[['params','mean_test_score']].sort_values(by=['mean_test_score'],ascending=False)
#
#
###c. Haz una red neuronal con 3 hidden layers y diferentes combinaciones de hidden units por cada layer, utiliza la GridSearchCv de scikit.
#
#
#def create_model(neurons=16,l2s=0.001,dropout=0.6):
#    # create model
#    model = models.Sequential()
#    model.add(layers.Dense(neurons, kernel_regularizer=regularizers.l2(l2s), activation='relu', input_shape=(10000,)))
#    model.add(layers.Dropout(dropout))
#    model.add(layers.Dense(neurons, kernel_regularizer=regularizers.l2(l2s), activation='relu'))
#    model.add(layers.Dropout(dropout))
#    model.add(layers.Dense(neurons, kernel_regularizer=regularizers.l2(l2s), activation='relu'))
#    model.add(layers.Dropout(dropout))
#    model.add(layers.Dense(1, activation='sigmoid'))
#    model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#    return model
#
#model = KerasClassifier(build_fn=create_model, epochs=15, batch_size=512, verbose=2)
#neurons = [4, 8, 16]
#param_grid = dict(neurons=neurons)
#grid = GridSearchCV(estimator=model,
#                    param_grid=param_grid,
#                    verbose = 2,
#                    cv=2)
#
#grid_results=grid.fit(x_train,y_train)
#
#test_acc = grid_results.score(x_test, y_test)
#print('test_acc:', test_acc)
#
#results=pd.DataFrame(grid_results.cv_results_)
#results[['params','mean_test_score']].sort_values(by=['mean_test_score'],ascending=False)
#
#
###d. Haz una red neuronal con 4 hidden layers y diferentes combinaciones de hidden units por cada layer, utiliza la GridSearchCv de scikit
#
#def create_model(neurons=16,l2s=0.001,dropout=0.6):
#    # create model
#    model = models.Sequential()
#    #hidden layer1
#    model.add(layers.Dense(neurons, kernel_regularizer=regularizers.l2(l2s), activation='relu', input_shape=(10000,)))
#    model.add(layers.Dropout(dropout))
#    #hidden layer2
#    model.add(layers.Dense(neurons, kernel_regularizer=regularizers.l2(l2s), activation='relu'))
#    model.add(layers.Dropout(dropout))
#    #hidden layer3
#    model.add(layers.Dense(neurons, kernel_regularizer=regularizers.l2(l2s), activation='relu'))
#    model.add(layers.Dropout(dropout))
#    #hidden layer4
#    model.add(layers.Dense(neurons, kernel_regularizer=regularizers.l2(l2s), activation='relu'))
#    model.add(layers.Dropout(dropout))
#    #output layer
#    model.add(layers.Dense(1, activation='sigmoid'))
#    model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#    return model
#
#
#model = KerasClassifier(build_fn=create_model, epochs=15, batch_size=512, verbose=1)
#neurons = [4, 8, 16]
#param_grid = dict(neurons=neurons)
#grid = GridSearchCV(estimator=model,
#                    param_grid=param_grid,
#                    cv=2)
#
#grid_results=grid.fit(x_train,y_train)
#
#test_acc = grid_results.score(x_test, y_test)
#print('test_acc:', test_acc)
#
#results=pd.DataFrame(grid_results.cv_results_)
#results[['params','mean_test_score']].sort_values(by=['mean_test_score'],ascending=False)
#
##e. Escoge el mejor modelo del apartado 3 y 4. En qué te basas? Ahora haz una grid search con la regularización y el dropout. Mejora?
#
#def create_model(neurons=8,l2s=0.001,dropout=0.6):
#    # create model
#    model = models.Sequential()
#    #hidden layer1
#    model.add(layers.Dense(neurons, kernel_regularizer=regularizers.l2(l2s), activation='relu', input_shape=(10000,)))
#    model.add(layers.Dropout(dropout))
#    #hidden layer2
#    model.add(layers.Dense(neurons, kernel_regularizer=regularizers.l2(l2s), activation='relu'))
#    model.add(layers.Dropout(dropout))
#    #hidden layer3
#    model.add(layers.Dense(neurons, kernel_regularizer=regularizers.l2(l2s), activation='relu'))
#    model.add(layers.Dropout(dropout))
#    #output layer
#    model.add(layers.Dense(1, activation='sigmoid'))
#    model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#    return model
#
#
#model = KerasClassifier(build_fn=create_model, epochs=15, batch_size=512, verbose=1)
#l2s = [0.1, 0.01, 0.001]
#dropout = [0.2, 0.4, 0.6]
#param_grid = dict(l2s=l2s,
#                  dropout=dropout)
#grid = GridSearchCV(estimator=model,
#                    param_grid=param_grid,
#                    cv=2)
#
#grid_results=grid.fit(x_train,y_train)
#
#test_acc = grid_results.score(x_test, y_test)
#print('test_acc:', test_acc)
#
#results=pd.DataFrame(grid_results.cv_results_)
#results[['params','mean_test_score']].sort_values(by=['mean_test_score'],ascending=False)
#
##f. Ahora cambia la función de activación de las hidden units.
#
#def create_model(neurons=8,l2s=0.01,dropout=0.2,activations='relu'):
#    # create model
#    model = models.Sequential()
#    #hidden layer1
#    model.add(layers.Dense(neurons, kernel_regularizer=regularizers.l2(l2s), activation=activations, input_shape=(10000,)))
#    model.add(layers.Dropout(dropout))
#    #hidden layer2
#    model.add(layers.Dense(neurons, kernel_regularizer=regularizers.l2(l2s), activation=activations))
#    model.add(layers.Dropout(dropout))
#    #hidden layer3
#    model.add(layers.Dense(neurons, kernel_regularizer=regularizers.l2(l2s), activation=activations))
#    model.add(layers.Dropout(dropout))
#    #output layer
#    model.add(layers.Dense(1, activation='sigmoid'))
#    model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#    return model
#
#
#model = KerasClassifier(build_fn=create_model, epochs=15, batch_size=512, verbose=1)
#activations = ['relu', 'elu', 'softmax', 'selu', 'softplus'] #hay muchas, aqui voy a probar unas pocas. 
##la lista completa se puede mirar en la documentacion: https://keras.io/activations/
##podríamos probar distintas activaciones para distintas capas
#param_grid = dict(activations=activations)
#grid = GridSearchCV(estimator=model,
#                    param_grid=param_grid,
#                    cv=2)
#
#grid_results=grid.fit(x_train,y_train)
#
#test_acc = grid_results.score(x_test, y_test)
#print('test_acc:', test_acc)
#
#results=pd.DataFrame(grid_results.cv_results_)
#results[['params','mean_test_score']].sort_values(by=['mean_test_score'],ascending=False)