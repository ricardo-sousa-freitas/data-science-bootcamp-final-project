# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:50:19 2019

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

#IMPORTAR DATASET A DATAFRAME PANDAS
df_features = pd.read_csv(dir_proyecto+'genres_feat_extracted.csv',delimiter=',', index_col=0)




