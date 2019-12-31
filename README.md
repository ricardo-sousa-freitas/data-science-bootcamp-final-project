# proyecto
Redes neuronales para la clasificación de música utilizando atributos y espectrogramas

Archivos:
* 1.pf_creacion_dataset_features.py: script para crear el dataset de features (dataframe Pandas) a partir de las funciones de la libreria de Python Librosa.

* 2.pf_kmeans_dataset_features.py: script para el análisis del dataset de features con el algoritmo de Machine Learning no supervisado K-Means.

* 3.pf_cleaning_dataset_features.py: script para el análisis, visualización y limpieza del dataset de features.

* 4.pf_mod_dataset_features.py: script para los algoritmos y resultados de Redes Neuronales tipo Perceptron Multicapa para el dataset de features.

* 1.pf_creacion_dataset_imagenes.py: script para cargar los audios, generar su espectrograma de Mel y guardar las imágenes en la carpeta del género musical correspondiente.

* 2.pf_mod_dataset_imagenes.py: script para los algoritmos y resultados de Redes Neuronales Convolucionales para el dataset de imágenes.

* Memoria del proyecto - Ricardo Sousa Freitas.pdf

* interfaz_prediccion_genero.pyw: interfaz gráfica con libreria de Python Tkinter para la predicción del género musical de una canción de Youtube usando el mejor modelo de la Red Neuronal para 10 clases (archivo: pf_mod_dataset_features.py).

* pf_funcion_prediccion.py: script de la función que descarga una canción de Youtube en formato .wav, extrae los respectivos features con Librosa y predice el género musical con el modelo "best_model_feat_10classes.h5". Esta es la función a la que llama la interfaz gráfica (archivo: interfaz_prediccion_genero.pyw).



