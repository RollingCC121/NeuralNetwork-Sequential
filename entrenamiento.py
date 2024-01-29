"""
# Redes Neuronales Profundas-alfabeto
1. Preparacion de datos
2.Division de datos
3.Aprendizaje del modelo
4.Evaluacion del modelo
5.Guardar el modelo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel("alfabeto.xlsx", sheet_name= "alfabeto")
data.head()

data .info()

data["letra"] = data["letra"].astype("category")
data.info()

from sklearn.preprocessing import  LabelEncoder
labelencoder = LabelEncoder()
data["letra"] = labelencoder.fit_transform(data["letra"])
data["letra"]

#divison 70-30
from sklearn.model_selection import train_test_split
X = data.drop("letra", axis = 1) # variables predictoras
Y = data["letra"] #variable objetivo
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.3, stratify = Y)
Ytrain.value_counts().plot(kind = "bar")

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

#Asignar semillas para generar numeros aleatorios
import tensorflow as tf
tf.random.set_seed(3)

#Arquitectura de la red neuronal
model_deep = Sequential()
model_deep.add(Dense(30,input_dim = 35, activation = "relu"))#35 entradas y capa oculta de 30 neuronas
model_deep.add(Dense(28, activation = "relu")) #capa oculta de 28 neuronas
model_deep.add(Dense(26, activation = "softmax")) #capa de salida (valores de la variable objetivo)

#Aprendizaje
optimizer = keras.optimizers.SGD(learning_rate=0.03, momentum = 0.02)
model_deep.compile(loss="sparse_categorical_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
history = model_deep.fit(Xtrain,Ytrain, epochs=500)

plt.plot(history.history["loss"])

model_deep.summary()

#evalucaion con exactitud del modelo deep
loss, acc = model_deep.evaluate(Xtest,Ytest,verbose=0) #30%
print(loss)
print(acc)

#guardamos el modelo
model_deep.save("model_deep.h5")

#guardamos el labelencoder y los nombres de columnas si son necesarios
import pickle
filename = "labelencoder.pkl"
pickle.dump(labelencoder,open(filename, "wb"))
