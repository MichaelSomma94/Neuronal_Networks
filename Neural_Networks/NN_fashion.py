# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 09:46:17 2022

@author: Michael
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion = keras.datasets.fashion_mnist
(xtrain, ytrain), (xtest, ytest) = fashion.load_data()

# imgIndex = 10
# image = xtrain[imgIndex]
# print("Image Label :",ytrain[imgIndex])
# plt.imshow(image)

# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.Dense(300, activation="relu"),
#     keras.layers.Dense(100, activation="relu"),
#     keras.layers.Dense(10, activation="softmax")
#])
#print(model.summary())

xvalid, xtrain = xtrain[:5000]/np.amax(xtrain), xtrain[5000:]/np.amax(xtrain) # scaling from 0 to 1
yvalid, ytrain = ytrain[:5000], ytrain[5000:]




# model.compile(loss="sparse_categorical_crossentropy",
#               optimizer="sgd",
#               metrics=['accuracy'])
# history = model.fit(xtrain, ytrain, epochs=30, 
#                     validation_data=(xvalid, yvalid))
# new = xtest[:5]
# predictions = model.predict(new)
# print(predictions)

# results = model.evaluate(xtest, ytest, batch_size=128)
# print("test loss, test acc:", results)