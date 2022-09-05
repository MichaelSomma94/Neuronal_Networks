# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 09:12:55 2022

@author: Michael
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras import models



(trainX, trainy), (testX, testy) = mnist.load_data()

# printing some handwritten digits
# for i in range(9):  
#     plt.subplot(330 + 1 + i)
#     plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
#     plt.show()

# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.Dense(300, activation="relu"),
#     keras.layers.Dense(100, activation="relu"),
#     keras.layers.Dense(10, activation="softmax")
# ])
# #subdivinding the training set into training and validation

# xvalid, xtrain = trainX[:5000]/np.amax(trainX), trainX[5000:]/np.amax(trainX) #scaling from 0 to 1
# yvalid, ytrain = trainy[:5000], trainy[5000:]

# n_batch = 100
# n_epochs = 10
# model.compile(loss="sparse_categorical_crossentropy",
#               optimizer="sgd",
#               metrics=['accuracy'])
# history = model.fit(xtrain, ytrain, epochs=n_epochs, 
#                     validation_data=(xvalid, yvalid), batch_size=n_batch, verbose=1, shuffle=True)
# saves the build model
# model.save('NN_digits')


# results = model.evaluate(testX, testy, batch_size=128)
# print("test loss, test acc:", results)

def process_image(fname):
    # Image.open() can also open other image types
    img = Image.open(fname)
    # compress the image to 28x28
    resized_img = img.resize((28, 28))
    #resized_img.save("resized_image.jpg")
    #converts the RGB image to a greyvalue numpy array with the right dimension for testing
    gray_img = 255.0-np.array(ImageOps.grayscale(resized_img))
   

    gray_img = np.expand_dims(gray_img,0)
    
    return gray_img


#now lets see if we can recognize it correctly
model = models.load_model("NN_digits") # reloads the built model
gray_img = process_image('zero_handwritten.png')
predictions = model.predict(gray_img)
print(predictions)

# this plots the image called img
# plt.subplot()
# plt.imshow(gray_img, cmap=plt.get_cmap('gray'))
# plt.show()


