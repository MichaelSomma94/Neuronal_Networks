# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 11:13:40 2022

@author: Michael
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import plotly.io as pio
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


pio.renderers.default = "browser"
pd.set_option('display.max_columns', None)

data = pd.read_json("Sarcasm.json", lines=True)


data = data[["headline", "is_sarcastic"]]
x = np.array(data["headline"])
y = np.array(data["is_sarcastic"])


vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000
epochs = 5

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.9, random_state = 73)

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(x_train)
x_train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(x_test)
x_test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,recurrent_dropout = 0.3 , dropout = 0.3, return_sequences = True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,recurrent_dropout = 0.1 , dropout = 0.1)),
    tf.keras.layers.Dense(512, activation = "relu"),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train_padded,y_train, batch_size = 128, epochs = epochs, validation_data = (x_test_padded, y_test))