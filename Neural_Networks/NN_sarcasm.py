# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 10:34:25 2022

@author: Michael
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import plotly.io as pio

pio.renderers.default = "browser"
pd.set_option('display.max_columns', None)

data = pd.read_json("Sarcasm.json", lines=True)


data = data[["headline", "is_sarcastic"]]
x = np.array(data["headline"])
y = np.array(data["is_sarcastic"])

cv = CountVectorizer()
X = cv.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = BernoulliNB()
model.fit(X_train, y_train)
#test the model with the test set
print(model.score(X_test, y_test))

user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
if output == 0:
    print("no saracasm")
else:
    print("saracasm")


