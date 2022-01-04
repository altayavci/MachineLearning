#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 17:32:38 2021

@author: altayavci
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Burda has written an algorithm for estimating profit, so we assigned profit to y, that is, to the dependent part.

ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[3])],remainder="passthrough")
X = np.array(ct.fit_transform(X))

#The reason we gave the index 3 is because there are strings in the 4th column and encoding.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 30)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Regressors are the same when doing MLR and SLR.
#We are using train sets when creating the Regressor object!!

y_pred = regressor.predict(X_test)

#We also use test sets when making predictions.
#TRENDLINE!! We use train sets for prediction while drawing. Just because there is more data.

np.set_printoptions(precision=2)
concatenateArray=np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)

for i,j in zip(concatenateArray[:,0],concatenateArray[:,1]):
    print("Predict:{}\t Test:{}".format(i,j))
    
#We use a test set to test the model, so to speak.
#np.set_printoptions(precision=2) code shows the maximum decimal. Looking at the output of the code, we can say that the code works.
#np.concatenate --> Code for combining vectors and arrays. The reason we do .reshape is because we want arrays to appear vertically, not horizontally.
# Thus, we will be able to compare the real y values more easily with the y_pred values obtained as a result of the application of our model.

