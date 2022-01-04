#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 16:28:39 2021

@author: altayavci
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Here we divided the dataset into X and Y arrays.

imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(X[:,2:4])
X[:,2:4]=imputer.transform(X[:,2:4])

#Here X has nan values, so we assigned them to their mean value. In order not to cause problems in regression and encoding

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough")
X = np.array(ct.fit_transform(X))
X=np.delete(X,3,1) 

#I did the delete because it was indexing rows and this was messing with the encoding.
#The reason we do Encoding is to convert the string elements in the array to floats, that is to encode them. 

le = LabelEncoder()
y = le.fit_transform(y)

#Here, it was enough to use LabelEnconder() since there are only yes and no strings in the y array. The machine will detect them as 1 and 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30)

#We create the model we created with the train set and then we check whether this model works for all values with the test set.
#0.2 means 80% goes to the train set and 20% goes to the test set.
#random_state indicates how many times the data in the sets should be shuffled.

sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:,3:] = sc.transform(X_test[:, 3:])

#This part is one of the most important parts. Regression is a bit of a problem when there are disparate data. Because when we square the values, they diverge too far from each other.
#Here, we briefly bring the data closer together and scale it in a certain order.













