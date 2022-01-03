#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 16:39:24 2021

@author: altayavci
"""

import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

dataset=pd.read_csv("nba-players.csv")
X=dataset.iloc[:,2:-1].values
y=dataset.iloc[:,-1].values


X_train,X_test,y_train,y_test = train_test_split(X,y ,test_size=0.2, random_state=12)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

classifier=SVC(kernel="rbf",random_state=12)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
