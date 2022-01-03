#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 16:56:59 2021

@author: altayavci
"""

import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

dataset=pd.read_csv("nba-players.csv")
X=dataset.iloc[:,2:-1].values
y=dataset.iloc[:,-1].values


X_train,X_test,y_train,y_test = train_test_split(X,y ,test_size=0.2, random_state=12)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


#Here is the code I wrote to detect the n_neighbor and p value in the k-nn neighbor classification. 
#Thanks to the code I wrote, we are trying to achieve the highest success percentage of the classification object that works with n and p values.


dictCollector=dict()
for i in range(1,10): 
        for j in range(1,10):
            classifier = KNeighborsClassifier(n_neighbors =i, metric = "minkowski", p = j)
            classifier.fit(X_train,y_train)
            y_pred=classifier.predict(X_test)
            accuracy=accuracy_score(y_test, y_pred)
            dictCollector[i,j]=accuracy
    
print(max(dictCollector, key=dictCollector.get))
print(max(dictCollector.values()))
   
#I used dict data structure because of functionalty.
        
        
        
        
    