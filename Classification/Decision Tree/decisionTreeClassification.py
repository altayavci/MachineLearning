#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 14:54:25 2021

@author: altayavci
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


dataset=pd.read_csv("bank_marketing_dataset.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,2,3,4,5,6,7,8,9,14])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


#The reason we transform the columns is to encode the string values, we preferred the remainder to passthrough.

le=LabelEncoder()
y = le.fit_transform(y)

#Encoding 


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train[:,53:63]=sc.fit_transform(X_train[:,53:63])
X_test[:,53:63]=sc.transform(X_test[:,53:63])

#Scaling for the train and test sets for the best results.

classifier=DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

cm=confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test,y_pred)*100


     
     
     
     
     