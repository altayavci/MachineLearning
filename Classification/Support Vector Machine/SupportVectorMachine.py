#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 16:01:41 2021

@author: altayavci
"""

import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Basically, we worked on the logic of the Support Vector while doing the regression, actually, we said that it works very well on grouped data.
# SVM basically separates the data by means of a plane or a line, but makes this distinction according to the elements on the support vector, that is, on the border.
# I'm sending you a photo similar to the one in the mail, it's about SVM, you will understand it directly.
# We got a graph very similar to the logistic regression graph because we used a linear kernel. If you want to use non-linear kernel methods, I leave the parameters taken by the kernel below.
# kernel{'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'

dataset=pd.read_csv("heart_failure_clinical_records_dataset.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,3,5,9,10])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#The reason we encode here is not to risk it

X_train,X_test,y_train,y_test = train_test_split(X,y ,test_size=0.2, random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

classifier=SVC(kernel="linear",random_state=42)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))




