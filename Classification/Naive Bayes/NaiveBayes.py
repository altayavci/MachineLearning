#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 14:20:51 2021

@author: altayavci
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Bayes Theorem:
# Machine 1: 20products/hour Machine 2: 30products/hour
# --> 1% of manufactured products are defective.
# --> 50% of defective products come from machine 1, 50% from machine 2.
# Question: What is the probability that a manufactured defective product was produced on the 2nd machine?
# P(Machine 1) = 30/50 = 0.6
# P(Machine 2) = 20/50 = 0.4
# P(Defective Item) = 0.01
# P(Machine1|Defective) = 0.5 : The probability that the defective products are from machine 1
# P(Machine2|Defective) = 0.5 : The probability that the defective products are from machine 2
# Question : P(Defective|Machine2) = ?? : The probability that the product produced in Machine 2 is defective
# BAYES THEOREM FORMULA:
# P(Defective|Machine2) = [P(Machine2|Defective) * P(Defective)] / P(Machine2)
# P(Defective|Machine2) = [0.5*0.01] / 0.4 = 0.0125 = 1.25%
# Result: The probability that the product produced on Machine 2 is defective is 1.25%.

dataset=pd.read_csv("Social_Network_Ads.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

classifier=GaussianNB()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
accuracy=accuracy_score(y_test, y_pred)  

#TRAIN SET 

X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
    
#TEST SET 

X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
    
    
    



