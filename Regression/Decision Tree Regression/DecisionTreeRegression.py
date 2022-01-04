#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 18:37:39 2021

@author: altayavci
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#Decision Tree is a method used in both classification and regression, it is based on a completely incremental decision-making cycle. It is a model that we can talk about in more detail when we move on to Classification.
#For example, we are applying to a university and the university has some criteria.
#Criteria: 3+ GPA 2+ Certificate 100,000 dollars ...
#Student: 3.25 GPA 1 certificate 150.000 dollars
#When we create a decision tree accordingly:
#1 - GPA > 3 ? YES - NO
#IF answer is "YES" 2 - Certificate > 2 YES - NO ...
#We can extend it like. You already know what it is, I split it into train and test to make it better; so the graphics look clearer.

dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/5,random_state=2)

regressor=DecisionTreeRegressor(random_state=2)
regressor.fit(X_train,y_train)

regressor.predict([[6.5]])
y_pred=regressor.predict(X_test)

#TRAIN SET

X_grid = np.arange(min(X_train), max(X_train), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression) (Train Set)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#TEST SET 

X_grid = np.arange(min(X_train), max(X_train), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression) (Test Set)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()




