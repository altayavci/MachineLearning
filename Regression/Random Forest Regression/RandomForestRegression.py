#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 18:55:48 2021

@author: altayavci
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sklearn.metrics as metrics



#Random Forest is a combination of many decision trees bro, so much more selection and restriction criteria are revealed.
#By changing #n_estimators = 10, you can examine the changes in the graph.
#If you examine the decision tree graph with the #random forest graph, you will see that there are many more bumps in the random forest graph.

dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
regressor=RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(X_train,y_train)

print(regressor.predict([[7]]))
y_pred=regressor.predict(X_test)

print(r2_score(y_test,y_pred))

#We want to see the r^2 score here
# you can use this when applying any regression.
#You can even extract the r-square values of the regression variants we're working with. r^2 helps us to understand how well the regression is working for our dataset.
#The r^2 value tells us whether the dependent and independent variables are really related, and uses variance values to do so.
#For us, a higher r^2 value represents a better model, but that's not always the case, but I'm not getting there.
#In general, r^2 values above 0.7 represent a good model, and values below 0.4 represent a bad model.


def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
    print("MDAE:",round(median_absolute_error,4))
    
regression_results(y_test, y_pred)

#TRAIN SET 

X_grid = np.arange(min(X_train), max(X_train), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression TRAIN)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#TEST SET 

X_grid = np.arange(min(X_train), max(X_train), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression TEST)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
