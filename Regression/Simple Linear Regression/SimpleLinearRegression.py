#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 17:02:59 2021

@author: altayavci
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#We do the same as we did in data preprocessing.
#In this dataset, we examine how much work experience affects salary.
#Experience will be our value x and salary will be our value y. Since we are doing Simple Linear Regression we only have 1 x value.
#In Multiple Linear Regression we will diversify this x value. For example, the salary, the certificates we receive, etc. can also affect it.

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 30)

# Since there is no nan value in the data, we created direct train and test arrays.

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Here we regressed the train data.

y_pred = regressor.predict(X_test)

#Here we predicted the test sets
#Let's say #delivery is "Salary = 25000 + 0.4*YearsofExperience.
#Then, if YearsofExperience is 5, asking a question like what will our salary be is to make a prediction.

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#The simplest way to understand the result of the regression is to draw a scatter plot and a trend line. Since we are doing it on the training set, we write our train sets in plt.scatter().
# Likewise, the set we make predictions should also be a train set. = regressor.predict(X_train) The rest is simply edits anyway.
#In summary, while the scatter plot gives us the actual values in the data set, it shows the values we predicted according to our trend line model.

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
#Here, we did the same with the test set.
#Since the model train is built from the set, we need to do the predictin again with the train set, so the trend should be the line train set.

#train set has data loaded while test set has little data. The aim is to load too much data on the train set and get the best performance possible with less data on the test set.
#Machine learns on the train set, and applies what he learns on the test set.


