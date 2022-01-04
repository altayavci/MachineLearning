#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:40:51 2021

@author: altayavci
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset=pd.read_csv("Position_Salaries4.csv")
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

lin_reg = LinearRegression()
lin_reg.fit(X, y)

# We made it to make a comparison on what will be seen below.

poly_reg = PolynomialFeatures(degree =10)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#poly_reg = PolynomialFeatures(degree = 4) --> We determine the degree of polynomial regression. 4th order regression means the equation where the exponential degree of X goes up to 4.
#X_poly = poly_reg.fit_transform(X) --> Most importantly, fit_transform makes our X values fit for regression. So what we define as X_poly now includes 1, 2, 3 and 4 degrees of X. I wanted you to understand this better by printing it.
#lin_reg_2 = LinearRegression() --> We simply create Linear Regression.
#lin_reg_2.fit(X_poly, y) --> We put the y value into linear regression with X_poly, which includes all X values up to the 4th degree.
#The logic of Python is actually very very simple, in python the whole event is about defining the X value properly.

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#As it can be seen here, we have created a polynomial regression function, so when we tried to do linear operation, we detected quite a few deviations in the result, namely variations.

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Here, when we predicted the polynomial regression, the result was much more satisfactory.

print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))

#We wanted to see predict here too, according to the number 6.5






