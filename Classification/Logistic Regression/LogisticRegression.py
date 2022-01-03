#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:19:30 2021

@author: altayavci
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

#The main difference between classification and regression:
#Classification gives a class label. Regression gives computational value.
#When estimating a numerical result in regression (salary, age, etc.);
#We estimate a category in the classification. (boolean) takes/doesn't get sick/not sick and can/can't have a heart attack
#The first topic of the classification section is "Logistic Regression", don't look at its name being Regression, it is actually a basic classification model.
#If you remember in previous regressions, we always tried to predict a numerical value.
#For example: Salary = 25000 + 0.4*YearsofExperience
#In Logistic Regression, what we are trying to predict will be a category.
#You can check the dataset by opening it for better understanding. The dataset represents exactly 400 potential customers. We can create a story on that. We are a car company and we sent our advertisement to these 400 customers by e-mail. As you can see in the dataset, our database has these 400 customers' ages and estimated salary levels and whether they bought the car we're advertising.
#In this case, while our independent variables, namely X = Age and EstimatedSalary, are the dependent variable we are looking for, y = Purchased.
#Purchased logically only takes two values: 0 or 1. So it stays in our dataset, self-encoded.
#We want to predict whether our new potential customers will buy our car by putting this dataset into logistic regression.
#In summary, we can say that we use logistic regression when searching for a category. For example: Dead/Alive, Takes/Not etc.

dataset=pd.read_csv("Social_Network_Ads.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/4,random_state=0)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#What we call Classifier is, after all, our classifier.
#We do this classification with Logistic Regression.
#Model building codes in #Python are always the same logic and easy so far.

print(classifier.predict(sc.transform([[30,87000]])))

#The important thing for us is, of course, the predictions that the model will make, we are guessing manually here.
#If we are using the data by scaling, we need to put the values that we will predict into our scaler object.
# That's why we write it in sc.transform. Age = 30 Salary = 87000 arguments for the customer we predicted.
#According to our model, this customer will not buy our car. (predict=0)
#We use sc.transform for specific data of scaled values, that is, for manual entries.

y_pred=classifier.predict(X_test)
concatenateArray=(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
accuracy=0
for i,j in zip(concatenateArray[:,0],concatenateArray[:,1]):
    print("Predict:{}\t Test:{}".format(i,j))
    if (i==j):
        accuracy+=1

print("%",accuracy/len(y_pred)*100,"accuracy ")

#Here, we write our predictions to our predict command for the test set that we have allocated 100 of 400 customers.
#Also, here we combined test and predict array and printed the result with loop.
#We don't need to sc.transform when creating the #predict array.

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

#[65 3] --> There are 68 customers in our dataset who did not buy our car, so purchased=0. However, our model predicted the behavior of 3 customers incorrectly, so those 3 customers should have bought the car.
#[8 24] --> There are 32 customers who bought our car in our dataset, so purchased=1. However, our model predicted the behavior of 8 customers incorrectly, so those 8 customers should not have bought the car.
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
plt.title('Logistic Regression (Training set)')
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
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25), np. arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25)) plt.contourf(X1, X2, classifier.predict(sc) .transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green') )) plt.xlim(X1.min(), X1.max()) plt.ylim(X2.min(), X2.max())
# With these codes, we define two regions and divide these regions with our Classifier, plt.contourf helps this, and we separate the regions as red and green.
# for i, j in enumerate(np.unique(y_set)): plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', ' green'))(i), label = j)
# If this code is actually the most important, what we need to understand from this code is: If there is a point in the red area that should not be, show it in green, if there is a point in the green area that should not be, show it in red. By taking it to the for set, we have the whole set checked.
# The graphic tells us a lot, if you look carefully, there are 3 red dots in the green zone, and there were 3 wrong decisions in the confusion matrix in the model's estimation. These 3 customers should have bought the car (they are in the green zone) but did not buy the car (red dot).
# Likewise, there are 8 green dots in the red zone, and 8 wrong decisions in the confusion matrix. These 8 customers should not have bought the car (they are in the red area) but they did buy the car (green dots).
# Finally, the logistic regression is a linear classifier. If we look at the graph, the classifier is linear that separates the regions, but in real life it is not always possible to make such a linear distinction.



