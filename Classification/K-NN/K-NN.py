#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 20:58:58 2021

@author: altayavci
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from matplotlib.colors import ListedColormap

dataset=pd.read_csv("Social_Network_Ads.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#K-NN (K-Nearest Neighbors) is a model used for both regression and classification, but it is a bit lazy and not ideal method.
#It appeared in 1951 and has some unique features. If you look at the picture I sent you in the mail, you will have an idea.
#K-NN basically calculates the nearest neighbors of the points using a certain calculation method (default: Minkowski) and positions the new data point relative to its nearest neighbor.
#So, according to the determined k value, our new data point is classified according to the characteristics of the k points closest to it.
# I found an example on the internet for explanatory purposes:
#A spam filter for #e-mails is being developed. Suppose we can distinguish spam emails by simply checking their headers. Create a simple dataset by manually labeling enough emails as spam and non-spam.
#What we need to do is: Compare each new incoming email with all the emails in the dataset and find which ones are most similar. The same label should be accepted for the incoming e-mail if the majority of k of those adjacent to the incoming e-mail, that is, most similar to it, are labeled as whatever (spam or non-spam). In this way, it is determined whether each new e-mail is spam or not.
#We can briefly summarize the algorithm as follows:
#1- To determine the k value (default=1), that is, to classify according to the nearest 1 point. 2- Determining the neighbors in the data set (distance calculation, minkowski, euclidean) 3- Determining the properties of the k nearest neighbors 4- Classifying the new point according to the majority of the k neighbors

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/4,random_state=0)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#n_neighbors = This is the "k" value I mentioned at the beginning, meaning it will consider the 5 closest points when defining the property of the new point.
#For example, if 3 of 5 points are "0" and 2 are "1", the property of the new point will be "0" (3>2). Its default value is 5. I leave the accuracy score below for k values less than 5 for you.
#k=1 accuracy= 0.87
#k=2 accuracy= 0.90
#k=3 accuracy= 0.93
#k=4 accuracy= 0.92
#metric = This is the code we enter with which method we will calculate the distance of the neighbors. Its default is "minkowski".
#p= Indicates the power value of Minkowski. In order to use the Euclidean calculation that we used in high school and university, we need to make the exponential 2, dear. So right now our code calculates the distance between neighbors with the Euclidean method.

classifier = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p = 2)
classifier.fit(X_train,y_train)

classifier.predict(sc.transform([[30,87000]]))

y_pred=classifier.predict(X_test)

np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)

cm=confusion_matrix(y_pred,y_test)
print(cm)
print(accuracy_score(y_pred,y_test))

#[64 3] --> There are 67 customers in our test set who did not buy our car, so purchased=0, but 3 customers should have bought our car according to our model.
#[4 29] --> In our test set, there are 33 customers who bought our car, so purchased=1, but according to our model, 4 customers should not have bought our car.
#Accuracy Score = (64+29) / 100 = 0.93
#TRAIN SET 

X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#TEST SET 

X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#If you remember, our classifier line was linear in logistic regression, but not so in K-NN. We achieved a better result than logistic regression both in terms of Accuracy score and graphically. (For this dataset)


