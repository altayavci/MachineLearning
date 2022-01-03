#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 17:33:33 2021

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

class KNN:
    def __init__(self,X_train,X_test,y_train,y_test):
   
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
    
        
    def KNeighborMaxFinder(self):

        dictCollector=dict()
        for i in range(1,100): 
            
                for j in range(1,10):
                    
                    classifier = KNeighborsClassifier(n_neighbors =i, metric = "minkowski", p = j)
                    classifier.fit(X_train,y_train)
                    self.y_pred=classifier.predict(X_test)
                    accuracy=accuracy_score(y_test, self.y_pred)
                    dictCollector[i,j]=accuracy
       
        return "For the max (n_neighbors,p) value: {}\tThe max accuracy: {}".format(max(dictCollector, key=dictCollector.get),max(dictCollector.values()))
    
    def yPredict(self):
        
        return self.y_pred
    
    def confusionMatrix(self):
        
        cm=confusion_matrix(y_test,self.y_pred)
        return cm
    

knn=KNN(X_train,X_test,y_train,y_test) 
print(knn.KNeighborMaxFinder())
print(knn.yPredict())
print(knn.confusionMatrix())

    
    
    
    
    