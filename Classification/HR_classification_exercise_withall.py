
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("HR_comma_sep.csv")
X = dataset.iloc[:,[0,1,2,3,4,5,7,8,9]].values
y = dataset.iloc[:,-4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_7 = LabelEncoder()
X[:, 7] = labelencoder_X_7.fit_transform(X[:, 7])
labelencoder_X_8 = LabelEncoder()
X[:, 8] = labelencoder_X_8.fit_transform(X[:, 8])

X = np.array(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,0:5] = sc.fit_transform(X_train[:,0:5])
X_test[:,0:5] = sc.transform(X_test[:,0:5])

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
cm = confusion_matrix(y_pred,y_test)
ac = accuracy_score(y_pred,y_test)
print(cm)
print(ac)