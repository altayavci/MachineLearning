#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 16:25:50 2021

@author: altayavci
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values
y=y.reshape(len(y),1)

#Burda reshape'i x ile aynı boyutlarda olması için yaptık.

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#Preprocessing yaparken işlemiştik bunu, SVR kullanacaksak scaling yapmadan kullanmamalıyız.
#Veriler zaten grup ve uzak haldeyken, scaling yapmamak manasız olacaktır.
#StandartScaler() ile değerleri scale ediyoruz, daha sonra fit_transform ile scale edilmiş X ve y değerlerini yeni X ve y değerleri olarak atıyoruz.


regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

#SVR modeli oluşturmak dünyanın en kolay işlerinden biri gördüğün gibi, işin matematiği karışık o yüzden çok girmeme gerek yok fakat kernel = "rbf" en çok kullanılan fonksiyondur
#Kernelin çeşitleri var: Linear kernel, Polynomial kernel, RBF kernel ....
#RBF en çok tercih edileni ve genelde en başarılı olanı; o yüzden biz de onu kullanıyoruz.
#RBF = Gaussian Radial Basis Function


print(regressor.predict((sc_X.transform([[6.5]]))))
print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]]))))

# En çok dikkat edilmesi ve anlaşılması gereken kodun alttaki ikili olduğunu düşünüyorum.
# İlk kodda: Klasik şekilde .predict kullanarak regresyon modelinde X=6.5 olursa y değeri kaç olur onu görmek istiyoruz.
# [[ iki tane bracket kullanmamızın sebebi 2d array şeklinde tanımlamak.
# İyi güzel ama diğer regresyonlarda direkt 6.5 yazabiliyorduk bunda neden sc_X.transform kullandık, çünkü biz X ve y değerlerimizi yukarıda scale etmiştik; bundan mütevellit 6.5 değerini de scale ederek predict etmemiz gerekiyor.

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



