# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
print(X)

"""Determining the optimal number of clusters using the Elbow method

We calculate the wcss values of the cluster numbers from 1 to 10 using for (wcss.append(k.means.inertia_)) and then plot them with plt.plot().

The value "5", the point where the graph bends, represents the optimal number of clusters for our dataset.
"""

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

"""We create our model and train it with our dataset. 
We create our model, predict with our X values and assign these values to the y_kmeans object. """

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)

"""Graphic

It takes values as plt.scatter(X,y,......), so first of all we must carefully determine our X and y values.
Our x-value will be our customers' annual income, while our y-value will be their spending scores.

E.g:

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],......)

Represents the scatter plot of customers in the first set.
X[y_kmeans==0, 0] : It represents the annual income values of the customers in the first cluster with the cluster value "0".

X[y_kmeans == 0, 1] : It represents the spending score of the customers in the first cluster with the cluster value "0".

!! X[:,0] : Annual Income
!! X[:,1] : Spending Score

The s value determines the size of the scatter points, and the c color determines.

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

The code above is the code we use to draw the cluster centers.
"""

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()