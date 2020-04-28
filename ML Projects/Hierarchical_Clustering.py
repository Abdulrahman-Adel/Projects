# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:22:15 2019

@author: Abdelrahman
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("Mall_Customers.csv")
X  = df.iloc[:, 3:]

import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method = "ward"))
plt.title("dendogram")
plt.xlabel("Cutomers")
plt.ylabel("Eucledian Distance")
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5)
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()