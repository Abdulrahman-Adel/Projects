# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:32:15 2019

@author: Abdelrahman
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("Mall_Customers.csv")
X = df.iloc[:, 3:]

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = "k-means++",random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("The Elbow method") 
plt.show()

kmeans = KMeans(n_clusters = 5, init = "k-means++",random_state = 0)
kmeans.fit(X)   
y_means = kmeans.predict(X)