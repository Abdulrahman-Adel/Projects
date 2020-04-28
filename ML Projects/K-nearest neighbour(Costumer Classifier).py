# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 21:57:36 2019

@author: Abdelrahman
"""

import pandas as pd
from sklearn import preprocessing

dataset = pd.read_csv("teleCust1000t.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 11].values

x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)

from sklearn.neighbors import KNeighborsClassifier
k = 6
neigh = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
yhat = neigh.predict(x_test)

from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(x_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))