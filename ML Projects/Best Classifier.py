# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 13:28:07 2019

@author: Abdelrahman
"""

import pandas as pd
from sklearn import preprocessing

test_df = pd.read_csv("loan_train.csv")

test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_Feature = test_df[['Principal','terms','age','Gender','weekend']]
test_Feature = pd.concat([test_Feature,pd.get_dummies(test_df['education'])], axis=1)
test_Feature.drop(['Master or Above'], axis = 1,inplace=True)
x = test_Feature
x = preprocessing.StandardScaler().fit(x).transform(x)
y = test_df['loan_status']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.svm import SVC
svm = SVC(kernel = "linear")
svm.fit(x,y)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7 )
knn.fit(x,y)

from sklearn.linear_model import LogisticRegression
log = LogisticRegression(C = 0.01,solver = "saga", random_state = 0)
log.fit(x,y)
y_pred = knn.predict(x)

from sklearn.metrics import accuracy_score
print(accuracy_score(y,y_pred))

from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
j_svm = jaccard_similarity_score(y, y_pred)
f1_svm = f1_score(y,y_pred)
print(j_svm,f1_svm )