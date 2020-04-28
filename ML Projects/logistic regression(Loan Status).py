# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 22:58:09 2019

@author: Abdelrahman
"""

import pandas as pd
from sklearn import preprocessing 

df = pd.read_csv("ChurnData.csv")
df["churn"] = df["churn"].astype(int)
x = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']]
y = df['churn']

x = preprocessing.StandardScaler().fit(x).transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='liblinear').fit(x_train,y_train)
yhat = LR.predict(x_test)
yhat_prob = LR.predict_proba(x_test)

from sklearn.metrics import jaccard_similarity_score
print(jaccard_similarity_score(y_test, yhat))

