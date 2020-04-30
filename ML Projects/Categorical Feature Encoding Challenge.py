# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:18:02 2019

@author: Abdelrahman
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("train.csv")
X_test = pd.read_csv("test.csv")

X = df.iloc[:, :-1]
y = df.target

le = LabelEncoder()
ohe = OneHotEncoder()

def categorical_col(Z):
    low_cardinality_cols = [col for col in Z.columns if Z[col].dtype =="object" and Z[col].nunique() < 3]
    mid_cardinality_cols = [col for col in Z.columns if Z[col].dtype =="object" and Z[col].nunique() < 11]
    high_cardinality_cols = [col for col in Z.columns if Z[col].dtype =="object" and Z[col].nunique() > 10]
    Z.drop(high_cardinality_cols,axis = 1,inplace=True)
    categorical_cols = low_cardinality_cols + mid_cardinality_cols
    return categorical_cols
def categorical_encoding(M,H):
    for col in M:
        H[col] = le.fit_transform(H[col])
        H[col] = ohe.fit_transform(H[col].values.reshape(-1,1)).toarray()
        

categorical_encoding(categorical_col(X),X)
categorical_encoding(categorical_col(X_test),X_test)

from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X.iloc[:, 1:],y)

y_pred = NB.predict(X_test.iloc[:, 1:])

out = pd.DataFrame({"id":X_test.id,"target":y_pred})
out.to_csv("output1.csv")

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X.iloc[:, 1:],y)

y_pred2 = rfc.predict(X_test.iloc[:, 1:])

out2 = pd.DataFrame({"id":X_test.id,"target":y_pred2})
out2.to_csv("output2.csv")



        
        