# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 08:40:04 2020

@author: Abdelrahman
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


df =  pd.read_csv("Placement_Data_Full_Class.csv",index_col="sl_no")
le = LabelEncoder()
df.status = le.fit_transform(df.status)
df.workex = le.fit_transform(df.workex)
df.drop(["gender","ssc_b","hsc_b","hsc_s","degree_t","specialisation","salary"],axis=1,inplace=True)

X = df.iloc[:, :-1]
y = df.status

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = sc_x.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state=0)

from sklearn.ensemble import RandomForestClassifier
rfr = RandomForestClassifier()
rfr.fit(X_train,y_train)
y_pred1 = rfr.predict(X_test)

print(rfr.score(X_test, y_test))

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred2 = xgb.predict(X_test)

print(xgb.score(X_test, y_test))

"""from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred1))
print(mean_squared_error(y_test,y_pred2))"""