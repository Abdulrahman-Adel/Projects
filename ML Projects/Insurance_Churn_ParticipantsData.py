# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 19:38:16 2020

@author: Abdelrahman
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("train.csv")
X_test = pd.read_csv("test.csv")

X = df.iloc[:, :-1]
y = df.labels

sc_x = StandardScaler()
X = sc_x.fit_transform(X)
sc_x_test = StandardScaler()
X_test = sc_x_test.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
le = LogisticRegression()
le.fit(X,y)
y_pred1 = le.predict(X_test)

out1 = pd.Series(data = y_pred1 ,name = "labels")


from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X,y)
y_pred2 = xgb.predict(X_test)

out2 = pd.Series(data = y_pred2 ,name = "labels")

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 1000 ,criterion = "entropy")
rfc.fit(X,y)
y_pred3 = rfc.predict(X_test)

out3 = pd.Series(data = y_pred3 ,name = "labels")


out1.to_excel("output1.xlsx")
out2.to_excel("output2.xlsx")
out3.to_excel("output3.xlsx")

from sklearn.svm import SVC
svc = SVC()
svc.fit(X,y)
y_pred4 = svc.predict(X_test)

out4 = pd.Series(data = y_pred4 ,name = "labels")

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(X,y)
y_pred5 = KNN.predict(X_test)

out5 = pd.Series(data = y_pred5 ,name = "labels")

out4.to_excel("output4.xlsx")
out5.to_excel("output5.xlsx")
out3.to_excel("output3.xlsx")

