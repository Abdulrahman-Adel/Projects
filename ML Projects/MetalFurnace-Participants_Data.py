# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 08:08:30 2020

@author: Abdelrahman
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


df =  pd.read_csv("Placement_Data_Full_Class.csv",index_col="sl_no")
y = df.status
df.fillna(0,inplace=True)
df.drop(["status"],axis=1,inplace=True)
x = df

def le_ohe(z):
    le  = LabelEncoder()
    ohe = OneHotEncoder()
    x[z] = le.fit_transform(x[z])
    x[z] = ohe.fit_transform(x[z].values.reshape(-1,1)).toarray()
    

categorical_cols = ["gender","ssc_b","hsc_s","degree_t","workex","etest_p","specialisation","hsc_b"]
for col in categorical_cols:
    le_ohe(col)
        
le  = LabelEncoder()
#ohe = OneHotEncoder()
y = le.fit_transform(y)
#y = ohe.fit_transform(y.reshape(-1,1)).toarray()   


from sklearn.preprocessing import StandardScaler
sc_x  = StandardScaler()
x = sc_x.fit_transform(x)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state=0)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100,criterion="entropy")
rfc.fit(x_train,y_train)
y_pred = rfc.predict(x_test)

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x_train,y_train)
y_pred2 = xgb.predict(x_test)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred2))

from sklearn.metrics import log_loss
print(log_loss(y_test,y_pred))
print(log_loss(y_test,y_pred2))