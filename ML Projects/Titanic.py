# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 18:43:33 2019

@author: Abdelrahman
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df = pd.read_csv("train.csv")
y = df.Survived
X = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]#,"Cabin"]]
df_test = pd.read_csv("test.csv")
X_test = df_test[["Pclass","Sex","Age","Fare","SibSp","Parch","Embarked"]]#,"Cabin"]]

numerical_cols = [col for col in X.columns if X[col].dtype in ["int64","float64"]]
categorical_cols = [col for col in X.columns if X[col].dtype =="object"]

numerical_transformer = SimpleImputer(strategy = "mean")
categorical_transformer = Pipeline(steps = [("imputer",SimpleImputer(strategy = "constant")),("onehot",OneHotEncoder(handle_unknown = "ignore"))])
preprocessor = ColumnTransformer(transformers = [("num",numerical_transformer,numerical_cols),("cat",categorical_transformer,categorical_cols)])

from xgboost import XGBClassifier
model = XGBClassifier(n_estimators = 1000,learning_rate = 0.05)

my_pipeline = Pipeline(steps = [("preprocessor",preprocessor),("model",model)])
my_pipeline.fit(X,y)
predictions = my_pipeline.predict(X_test)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(my_pipeline,X,y,cv = 5 , scoring = "accuracy")
print(scores)
print(scores.mean())

output = pd.DataFrame({"PassengerId":df_test.PassengerId,"Survived":predictions})
output.to_csv("My_Sub6.csv",index= False)


#Extra Trial
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = "entropy",random_state = 0)

my_pipeline = Pipeline(steps = [("preprocessor",preprocessor),("model",model)])
my_pipeline.fit(X,y)
predictions = my_pipeline.predict(X_test)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(my_pipeline,X,y,cv = 5 , scoring = "accuracy")
print(scores)
print(scores.mean())
