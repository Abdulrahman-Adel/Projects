# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 21:15:31 2019

@author: Abdelrahman
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("FuelConsumptionCo2.csv")
x = dataset[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_CITY"]]
y = dataset[["CO2EMISSIONS"]]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print("Coeffcients:",reg.coef_)

plt.scatter(x_train.ENGINESIZE,y_train,color = "blue")
plt.show()

