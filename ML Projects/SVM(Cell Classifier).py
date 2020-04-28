# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:39:13 2019

@author: Abdelrahman
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 

cell_df = pd.read_csv("cell_samples.csv")
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

from sklearn import svm
clf = svm.SVC(kernel='rbf',gamma = "auto")
clf.fit(X_train, y_train) 
yhat = clf.predict(X_test)