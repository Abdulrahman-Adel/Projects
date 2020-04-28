# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 03:57:36 2020

@author: Abdelrahman
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_csv = pd.read_csv("Restaurant_Reviews.tsv",delimiter  = "\t", quoting = 3) 

#cleaning the text
import re
import nltk
#nltk.download("stopwords")
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from nltk.corpus import stopwords

corpus = []
for i in range(0,1000):
    reviews = re.sub("[^a-zA-Z]"," ",df_csv["Review"][i])
    reviews = reviews.lower()
    reviews = reviews.split()
    reviews = [ps.stem(w) for w in reviews if not w in set(stopwords.words("english"))]
    reviews = " ".join(reviews)
    corpus.append(reviews)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)    
X = cv.fit_transform(corpus).toarray()
#Note: we can clean the text by using the parameters from "cv" 
y = df_csv.Liked

"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = sc_x.fit_transform(X)"""

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state=0)

from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train,y_train)
y_pred = NB.predict(X_test)

from sklearn.metrics import confusion_matrix
cm  = confusion_matrix(y_test,y_pred)
print((67+113)/200)

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train,y_train)
y_pred2 = xgb.predict(X_test)

cm2  = confusion_matrix(y_test,y_pred2)
print((109+65)/200)
