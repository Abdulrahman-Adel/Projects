# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:29:55 2020

@author: Abdelrahman
"""

import pandas as pd 
import nltk
from nltk.stem.porter import PorterStemmer
import re
import unicodedata
from nltk.stem import WordNetLemmatizer
from nltk.corpus import names, stopwords

df = pd.read_csv("train.csv")
nltk.download("names")
from nltk.corpus import stopwords
ps = PorterStemmer()

df['keyword'].fillna(' ', inplace=True)
df['location'].fillna(' ', inplace=True)
df['text'] = df['text'] +' '+ df['location'] +' '+ df['keyword']

lemmetizer = WordNetLemmatizer()
all_names = set(names.words())
stop_words = set(stopwords.words('english'))

def cleaning(h):
    corpus = []
    for i in range(len(h["text"])):
        string = re.sub(r'\d', '', h["text"][i])
        # Removing accented data
        string = unicodedata.normalize('NFKD', h["text"][i]).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        # Removing Mentions
        string = re.sub(r'@\w+', ' ', h["text"][i])
        # Removing links 
        string = re.sub(r'(https?:\/\/)?([\da-zA-Z\.-\/\#\:]+)\.([\da-zA-Z\.\/\:\#]{0,9})([\/\w \.-\/\:\#]*)', ' ', h["text"][i])
        # Removing all the digits special caharacters
        string = re.sub(r'\W', ' ', h["text"][i])
        # Removing double whitespaces
        string = re.sub(r'\s+', ' ', h["text"][i], flags=re.I)
        string = string.strip()
        #Removing all Single characters
        string = re.sub(r'\^[a-zA-Z]\s+','' , string)
        # Lemmetizing the string and removing stop words
        string = string.split()
        string = [lemmetizer.lemmatize(word) for word in string if word not in stop_words and word not in all_names]
        string = ' '.join(string)
        # Lowercasing all data
        string = string.lower()
        corpus.append(string)
        
    return corpus

nltk.download("wordnet")
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf = TfidfVectorizer(min_df=0.1, max_df=0.7)
X = tf_idf.fit_transform(cleaning(df)).toarray()
y = df.target     

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X, y)

df_test = pd.read_csv("test.csv") 
X_test = tf_idf.fit_transform(cleaning(df_test)).toarray()    

y_pred = LR.predict(X_test)
k = pd.Series(data = y_pred)
out = pd.concat([df_test["id"],k],axis = 1,names = ["id","target"])
out.rename(columns = {0:"target"},inplace= True)
out.to_csv("output4.csv")