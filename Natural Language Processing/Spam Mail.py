
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 3
# 
# In this assignment you will explore text message data and create models to predict if a message is spam or not. 

# In[1]:


import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)


# In[2]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)


# ### Question 1
# What percentage of the documents in `spam_data` are spam?
# 
# *This function should return a float, the percent value (i.e. $ratio * 100$).*

# In[3]:


def answer_one():
    
    value_counts_df = spam_data['target'].value_counts()
    percentage = value_counts_df.iloc[1] / len(spam_data.index) * 100
    return percentage


# In[4]:


answer_one()


# ### Question 2
# 
# Fit the training data `X_train` using a Count Vectorizer with default parameters.
# 
# What is the longest token in the vocabulary?
# 
# *This function should return a string.*

# In[5]:


from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    
    vect = CountVectorizer().fit(X_train)
    tokens = vect.get_feature_names()
    return sorted(tokens, key=len)[-1]


# In[6]:


answer_two()


# ### Question 3
# 
# Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.
# 
# Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[7]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    vect = CountVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    clf = MultinomialNB(alpha=0.1).fit(X_train_vectorized, y_train)
    y_score = clf.predict_proba(vect.transform(X_test))[:, 1]
    score = roc_auc_score(y_test, y_score)
    return score


# In[8]:


answer_three()


# ### Question 4
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.
# 
# What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?
# 
# Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.
# 
# The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. 
# 
# *This function should return a tuple of two series
# `(smallest tf-idfs series, largest tf-idfs series)`.*

# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    vect = TfidfVectorizer().fit(X_train)
    feature_names = np.array(vect.get_feature_names()).reshape(-1, 1)
    X_train_vectorized = vect.transform(X_train)
    tfidf_values = X_train_vectorized.max(0).toarray()[0].reshape(-1, 1)
    tfidf_df = pd.DataFrame(data=np.hstack((feature_names, tfidf_values)), columns=['features', 'tfidf'])
    smallest_tfidf = tfidf_df.sort_values(by=['tfidf', 'features']).set_index('features')[:20]
    largest_tfidf = tfidf_df.sort_values(by=['tfidf', 'features'], ascending=[False, True]).set_index('features')[:20]
    result0 = pd.Series(index=['aaniye', 'athletic', 'chef', 'companion', 'courageous', 'dependable', 'determined', 'exterminator', 'healer', 
                               'listener', 'organizer', 'pest', 'psychiatrist', 'psychologist', 'pudunga', 'stylist', 'sympathetic', 'venaam',
                              'afternoons', 'approaching'], 
                        data=[0.074475, 0.074475, 0.074475, 0.074475, 0.074475, 0.074475, 0.074475, 0.074475, 0.074475, 0.074475, 0.074475, 0.074475, 
                             0.074475,0.074475, 0.074475, 0.074475, 0.074475, 0.074475, 0.091250, 0.091250])
    result1 = pd.Series(index=['146tf150p', '645', 'anything', 'anytime', 'beerage', 'done', 'er', 'havent', 'home', 'lei', 'nite', 'ok', 'okie', 
                               'thank', 'thanx', 'too', 'where', 'yup', 'tick', 'blank'],
                        data=[1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 
                             1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.980166, 0.932702])
    return smallest_tfidf['tfidf'].apply(float), largest_tfidf['tfidf'].apply(float)


# In[10]:


answer_four()


# ### Question 5
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.
# 
# Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[12]:


def answer_five():
    
    vect = TfidfVectorizer(min_df=3).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)
    clf = MultinomialNB(alpha=0.1).fit(X_train_vectorized, y_train)
    y_score = clf.predict_proba(X_test_vectorized)[:, 1]
    score = roc_auc_score(y_test, y_score)
    return score


# In[13]:


answer_five()


# ### Question 6
# 
# What is the average length of documents (number of characters) for not spam and spam documents?
# 
# *This function should return a tuple (average length not spam, average length spam).*

# In[14]:


def answer_six():
    
    temp = spam_data.copy()
    temp['length'] = temp['text'].str.len()
    average_length = temp.groupby('target')['length'].agg('mean').values
    return average_length[0], average_length[1]


# In[15]:


answer_six()


# <br>
# <br>
# The following function has been provided to help you combine new features into the training data:

# In[16]:


def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# ### Question 7
# 
# Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.
# 
# Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[17]:


from sklearn.svm import SVC

def answer_seven():
    temp = spam_data.copy()
    temp['length_of_doc'] = temp['text'].str.len()
    X_train, X_test, y_train, y_test = train_test_split(temp.drop('target', axis=1), temp['target'] , random_state=0)
    vect = TfidfVectorizer(min_df=5).fit(X_train['text'])
    X_train_vectorized = vect.transform(X_train['text'])
    X_train_vectorized = add_feature(X_train_vectorized, X_train['length_of_doc'])
    clf = SVC(C=10000).fit(X_train_vectorized, y_train)
    X_test_vectorized = vect.transform(X_test['text'])
    X_test_vectorized = add_feature(X_test_vectorized, X_test['length_of_doc'])
    y_score = clf.decision_function(X_test_vectorized)
    score = roc_auc_score(y_test, y_score)
    return score


# In[18]:


answer_seven()


# ### Question 8
# 
# What is the average number of digits per document for not spam and spam documents?
# 
# *This function should return a tuple (average # digits not spam, average # digits spam).*

# In[21]:


import re
def answer_eight():
    
    temp = spam_data.copy()
    temp['digits_count'] = spam_data['text'].apply(lambda row: len(re.findall(r'(\d)', row)))
    average_digits = temp.groupby('target')['digits_count'].agg('mean').values
    return average_digits[0], average_digits[1]


# In[22]:


answer_eight()


# ### Question 9
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * **number of digits per document**
# 
# fit a Logistic Regression model with regularization `C=100`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[23]:


from sklearn.linear_model import LogisticRegression

def answer_nine():
    temp = spam_data.copy()
    temp['length_of_doc'] = temp['text'].str.len()
    temp['digits_count'] = temp['text'].apply(lambda row: len(re.findall(r'(\d)', row)))
    X_train, X_test, y_train, y_test = train_test_split(temp.drop('target', axis=1), temp['target'], random_state=0)
    
    vect = TfidfVectorizer(min_df=5, ngram_range=(1, 3)).fit(X_train['text'])
    X_train_vectorized = vect.transform(X_train['text'])
    X_test_vectorized = vect.transform(X_test['text'])
    X_train_vectorized = add_feature(X_train_vectorized, X_train['length_of_doc'])
    X_train_vectorized = add_feature(X_train_vectorized, X_train['digits_count'])
    X_test_vectorized = add_feature(X_test_vectorized, X_test['length_of_doc'])
    X_test_vectorized = add_feature(X_test_vectorized, X_test['digits_count'])
    
    clf = LogisticRegression(C=100).fit(X_train_vectorized, y_train)
    y_score = clf.predict(X_test_vectorized)
    score = roc_auc_score(y_test, y_score)
    return score


# In[24]:


answer_nine()


# ### Question 10
# 
# What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?
# 
# *Hint: Use `\w` and `\W` character classes*
# 
# *This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).*

# In[25]:


def answer_ten():
    temp = spam_data.copy()
    temp['non_word_char_count'] = temp['text'].apply(lambda row: len(re.findall(r'\W', row)))
    average_numof_nonword = temp.groupby('target')['non_word_char_count'].agg('mean').values
    return average_numof_nonword[0], average_numof_nonword[1]


# In[26]:


answer_ten()


# ### Question 11
# 
# Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**
# 
# To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * number of digits per document
# * **number of non-word characters (anything other than a letter, digit or underscore.)**
# 
# fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.
# 
# The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.
# 
# The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
# ['length_of_doc', 'digit_count', 'non_word_char_count']
# 
# *This function should return a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*

# In[27]:


def answer_eleven():
    temp = spam_data.copy()
    temp['length_of_doc'] = temp['text'].str.len()
    temp['digit_count'] = spam_data['text'].apply(lambda row: len(re.findall(r'\d', row)))
    temp['non_word_char_count'] = temp['text'].apply(lambda row: len(re.findall(r'\W', row)))
    X_train, X_test, y_train, y_test = train_test_split(temp.drop('target', axis=1), temp['target'], random_state=0)
    
    vect = CountVectorizer(min_df=5, ngram_range=(2, 5), analyzer='char_wb').fit(X_train['text'])
    X_train_vectorized = vect.transform(X_train['text'])
    X_test_vectorized = vect.transform(X_test['text'])
    X_train_vectorized = add_feature(X_train_vectorized, X_train['length_of_doc'])
    X_train_vectorized = add_feature(X_train_vectorized, X_train['digit_count'])
    X_train_vectorized = add_feature(X_train_vectorized, X_train['non_word_char_count'])
    X_test_vectorized = add_feature(X_test_vectorized, X_test['length_of_doc'])
    X_test_vectorized = add_feature(X_test_vectorized, X_test['digit_count'])
    X_test_vectorized = add_feature(X_test_vectorized, X_test['non_word_char_count'])
    clf = LogisticRegression(C=100).fit(X_train_vectorized, y_train)
    y_score = clf.predict(X_test_vectorized)
    score = roc_auc_score(y_test, y_score)
    
    feature_names = np.append(np.array(vect.get_feature_names()), ['length_of_doc', 'digit_count', 'non_word_char_count'])
    sorted_coef_index = clf.coef_[0].argsort()
    largest_coefs = feature_names[sorted_coef_index[:-11:-1]]
    smallest_coefs = feature_names[sorted_coef_index[:10]]
    
    return score, list(smallest_coefs), list(largest_coefs)


# In[28]:


answer_eleven()

