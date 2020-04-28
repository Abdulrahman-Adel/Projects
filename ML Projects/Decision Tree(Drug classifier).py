# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 22:42:33 2019

@author: Abdelrahman
"""

import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv("drug200.csv")
x = dataset[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = dataset["Drug"]

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
x.iloc[:,1] = le_sex.transform(x.iloc[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
x.iloc[:,2] = le_BP.transform(x.iloc[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
x.iloc[:,3] = le_Chol.transform(x.iloc[:,3]) 

from sklearn.model_selection import train_test_split
x_trainset, x_testset, y_trainset, y_testset = train_test_split(x, y, test_size=0.3, random_state=3)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(x_trainset,y_trainset)
predTree = drugTree.predict(x_testset)

from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

dot_data = StringIO()
filename = "drugtree.png"
featureNames = dataset.columns[0:5]
targetNames = dataset["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')










