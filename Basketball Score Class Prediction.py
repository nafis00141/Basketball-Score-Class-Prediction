# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 00:15:18 2017

@author: NafiS
"""

from sklearn import metrics
import sklearn
from sklearn import datasets
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv('basketball.csv')

data["Class"].unique()
sns.countplot(x="Class",data=data)

print(data.info())

data.drop('Player Name',axis=1, inplace=True)
data.drop('own_team',axis=1, inplace=True)
data.drop('opponent_team',axis=1, inplace=True)
data.drop('Point',axis=1, inplace=True)

y = data['Class']

data.drop('Class',axis=1, inplace=True)

X = data

print(data.info())


clf = RandomForestClassifier()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

scaler = preprocessing.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf.fit(X_train, y_train)
    
y_pred = clf.predict(X_test)

acc = metrics.accuracy_score(y_test, y_pred)
pre = metrics.precision_score(y_test, y_pred,average='macro')
rec = metrics.recall_score(y_test, y_pred,average='macro')
f1 = metrics.f1_score(y_test, y_pred,average='macro')
mse = metrics.mean_squared_error(y_test, y_pred)


print("FOR Random Forest Classifier:")

print ("    Accuracy : ",acc)
print ("    Precision : ",pre)
print ("    Recall : ",rec)
print ("    F1 : ",f1)
print ("    Mean Squred error : ",mse)
print ("     ")
print ("     ")


clf2 = SGDClassifier()

clf2.fit(X_train, y_train)

y_pred = clf2.predict(X_test)

acc = metrics.accuracy_score(y_test, y_pred)
pre = metrics.precision_score(y_test, y_pred,average='macro')
rec = metrics.recall_score(y_test, y_pred,average='macro')
f1 = metrics.f1_score(y_test, y_pred,average='macro')
mse = metrics.mean_squared_error(y_test, y_pred)

print("FOR SGD Classifier:")

print ("    Accuracy : ",acc)
print ("    Precision : ",pre)
print ("    Recall : ",rec)
print ("    F1 : ",f1)
print ("    Mean Squred error : ",mse)
print ("     ")
print ("     ")

clf3 = svm.SVC(kernel='linear',C=0.4)

clf3.fit(X_train, y_train)

y_pred = clf3.predict(X_test)

acc = metrics.accuracy_score(y_test, y_pred)
pre = metrics.precision_score(y_test, y_pred,average='macro')
rec = metrics.recall_score(y_test, y_pred,average='macro')
f1 = metrics.f1_score(y_test, y_pred,average='macro')
mse = metrics.mean_squared_error(y_test, y_pred)

print("FOR SVC Classifier:")

print ("    Accuracy : ",acc)
print ("    Precision : ",pre)
print ("    Recall : ",rec)
print ("    F1 : ",f1)
print ("    Mean Squred error : ",mse)
print ("     ")
print ("     ")