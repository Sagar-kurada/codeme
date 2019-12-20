#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 12:32:41 2019

@author: sagarkurada
"""
#test comment by akshay
#pip xgboost
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Basic Libraries for Data organization, Statistical operations and Plotting
import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib inline
# For loading .arff files
from scipy.io import arff
# To analyze the type of missing data
import missingno as msno
# Library for performing k-NN and MICE imputations
import fancyimpute
# Library to perform Expectation-Maximization (EM) imputation
#import impyute as impy
# To perform mean imputation
from sklearn.preprocessing import Imputer
#To perform kFold Cross Validation
from sklearn.model_selection import KFold
# Formatted counter of class labels
from collections import Counter
# Ordered Dictionarya
from collections import OrderedDict
# Library imbalanced-learn to deal with the data imbalance. To use SMOTE oversampling
from imblearn.over_sampling import SMOTE

# Impoting classification models
#from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
import random

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
 #pip install xgboost
#pip install missingno
#pip install fancyimpute
#pip
from xgboost import XGBClassifier

import os
os.getcwd()

def load_raw_data():
    n=5
    return [arff.loadarff('data/' + str(i+1) + 'year.arff') for i in range(n)]


def convert_in_df():
    return [pd.DataFrame(data_i_year[0]) for data_i_year in load_raw_data()]

def set_headers(dataframes):
    cols = ['X' + str(i+1) for i in range(len(dataframes[0].columns)-1)]
    cols.append('Y')
    for df in dataframes:
        df.columns = cols

dataframes = convert_in_df()

set_headers(dataframes)

dataframes[0].head()

dataframes[0].describe()
dataframes[0]['Y']


n=5
for i in range(n):
    print(dataframes[i].shape)

#dataframes[0].isnull().sum().sort_values()

n=5
for i in range(n):
    print(dataframes[i].isnull().sum().sort_values(ascending=False).head(5))

for i in range(n):
        col = getattr(dataframes[i], 'Y')
        dataframes[i]['Y'] = col.astype(int)

for i in range(n):
        index = 1
        while(index<=63):
            colname = dataframes[i].columns[index]
            col = getattr(dataframes[i], colname)
            dataframes[i][colname] = col.astype(float)
            index+=1

dataframes[0].describe()

n=5
for i in range(n):
    print(dataframes[i]['Y'].value_counts())

for i in range(5):
        missing_df_i = dataframes[i].columns[dataframes[i].isnull().any()].tolist()
        msno.matrix(dataframes[i][missing_df_i], figsize=(20,5))

imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0)
mean_imputed_dfs = [pd.DataFrame(imputer.fit_transform(df)) for df in dataframes]
for i in range(len(dataframes)):
        mean_imputed_dfs[i].columns = dataframes[i].columns


mean_imputed_dfs[0].head()

consolidatedDfs=pd.DataFrame(index=mean_imputed_dfs[0].index,columns=mean_imputed_dfs[0].columns)

#for i in range(5):
consolidatedDfs=pd.concat([mean_imputed_dfs[0],mean_imputed_dfs[1],mean_imputed_dfs[2],mean_imputed_dfs[3],mean_imputed_dfs[4]
])

consolidatedDfs.shape
consolidatedDfs.to_csv('consolidates.csv')

X = consolidatedDfs.iloc[:,:-1]
y = consolidatedDfs.iloc[:,64]

# splitting the set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)
#y_test


X = mean_imputed_dfs[0].iloc[:,:-1]
y = mean_imputed_dfs[0].iloc[:,64]
print(y)
# splitting the set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)
y_test

X_train.shape

threshold = 1e-6

clf1 = LogisticRegression(penalty = 'l1')
clf2 = LogisticRegression(penalty = 'l2')

model_list = (clf1, clf2)

for clf in model_list:
    clf.fit(X_train, y_train)
    feature_weight = clf.coef_
    print("The validation score of model",clf.score(X_test,y_test))
    print("The number of selected feature of model",(np.sum(abs(feature_weight) > threshold)))

 #logistic regression
listC = 10.0**np.arange(-4,4)
parameter = {'C':listC}
lr = LogisticRegression(penalty = 'l1')
clf = GridSearchCV(lr, parameter)

clf.fit(X_train, y_train)
print("The best parameter is",clf.best_params_)
print("The best score is",clf.best_score_)

lr = LogisticRegression(penalty = 'l1', C = 0.001)
lr.fit(X_train, y_train)
print("Training score of Logistic Regression model is",lr.score(X_train, y_train))
print("Testing score of Logistic Regression model is",lr.score(X_test, y_test))


#Decision Tree
maxDepth = 20
kFold = 5
scores = np.zeros((maxDepth, kFold))

for depth in np.arange(1, maxDepth + 1):
    model_Dtree = DecisionTreeClassifier(max_depth=depth)
    scores[depth - 1] = cross_val_score(model_Dtree, X_train,
                                        y_train, cv=kFold)

plt.style.use('ggplot')
plt.errorbar(range(1, maxDepth + 1), np.average(scores, axis=1),
             color='blue', linestyle='--', marker='o', markersize=10,
             yerr=np.std(scores, axis=1), ecolor='pink',
             capthick=2)
plt.xlabel("Maximum tree depth", fontsize = 16)
plt.ylabel("Average accuracy", fontsize = 16)
plt.title("Average accuracy on 10-Fold CV vs. Tree depth",
          fontsize = 20)
plt.gcf().set_size_inches(12, 7)
plt.tight_layout()


trainging_scores = np.zeros((maxDepth, 1))
testing_scores = np.zeros((maxDepth, 1))

for depth in np.arange(1, maxDepth + 1):
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    trainging_scores[depth - 1] = clf.score(X_train, y_train)
    testing_scores[depth - 1] = clf.score(X_test, y_test)

# Plot the results
plt.style.use('ggplot')
plt.plot(range(1, maxDepth + 1), trainging_scores, 'o--',
         markersize=10, color='blue', lw=1, label='Training accuracy')
plt.plot(range(1, maxDepth + 1), testing_scores, 'o--',
         markersize=10, color='green', lw=1, label='Testing accuracy')
plt.xlabel("Maximum tree depth", fontsize = 16)
plt.ylabel("Average accuracy", fontsize = 16)
plt.title("Training and Testing accuracy vs. Tree depth",
          fontsize = 20)
plt.legend(loc="best")
plt.gcf().set_size_inches(12, 8)
plt.tight_layout()


clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)
print("Training score of Decision Tree Classifier is",clf.score(X_train, y_train))
print("Testing score of Decision Tree Classifier is",clf.score(X_test, y_test))

#Guassian Naive Bayes


def prepare_kfold_cv_data(k, X, y, verbose=False):
    X = X.values
    y = y.values
    kf = KFold(n_splits=k, shuffle=False, random_state=42)
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for train_index, test_index in kf.split(X):
        X_train.append(X[train_index])
        y_train.append(y[train_index])
        X_test.append(X[test_index])
        y_test.append(y[test_index])
    return X_train, y_train, X_test, y_test

X_train_list, y_train_list, X_test_list, y_test_list = prepare_kfold_cv_data(5, X_train, y_train)
for k_index in range(5):
                    X_train = X_train_list[k_index]
                    y_train = y_train_list[k_index]
                    X_test = X_test_list[k_index]
                    y_test = y_test_list[k_index]
                    gnb_classifier = GaussianNB()

                    clf_model=gnb_classifier.fit(X_train, y_train)
                    y_test_predicted = clf_model.predict(X_test)
                    print(y_test_predicted,y_test)

                    accuracy_gnb = accuracy_score(y_test, y_test_predicted, normalize=True)
                    recall_gnb = recall_score(y_test, y_test_predicted, average=None)
                    precision_gnb = precision_score(y_test, y_test_predicted, average=None)
                    confusion_matrix_gnb = confusion_matrix(y_test, y_test_predicted)

                    print("Accuracy",accuracy_gnb ,"\n","Precision",precision_gnb,"\n","Recall",recall_gnb)
                    print(confusion_matrix_gnb)
                    confusion_matrix_gnb_df = pd.DataFrame(confusion_matrix_gnb,
                                         index = ['survived','bankrupt'],
                                         columns = ['survived','bankrupt'])

                    plt.figure(figsize=(5.5,4))
                    sns.heatmap(confusion_matrix_gnb_df, annot=True)
                    plt.title('Naive Bayes classifier \nAccuracy:{0:.3f}'.format(accuracy_gnb))
                    plt.ylabel('True label')
                    plt.xlabel('Predicted label')
                    plt.show()



#random forest


#X_train_list, y_train_list, X_test_list, y_test_list = prepare_kfold_cv_data(5, X_train, y_train)
for k_index in range(5):
                    X_train = X_train_list[k_index]
                    y_train = y_train_list[k_index]
                    X_test = X_test_list[k_index]
                    y_test = y_test_list[k_index]
                    rf_classifier = RandomForestClassifier(n_estimators = 5, criterion = 'entropy')
                    clf_model=rf_classifier.fit(X_train, y_train)
                    y_test_predicted = clf_model.predict(X_test)
                    #print(y_test_predicted,y_test)
                    accuracy_rf = accuracy_score(y_test, y_test_predicted, normalize=True)
                    recall_rf = recall_score(y_test, y_test_predicted, average=None)
                    precision_rf = precision_score(y_test, y_test_predicted, average=None)
                    confusion_matrix_rf = confusion_matrix(y_test, y_test_predicted)
                    print("Accuracy",accuracy_rf ,"\n","Precision",precision_rf,"\n","Recall",recall_rf)
                    print(confusion_matrix_rf)
                    confusion_matrix_rf_df = pd.DataFrame(confusion_matrix_rf,
                                         index = ['survived','bankrupt'],
                                         columns = ['survived','bankrupt'])

                    plt.figure(figsize=(5.5,4))
                    sns.heatmap(confusion_matrix_rf_df, annot=True)
                    plt.title('Random Forest Classifier \nAccuracy:{0:.3f}'.format(accuracy_rf))
                    plt.ylabel('True label')
                    plt.xlabel('Predicted label')
                    plt.show()

#Balanced Bagging classifier


df_bk=pd.read_csv("/Users/sagarkurada/Documents/Courses/Data Mining/mgmt571/bankruptcy_Train.csv")
df_bk_test=pd.read_csv("/Users/sagarkurada/Desktop/bankruptcy_Test_X.csv")
print(df_bk)
X_df_bk = df_bk.iloc[:,0:64]
y_df_bk = df_bk.iloc[:,64]
X_df_bk_test = df_bk_test.iloc[:,0:64]
print(X_df_bk_test)
for k_index in range(5):
                    X_train = X_train_list[k_index]
                    y_train = y_train_list[k_index]
                    X_test = X_test_list[k_index]
                    y_test = y_test_list[k_index]
                    bb_classifier = BalancedBaggingClassifier(base_estimator = RandomForestClassifier(criterion='entropy'), n_estimators = 5, bootstrap = True)
                    clf_model=rf_classifier.fit(X_train, y_train)
                    X_test=X_df_bk_test
                   # print(X_df_bk)
                    y_test_predicted = clf_model.predict(X_test)
                    #y_main=y_df_bk
                    #print(y_test_predicted,y_test)
                    accuracy_bb = accuracy_score(y_main, y_test_predicted, normalize=True)
                    recall_bb = recall_score(y_main, y_test_predicted, average=None)
                    precision_bb = precision_score(y_main, y_test_predicted, average=None)
                    confusion_matrix_bb = confusion_matrix(y_main, y_test_predicted)
                    print("Accuracy",accuracy_bb ,"\n","Precision",precision_bb,"\n","Recall",recall_bb)
                    #print(confusion_matrix_bb)
                    #rint(accuracy_bb)
                    #print(recall_bb)
                    #print(precision_bb)
                    confusion_matrix_bb_df = pd.DataFrame(confusion_matrix_bb,
                                         index = ['survived','bankrupt'],
                                         columns = ['survived','bankrupt'])

                    plt.figure(figsize=(5.5,4))
                    sns.heatmap(confusion_matrix_bb_df, annot=True)
                    plt.title('Balanced Bagging Classifier \nAccuracy:{0:.3f}'.format(accuracy_bb))
                    plt.ylabel('True label')
                    plt.xlabel('Predicted label')
                    plt.show()

X_train = X_train_list[4]
y_train = y_train_list[4]
bb_classifier = BalancedBaggingClassifier(base_estimator = RandomForestClassifier(criterion='entropy'), n_estimators = 5, bootstrap = True)
clf_model=rf_classifier.fit(X_train, y_train)
X_test=X_df_bk_test
# print(X_df_bk)
y_test_predicted = clf_model.predict(X_test)
print(y_test_predicted)
sum(y_test_predicted)



     ####real _code
for k_index in range(5):
                    X_train = X_train_list[k_index]
                    y_train = y_train_list[k_index]
                    X_test = X_test_list[k_index]
                    y_test = y_test_list[k_index]
                    bb_classifier = BalancedBaggingClassifier(base_estimator = RandomForestClassifier(criterion='entropy'), n_estimators = 5, bootstrap = True)
                    clf_model=rf_classifier.fit(X_train, y_train)
                    y_test_predicted = clf_model.predict(X_test)
                    #print(y_test_predicted,y_test)
                    accuracy_bb = accuracy_score(y_test, y_test_predicted, normalize=True)
                    recall_bb = recall_score(y_test, y_test_predicted, average=None)
                    precision_bb = precision_score(y_test, y_test_predicted, average=None)
                    confusion_matrix_bb = confusion_matrix(y_test, y_test_predicted)
                    #print("Accuracy",accuracy_bb ,"\n","Precision",precision_bb,"\n","Recall",recall_bb)
                    print(confusion_matrix_bb)
                    print(accuracy_bb)
                    print(recall_bb)
                    print(precision_bb)
                    confusion_matrix_bb_df = pd.DataFrame(confusion_matrix_bb,
                                         index = ['survived','bankrupt'],
                                         columns = ['survived','bankrupt'])

                    plt.figure(figsize=(5.5,4))
                    sns.heatmap(confusion_matrix_bb_df, annot=True)
                    plt.title('Balanced Bagging Classifier \nAccuracy:{0:.3f}'.format(accuracy_bb))
                    plt.ylabel('True label')
                    plt.xlabel('Predicted label')
                    plt.show()


#XG Boost Claasifier

for k_index in range(5):
                    X_train = X_train_list[k_index]
                    y_train = y_train_list[k_index]
                    X_test = X_test_list[k_index]
                    y_test = y_test_list[k_index]
                    xgb_classifier = XGBClassifier()
                    clf_model=xgb_classifier.fit(X_train, y_train)
                    y_test_predicted = clf_model.predict(X_test)
                    #print(y_test_predicted,y_test)
                    accuracy_xgb = accuracy_score(y_test, y_test_predicted, normalize=True)
                    recall_xgb = recall_score(y_test, y_test_predicted, average=None)
                    precision_xgb = precision_score(y_test, y_test_predicted, average=None)
                    confusion_matrix_xgb = confusion_matrix(y_test, y_test_predicted)
                    print("Accuracy",accuracy_xgb ,"\n","Precision",precision_xgb,"\n","Recall",recall_xgb)
                    print(confusion_matrix_xgb)
                    confusion_matrix_xgb_df = pd.DataFrame(confusion_matrix_xgb,
                                         index = ['survived','bankrupt'],
                                         columns = ['survived','bankrupt'])

                    plt.figure(figsize=(5.5,4))
                    sns.heatmap(confusion_matrix_xgb_df, annot=True)
                    plt.title('XG boost \nAccuracy:{0:.3f}'.format(accuracy_xgb))
                    plt.ylabel('True label')
                    plt.xlabel('Predicted label')
                    plt.show()


df_bk=pd.read_csv("/Users/sagarkurada/Documents/Courses/Data Mining/mgmt571/bankruptcy_Train.csv")
df_bk_test=pd.read_csv("/Users/sagarkurada/Desktop/bankruptcy_Test_X.csv")
print(df_bk)
X_df_bk = df_bk.iloc[:,0:64]
y_df_bk = df_bk.iloc[:,64]
X_df_bk_test = df_bk_test.iloc[:,0:64]

for k_index in range(5):
                    X_train = X_train_list[k_index]
                    y_train = y_train_list[k_index]
                    X_test = X_test_list[k_index]
                    y_test = y_test_list[k_index]
                    xgb_classifier = XGBClassifier()
                    clf_model=xgb_classifier.fit(X_train, y_train)
                    X_test=X_df_bk

                    y_test_predicted = clf_model.predict(X_test)
                    y_test=y_df_bk
                    #print(y_test_predicted,y_test)
                    accuracy_xgb = accuracy_score(y_test, y_test_predicted, normalize=True)
                    recall_xgb = recall_score(y_test, y_test_predicted, average=None)
                    precision_xgb = precision_score(y_test, y_test_predicted, average=None)
                    confusion_matrix_xgb = confusion_matrix(y_test, y_test_predicted)
                    print("Accuracy",accuracy_xgb ,"\n","Precision",precision_xgb,"\n","Recall",recall_xgb)
                    print(confusion_matrix_xgb)
                    confusion_matrix_xgb_df = pd.DataFrame(confusion_matrix_xgb,
                                         index = ['survived','bankrupt'],
                                         columns = ['survived','bankrupt'])

                    plt.figure(figsize=(5.5,4))
                    sns.heatmap(confusion_matrix_xgb_df, annot=True)
                    plt.title('XG boost \nAccuracy:{0:.3f}'.format(accuracy_xgb))
                    plt.ylabel('True label')
                    plt.xlabel('Predicted label')
                    plt.show()


# this is the base file used for checking 
