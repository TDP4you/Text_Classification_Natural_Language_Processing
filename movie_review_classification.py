# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 16:48:41 2018

@author: tdpco
"""

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics

# Importing the dataset
dataset = pd.read_csv('moviereviews.tsv', sep='\t')


# Cleaning the dataset
print("Checking for null Values:")
print(dataset.isnull().sum())
print("\nDeleting Null Values....\n")
dataset.dropna(inplace=True)
print("Removing Spaces....\n")
blanks = []
for i, lb, rv in dataset.itertuples():
    if rv.isspace():
        blanks.append(i)
dataset.drop(blanks, inplace=True)
print(dataset.head())
print(f"Now, The Dataset has {len(dataset)} reviews")

# Dividing the Dataset in dependent and independent variable
X = dataset['review']
y = dataset['label']

# Divinding Dataset into test and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating a pipeline
pipeline_object = Pipeline([('vectorizer',TfidfVectorizer()),('model',LinearSVC())])
pipeline_object.fit(X_train, y_train)
predictions = pipeline_object.predict(X_test)

# Getting the result
print("\nThe Confusion Matrix is:")
print(metrics.confusion_matrix(y_test, predictions))
print(f"\nAccuracy of model is {round(metrics.accuracy_score(y_test,predictions)*100)} %")
