# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 13:04:59 2018

@author: tdpco
"""

# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics

# Importing Dataset
dataset = pd.read_csv('smsspamcollection.tsv', sep='\t')
print(dataset.head())

# Declaring Dependent and Independent variable
X = dataset['message']
y = dataset['label']

# Dividing into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorizing the test set
tfidf_vector = TfidfVectorizer()
tfidf_vector.fit_transform(X_train)

# Using Linear SVC Model
model = LinearSVC()

# Instead we can use pipeline
text_model = Pipeline([('tfidf', TfidfVectorizer()), ('model', LinearSVC())])
text_model.fit(X_train,y_train)

# Making predictions
predictions = text_model.predict(X_test)

# Confusion Matrix
cm = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['ham', 'spam'], columns=['ham', 'spam'])
print("\n")
print(cm)
print("\n")
accuracy = round(metrics.accuracy_score(y_test, predictions)*100, 2)
print(f"Accuracy of LinearSVC model is {accuracy}%")

# Asking user for checking messages
exit_flag = True
while exit_flag:
    user_input = input("\nInsert your message to check (Type exit to close): ")
    if user_input == "exit":
        exit_flag = False
        break
    else:
        answer = text_model.predict([user_input])
        print(f"The message is {answer}")
