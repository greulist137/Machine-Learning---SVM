#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#Create initial SVM with a Linear kernal and fit data to it
clf = svm.SVC(kernel = 'linear')
clf.fit(features_train, labels_train)

#use the features_test testing data now for predictions and assign it to a prediction variable
pred = clf.predict(features_test)
print('Prediction: ')
print(pred)

#Calculate the accuracy and present the results
acc_score = accuracy_score(pred, labels_test)

print('Accuracy: ')
print(acc_score) 
