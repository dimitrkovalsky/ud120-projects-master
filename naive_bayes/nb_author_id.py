#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

model = GaussianNB()
model.fit(features_train, labels_train)

errors = 0
for i in range(len(features_test)):
    predicted = model.predict(features_test[i].reshape(1, -1))
    if predicted != labels_test[i]:
        errors += 1
    print("Available : {}, expected : {}".format(predicted, labels_test[i]))

print("Examples : {} Errors : {}, percents {}".format(len(labels_test), errors, 1 - (errors / len(labels_test))))
#########################################################
### your code goes here ###


#########################################################
