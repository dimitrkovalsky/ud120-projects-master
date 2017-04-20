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
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 0.98976109215
# 0.984072810011


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

clf = svm.SVC(kernel="rbf", C=10000.0)
t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

t1 = time()
predicted = clf.predict(features_test)
print("prediction time:", round(time()-t1, 3), "s")

print(accuracy_score(predicted, labels_test))

print("10 element: ", predicted[10])
print("26 element: ", predicted[26])
print("50 element: ", predicted[50])

print(len(np.nonzero(predicted)))

# C-10 -> 0.616040955631
# C-100 -> 0.616040955631
# C-1000 -> 0.821387940842
# C-10000 -> 0.892491467577

# Full data set accuracy -> 0.990898748578
