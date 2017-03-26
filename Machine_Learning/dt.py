import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.externals import joblib
import pandas as pd
import ggplot
import os, sys
import subprocess
import time
import numpy as np

random_seed = 123

## readdata
f = open('/Users/vincent/Google_drive/public/cathay2017/feature_rf.txt','r')
X = []
y = []
for line in f.readlines():
    li = line.split()
    X.append([float(li[0]),float(li[1]),float(li[2]),float(li[3]),float(li[4]),float(li[5]),float(li[6]),float(li[7]),float(li[8]),float(li[9]),float(li[10])])
    y.append(float(li[11]))

# define training set, cross-validation set, and testing set.

Ntot = len(y)

Ncv = Ntot / 5
Ntest = Ntot / 5
Ntrain = Ntot - Ncv - Ntest

X_train = X[:Ntrain]
y_train = y[:Ntrain]

X_cv = X[Ntrain:Ntrain + Ncv]
y_cv = y[Ntrain:Ntrain + Ncv]

X_test = X[Ntrain + Ncv:]
y_test = y[Ntrain + Ncv:]

# define error




max_depth_array = []
train_error = []
cv_error = []
elapsetimearray=[]


# cross-validation
max_depth_of_level = 60
for depth in range(1, max_depth_of_level):
    start = time.time()
    dt = DecisionTreeClassifier(max_depth = depth)
    dt.fit(X_train, y_train)
#    y_cv_pred = clf.predict(X_cv)
#   y_train_pred = clf.predict(X_train)
    train_error.append(1 - dt.score(X_train,y_train))
    cv_error.append(1 - dt.score(X_cv,y_cv))
    max_depth_array.append(depth)
    #plot
    plt.figure()
    plt.xlim([1,61])
    plt.ylim([0,1])
    plt.plot(max_depth_array, cv_error, label='cross-val error')
    plt.plot(max_depth_array, train_error, label='training error')
    plt.legend()
    plt.xlabel('max depth of the decision tree')
    plt.ylabel('error rate')
    plt.savefig("error_depth=%02d_10.png"%depth)






