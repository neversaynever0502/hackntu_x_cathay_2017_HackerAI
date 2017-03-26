import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
import pandas as pd
import ggplot
import os, sys
import subprocess
from sklearn.svm import SVC

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



# Range of `n_estimators` i.e., the numbers of trees to be explored.
min_estimators = 1
max_estimators = 51


# cross-validation

#max_depth_of_level = 60
#for tree_number in range(1, max_depth_of_level):
#        dt = DecisionTreeClassifier(max_depth = depth)
#    dt.fit(X_train, y_train)
#    #    y_cv_pred = clf.predict(X_cv)
#    #   y_train_pred = clf.predict(X_train)
#    train_error.append(1 - dt.score(X_train,y_train))
#    cv_error.append(1 - dt.score(X_cv,y_cv))
#    max_depth_array.append(depth)
#    #plot
#    plt.figure()
#    plt.xlim([1,61])
#    plt.ylim([0,1])
#    plt.plot(max_depth_array, cv_error, label='cross-val error')
#    plt.plot(max_depth_array, train_error, label='training error')
#    plt.legend()
#    plt.xlabel("number of trees")
#    plt.ylabel('error rate')
#    plt.savefig('OOB-n_%02d.png' %tree_number)
#
#




n_estimators_array = []
train_error = []
cv_error = []
sub_c = []

C_list = [1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1e+00, 1e+01, 1e+02]
for c in C_list:
    clf = SVC(C=c, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
    clf.fit(X_train, y_train)
    print "done fit !"
    train_error.append(1 - clf.score(X_train,y_train))
    cv_error.append(1 - clf.score(X_cv,y_cv))
    sub_c.append(c)
    plt.figure()
    plt.xscale('log')
    plt.xlim([1e-05,1e+02])
    plt.ylim([0,1])
    plt.plot(sub_c, cv_error, label = 'cross-val error')
    plt.plot(sub_c, train_error, label = 'training error')
    plt.xlabel("SVM C")
    plt.ylabel("Error Rate")
    plt.savefig('svm_cv%07f.png' %c)
    plt.figure()

#
#for i in range(min_estimators, max_estimators+1 , 10):
#        rfc = RandomForestClassifier(warm_start=True, oob_score=True,
#                       max_features="sqrt",
#                       random_state=random_seed)
#        rfc.set_params(n_estimators=i)
#        rfc.fit(X_train, y_train)
#        train_error.append(1 - rfc.score(X_train,y_train))
#        cv_error.append(1 - rfc.score(X_cv,y_cv))
#        #oob_error = 1 - clf.oob_score_
#        #error_rate[label].append((i, oob_error))
#        n_estimators_array.append(i)
#        plt.figure()
#        plt.xlim([1,50])
#        plt.ylim([0,1])
#        plt.plot(n_estimators_array, cv_error, label = 'cross-val error')
#        plt.plot(n_estimators_array, train_error, label = 'training error')
#        plt.xlabel("Number of Trees")
#        plt.ylabel("Error Rate")
#        plt.savefig('rf_cv%02d.png' %i)
#        plt.figure()
#

# Generate the "OOB error rate" vs. "n_estimators" plot.



#for label, clf in ensemble_clfs:
#    aaa = pd.DataFrame(error_rate["%s" %label])
#    n_min = aaa.ix[aaa[aaa[1]==aaa[1].min()].index[0],0]
#    print n_min
#    clf.set_params(n_estimators = n_min)
#    clf.fit(X,y)
#    oob_error = 1 - clf.oob_score_
#    clf = joblib.dump(clf,"%s_rf_%d.pks" %(label,n_min))
#    print "the error for %s is %.4f" %(label,oob_error)


args = (['convert', '-delay', '20' , 'rf_cv*.png' ,'rf_cv.gif'])
subprocess.check_call(args)

#
#g = open("results_%d-%d.txt" %(min_estimators, max_estimators),'w')
#
#
#
#for label, clf_err in error_rate.items():
#    xs, ys = zip(*clf_err)
#    g.write("%s \n" %label)
#    #plt.plot(xs, ys, label=label)
#    for i in range(len(clf_err)):
#        g.write('%.3f %.6f  ' %(clf_err[i][0],clf_err[i][1]))
#        g.write('\n')
#
#g.close()
#






