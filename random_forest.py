import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
import pandas as pd
import ggplot
import os, sys
import subprocess

random_seed = 123


f = open('/Users/vincent/Google_drive/public/cathay2017/feature_rf.txt','r')
X = []
y = []
for line in f.readlines():
    li = line.split()
    X.append([float(li[0]),float(li[1]),float(li[2]),float(li[3]),float(li[4]),float(li[5]),float(li[6]),float(li[7]),float(li[8]),float(li[9]),float(li[10])])
    y.append(float(li[11]))



# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for parallelized ensembles but is necessary for tracking the OOB
# error trajectory during training.
ensemble_clfs = [
                 ("RandomForestClassifier, max_features='sqrt'",
                  RandomForestClassifier(warm_start=True, oob_score=True,
                                         max_features="sqrt",
                                         random_state=random_seed)),
                 ("RandomForestClassifier, max_features='log2'",
                  RandomForestClassifier(warm_start=True, max_features='log2',
                                         oob_score=True,
                                         random_state=random_seed))
                 ]

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` i.e., the numbers of trees to be explored.
min_estimators = 1
max_estimators = 50


for label, clf in ensemble_clfs:
    n_estimators_array = []
    error_array = []
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X, y)
        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))
        # print "done1 %d with error %.5f" %(i,oob_error)
        n_estimators_array.append(i)
        error_array.append(oob_error)
        plt.figure()
        plt.xlim([1,50])
        plt.ylim([0.10,0.40])
        plt.plot(n_estimators_array,error_array)
        plt.title('Random Forest')
        plt.xlabel("number of trees")
        plt.ylabel("Out of Bag error rate")
        plt.savefig('OOB-n_%02d.pdf' %i)
        plt.figure()


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


args = (['convert', '-delay', '20' , 'OOB-n_*.png' ,'OOB-n.gif'])
subprocess.check_call(args)


g = open("results_%d-%d.txt" %(min_estimators, max_estimators),'w')



for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    g.write("%s \n" %label)
    #plt.plot(xs, ys, label=label)
    for i in range(len(clf_err)):
        g.write('%.3f %.6f  ' %(clf_err[i][0],clf_err[i][1]))
        g.write('\n')

g.close()







