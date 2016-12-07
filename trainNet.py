import numpy as np
import time
from collections import Counter

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

start = time.time()
#data_path = 'Data/Training/training_trigram_data.txt'
#labels_path = 'Data/Training/training_trigram_labels.txt'
data_path = '../Data/Training/training_trigram_data.txt'
labels_path = '../Data/Training/training_trigram_labels.txt'

"""X = np.loadtxt(data_path, delimiter=',')
with open(labels_path) as f:
    y = f.read().splitlines()"""

X = np.loadtxt(data_path, delimiter=',')
y = np.genfromtxt(labels_path, delimiter=',', dtype='str')
(r,c) = X.shape
end = time.time()-start
print 'Size of X: ', X.shape
print 'Size of y: ', len(y)

print 'Finished loading data after ', end, ' seconds'
print 'Loading data and making dictionaries'
start = time.time()

dictX = {}
dictY = {}
for i, x in enumerate(X):
    key = x[c-1]
    if key in dictX:
        dictX[key] += [x]
        dictY[key] += [y[i]]
    else:
        dictX[key] = [x]
        dictY[key] = [y[i]]

print 'Finished loading data and making dictionaries after ', time.time() - start, ' seconds'

indices = np.random.permutation(len(dictY.items()))
test_split = .33
train_indices = indices[0:int(len(indices)*(1-test_split))]
test_indices = indices[int(len(indices)*(1-test_split)):]
X = dictX.items()
y = dictY.items()
X_train = []
y_train = []
for i in train_indices:
    data_batch = []
    labels_batch = []
    for j, x in enumerate(X[i][1]):
         data_batch += [x[0:100]]
         labels_batch += [y[i][1][j].split()[0]]
    #data_batch = preprocessing.scale(data_batch, axis=1).tolist()

    X_train += data_batch
    y_train += labels_batch


start = time.time()
#Train MLP
clf = MLPClassifier(hidden_layer_sizes=(300,200,100,50), verbose=True, activation="relu", tol=.001)
#clf = neighbors.KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='kd_tree', n_jobs=-1)
#clf = RandomForestClassifier()
#clf = SVC()
#clf = linear_model.LogisticRegression(penalty='l2', verbose=1)
#clf = LinearSVC()
#clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
#5437+4972+4611
#clf = QuadraticDiscriminantAnalysis(reg_param=0.001,priors=[float(5437)/float(15020),float(4972)/float(15020),float(4611)/float(15020)])
#clf=  GaussianProcessClassifier()

clf.fit(X_train, y_train)
print 'Finished training after ', time.time()-start, ' seconds'

#Test
#print 'Score = ', clf.score(X_test, y_test)

print 'Testing...'
start = time.time()
accuracy = 0.0
y_true_list = []
y_pred = []
for ii, i in enumerate(test_indices):
    if ii % 200 == 0:
        print ii,'/',len(test_indices)
    cnt = Counter()
    y_true = y[i][1][0].split()[0]
    y_true_list += [y_true]
    #print "y_true = ", y_true

    #NEW CODE
    test_batch = []
    for j, x in enumerate(X[i][1]):
        test_batch += [x[0:100]]
    #test_batch = preprocessing.scale(test_batch,axis=1).tolist()

    for x in test_batch:
        label = clf.predict(np.array(x).reshape(1,-1))[0]
        #print "label = ", label
        cnt[label] += 1

    """for j, x in enumerate(X[i][1]):
        label = clf.predict((x[0:100]).reshape(1,-1))[0]
        #print "label = ", label
        cnt[label] += 1"""
    #print cnt.most_common()
    y_hat = max(cnt, key=cnt.get)
    y_pred += [y_hat]
    #print "y_hat = ", y_hat
    if y_true == y_hat:
        accuracy += 1.0
accuracy /= float(len(test_indices))
print 'Testing accuracy = ', accuracy
print 'Finished after ', time.time()-start, ' seconds'

from sklearn.metrics import f1_score
f1 = f1_score(y_true_list, y_pred, average='macro')
print 'fmeasure = ', f1
