import numpy as np
import pickle
import time
from collections import Counter
import platform
from sklearn.neural_network import MLPClassifier


from sklearn.model_selection import train_test_split


start = time.time()
#data_path = 'Data/Training/training_trigram_data.txt'
#labels_path = 'Data/Training/training_trigram_labels.txt'
if platform.system() == 'Windows':
    data_path = '..\\Data\\Training\\training_trigram_50_data.txt'
    labels_path = '..\\Data\\Training\\training_trigram_50_labels.txt'
else:
    data_path = '../Data/Training/training_trigram_data.txt'
    labels_path = '../Data/Training/training_trigram_labels.txt'

net_results_path = 'net_results.txt'

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
test_split = .10
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
         data_batch += [x[0:len(x)-1]]
         labels_batch += [y[i][1][j].split()[0]]
    #data_batch = preprocessing.scale(data_batch, axis=1).tolist()

    X_train += data_batch
    y_train += labels_batch


start = time.time()
#Train MLP
#clf = MLPClassifier(hidden_layer_sizes=(1500,700,500,300,200,100,50), verbose=True, activation="relu", tol=.0001)
clf = MLPClassifier(hidden_layer_sizes=(400,300,200,100,50,10), verbose=True, activation="relu", tol=.0001)
#clf = MLPClassifier(hidden_layer_sizes=(350,350,350,350,350), verbose=True, activation="relu", tol=.0001)

clf.fit(X_train, y_train)
print 'Finished training after ', time.time()-start, ' seconds'

with open('net.pickle', 'wb') as handle:
  pickle.dump(clf, handle)
handle.close()

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
        test_batch += [x[0:len(x)-1]]
    #test_batch = preprocessing.scale(test_batch,axis=1).tolist()

    avg_prob = np.zeros((1,3))
    print "True class = ", y_true
    for x in test_batch:
        label = clf.predict(np.array(x).reshape(1,-1))[0]
        prob = clf.predict_proba(np.array(x).reshape(1,-1))
        avg_prob += prob
        print "prob = ", prob
        cnt[label] += 1
    avg_prob /= 3

    """for j, x in enumerate(X[i][1]):
        label = clf.predict((x[0:100]).reshape(1,-1))[0]
        #print "label = ", label
        cnt[label] += 1"""
    #print cnt.most_common()
    #y_hat = max(cnt, key=cnt.get)
    y_hat = clf.classes_[np.argmax(avg_prob)]
    print "predicted class = ", y_hat
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

results_file = open(net_results_path,'w')
results_file.write('Testing accuracy = ' + str(accuracy) + '\n' + 'fmeasure = ' + str(f1))
results_file.close()
