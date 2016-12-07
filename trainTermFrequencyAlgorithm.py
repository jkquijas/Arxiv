from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import os
import platform
import pickle
from collections import Counter
from nltk.corpus import stopwords
import re
import nltk
import numpy as np
import time



#
# Create stopwords set
#
stop_words = set(stopwords.words('english'))
stop_words.update(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                   'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'cid', 'et', 'al', '.al', 'al.', 'circa', 'figure'])
stop_words = [word for word in stop_words]
#
# Set up noun phrase (NP) grammar
#
tags = ['NN','NNS', 'JJ', 'RB', 'VBP', 'VB']
tag_filter = ['NN', 'NNS', 'JJ']


if(platform.system() == 'Windows'):
    common_words_path = 'C:\\Users\\Jona Q\\Documents\GitHub\\Arxiv\\Data\\20k.txt'
    rake_common_path = "C:\\Users\\Jona Q\\Documents\GitHub\\Arxiv\\Data\\500common.txt"
    papers_path = "C:\\Users\\Jona Q\\Documents\GitHub\\Arxiv\\Data\\ACMSerializedCoPPubs\\serializedPubsCS.txt"
    rootDir = 'Data\\Abstracts\\'
else:
    common_words_path = '/home/jonathan/Documents/WordEmbedding/Arxiv/Data/20k.txt'
    papers_path = "/home/jonathan/Documents/WordEmbedding/Arxiv/Data/ACMSerializedCoPPubs/serializedPubsCS.txt"
    rake_common_path = "/home/jonathan/Documents/WordEmbedding/Arxiv/Data/500common.txt"
    rootDir = 'Data/Abstracts/'



titleKey = 'Title : '
typeKey = 'Type : '
classKey = 'NSF Org : '
abstractKey = 'Abstract : '
latestAmendmentKey = 'Latest Amendment Date'

#'DMS ': 9910, 'DUE ': 8808, 'EAR ': 7255, 'DMI ': 6525, 'OCE ': 5998, 'INT ': 5946, 'SES ': 5943, 'CHE ': 5702, 'IBN ': 5437, 'DMR ': 5379, 'DEB ': 4972, 'BCS ': 4702, 'MCB ': 4611, 'ESI ': 4434, 'ATM ': 4237, 'CCR ': 4163, 'CMS ': 4038, 'DBI ': 3521, 'CTS ': 3137, 'PHY ': 3120, 'ECS ': 2998, 'OPP ': 2862, 'IIS ': 2329, 'BES ': 1973, 'DGE ': 1917, 'AST ': 1869, 'EIA ': 1752, 'ANI ': 1619, 'HRD ': 1494, 'EEC ': 1446, 'REC ': 905, 'ACI ': 717, 'OIA ': 530, 'OIG ': 428, 'ESR ': 343, 'DIS ': 252, 'SRS ': 231, 'EPS ': 162, 'EHR ': 102, 'GEO ': 101, 'BIO ': 62, 'ENG ': 43, 'EID ': 36, 'RED ': 30, 'HRM ': 29, 'MIP ': 28, 'SBE ': 27, 'LPA ': 22, 'DOB ': 21, 'EF ': 17, 'DAS ': 17, 'MPS ': 16, 'SBR ': 9, 'DFM ': 8, 'NCO ': 8, 'MDR ': 8, 'OEO ': 7, 'CPO ': 7, 'III ': 6, 'O/D ': 5, 'CSE ': 4, 'NONE ': 4, 'IRM ': 4, '': 3, 'NCR ': 3, 'STI ': 3, 'DCB ': 3, 'NSB ': 2, 'PRA ': 2, 'null ': 1, 'BFA ': 1
#labelsList = ['DMS ','DUE ','EAR ']
labelsList = ['IBN ','DEB ','MCB ']
fileSet = set()

for dir_, _, files in os.walk(rootDir):
    for fileName in files:
        relDir = os.path.relpath(dir_, rootDir)
        relFile = os.path.join(relDir, fileName)
        fileSet.add(relFile)

fileList = [rootDir + filename for filename in fileSet]


print 'Building abstract list'
start = time.time()
X = []
y = []
n_features = 1000
for i, f in enumerate(fileList):
    if i % 5000 == 0:
        print i
    if f[len(f)-3:] != 'txt':
        continue
    file = open(f, 'r')
    text = file.read()
    text = ' '.join(text.split())
    try:
        title = text[text.index(titleKey)+len(titleKey):text.index(typeKey)]
        label = text[text.index(classKey) + len(classKey):text.index(latestAmendmentKey)]
        if label not in labelsList:
            continue
        abstract = text[text.index(abstractKey)+len(abstractKey):]
        abstract = re.sub(r'[^\x00-\x7F]+','', abstract)
        X = X + [abstract]
        y = y + [label]
    except:
        continue
    file.close()
print 'Finished building abstract list after', time.time() - start, ' seconds'

print 'Building dictionary '
start = time.time()
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
indices = np.random.permutation(len(y))
test_split = .33
train_indices = indices[0:int(len(indices)*(1-test_split))]
test_indices = indices[int(len(indices)*(1-test_split)):]
X_train = [X[i] for i in train_indices]
y_train = [y[i] for i in train_indices]
X_test = [X[i] for i in test_indices]
y_test = [y[i] for i in test_indices]

X_train = tfidf_vectorizer.fit_transform(X_train)
print 'Finished building dictionary after', time.time() - start, ' seconds'
print 'Training LinearSVC'
start = time.time()
clf = LinearSVC()
clf.fit(X_train,y_train)
print 'Finished training after ', time.time() - start, ' seconds'

X_test = tfidf_vectorizer.transform(X_test)
#print 'Score = ', clf.score(X_test, y_test)
accuracy = 0.0
y_pred = []
for i, x in enumerate(X_test):
    y_hat = clf.predict(x)[0]
    if y_hat == y_test[i]:
        accuracy+=1
    y_pred+=[y_hat]
accuracy/=i
print 'Test accuracy = ', accuracy
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, average='macro')
print 'fmeasure = ', f1
