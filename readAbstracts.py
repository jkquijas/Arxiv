import os
import platform
import pickle
from collections import Counter
from TopicExtraction.KeywordExtractionModule import ChunkTreeCSM
from TopicExtraction.EmbeddingModule import EmbeddingsReader
from Chunking.ChunkingModule import ConsecutiveNPChunker
from Chunking import ChunkingModule
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import nltk
import numpy as np



#
# Get embeddings
#
embedding_size = 50
embObj = EmbeddingsReader(embedding_size = embedding_size)
normalize = True
embeddings = embObj.readEmbeddings(normalize)
values_array = embObj.valuesArray()
keys_array = embObj.keysArray()

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
    tagger_path = 'C:\\Users\\Jona Q\\Documents\\GitHub\\Arxiv\\chunker.pickle'
    rootDir = 'Data\\Abstracts\\'
else:
    common_words_path = '/home/jonathan/Documents/WordEmbedding/Arxiv/Data/20k.txt'
    papers_path = "/home/jonathan/Documents/WordEmbedding/Arxiv/Data/ACMSerializedCoPPubs/serializedPubsCS.txt"
    rake_common_path = "/home/jonathan/Documents/WordEmbedding/Arxiv/Data/500common.txt"
    tagger_path = "/home/jonathan/Documents/WordEmbedding/Arxiv/chunker2.pickle"
    rootDir = 'Data/Abstracts/'
    tagger_path = "/home/jonathan/Documents/WordEmbedding/Arxiv/chunker2.pickle"
    training_data_path = "Data/Training/training_trigram_data.txt"
    training_labels_path = "Data/Training/training_trigram_labels.txt"

#   Load trained chunker
with open(tagger_path, 'rb') as handle:
    chunker = pickle.load(handle)


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



cnt = Counter()
labels_file = open(training_labels_path,'w')

training_data_file = open(training_data_path,'w')
training_data_file.write("")
training_data_file.close()
training_data_file = open(training_data_path,'w+')

#Create chunks using our grammar
with open(tagger_path, 'rb') as handle:
    chunker = pickle.load(handle)

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
        abstract = nltk.sent_tokenize(abstract)


        abstract = [re.sub(r'[^\x00-\x7F]+','', sentence) for sentence in abstract]
        abstract = [re.sub("[^a-zA-Z0-9\-,]", " ", sentence).lower() for sentence in abstract]
        abstract = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in abstract]



        chunks = [chunker.parse(sentence) for sentence in abstract]

        tree_csm = ChunkTreeCSM(chunks, embeddings, tags)

        minCsmOutput = tree_csm.selectKeywords(tags, 'min', True)
        for r in minCsmOutput:
            if len(r) == 1:
                np.savetxt(training_data_file, np.hstack((r[0],r[0],r[0],i))[np.newaxis], delimiter=',')
                labels_file.write(label+str(i)+'\n')
            elif len(r) == 2:
                np.savetxt(training_data_file, np.hstack((r[0],r[0],r[1],i))[np.newaxis], delimiter=',')
                np.savetxt(training_data_file, np.hstack((r[0],r[1],r[1],i))[np.newaxis], delimiter=',')
                labels_file.write(label+str(i)+'\n')
                labels_file.write(label+str(i)+'\n')
            else:
                np.savetxt(training_data_file, np.hstack((r[0],r[0],r[1],i))[np.newaxis], delimiter=',')
                labels_file.write(label+str(i)+'\n')
                for k in range(len(r)-2):
                    np.savetxt(training_data_file, np.hstack((r[k],r[k+1],r[k+2],i))[np.newaxis], delimiter=',')
                    labels_file.write(label+str(i)+'\n')
                np.savetxt(training_data_file, np.hstack((r[len(r)-2],r[len(r)-1],r[len(r)-1],i))[np.newaxis], delimiter=',')
                labels_file.write(label+str(i)+'\n')
        #print "saved abstract ", i
    except:
        print 'Error in ', i
        continue

    cnt[label] += 1
    file.close()

labels_file.close()
training_data_file.close()

print len(cnt)
print cnt
