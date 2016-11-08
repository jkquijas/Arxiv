import numpy as np
import time
from nltk.corpus import stopwords
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
from phpserialize import unserialize
from Chunking.ChunkingModule import ConsecutiveNPChunker
from random import shuffle
import platform
from nltk.corpus import conll2000
import pickle

start = time.time()

#For each paper
if(platform.system() == 'Windows'):
    common_words_path = 'C:\\Users\\Jona Q\\Documents\GitHub\\Arxiv\\Data\\20k.txt'
    rake_common_path = "C:\\Users\\Jona Q\\Documents\GitHub\\Arxiv\\Data\\500common.txt"
    papers_path = "C:\\Users\\Jona Q\\Documents\GitHub\\Arxiv\\Data\\ACMSerializedCoPPubs\\serializedPubsCS.txt"
else:
    common_words_path = '/home/jonathan/Documents/WordEmbedding/Arxiv/Data/20k.txt'
    papers_path = "/home/jonathan/Documents/WordEmbedding/Arxiv/Data/ACMSerializedCoPPubs/serializedPubsCS.txt"
    rake_common_path = "/home/jonathan/Documents/WordEmbedding/Arxiv/Data/500common.txt"
if(platform.system() == 'Windows'):
    data = raw_data = open(papers_path, 'rb').read()
    data = unserialize(data)
    key_title = b'pubTitle'
    key_concept = b'pubConcepts'
    key_abstract = b'pubAbstract'
else:
    data = raw_data = open(papers_path, 'r').read()
    data = unserialize(data)
    key_title = 'pubTitle'
    key_concept = 'pubConcepts'
    key_abstract = 'pubAbstract'

numPapers = len(data)


test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
chunker = ConsecutiveNPChunker(train_sents)
print(chunker.evaluate(test_sents))

with open('chunker.pickle', 'wb') as handle:
  pickle.dump(chunker, handle)
handle.close()

end = time.time()
print("Finished After ", end-start)
