"""
Extract a set of Topics from Arxiv Data Repository
Programmed by Jonathan Quijas

"""
import numpy as np
import pickle
import re
import time
from nltk.corpus import stopwords
import nltk

from TopicExtraction.EmbeddingModule import EmbeddingsReader
from TopicExtraction.ArxivDataModule import ArxivReader
from TopicExtraction.ArxivDataModule import ArxivManager
from TopicExtraction.KeywordExtractionModule import ChunkTreeCSM
from TopicExtraction.EmbeddingModule import EmbeddingsReader
from Chunking.ChunkingModule import ConsecutiveNPChunker
from Chunking import ChunkingModule

import pprint
import platform

from collections import Counter

#Clock in time
start = time.time()
if(platform.system() == 'Windows'):
    common_words_path = 'C:\\Users\\Jona Q\\Documents\\Embeddings\\20k.txt'
else:
    common_words_path = '/home/jonathan/Documents/WordEmbedding/20Newsgroups/Data/20k.txt'


#Get embeddings
embObj = EmbeddingsReader()
normalize = True
embeddings = embObj.readEmbeddings(normalize)
#Store arrays with all embedding values
values_array = embObj.valuesArray()
keys_array = embObj.keysArray()


#Create stopwords set
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation
common_words_file = open(common_words_path, 'r')
common_words_list = [word for line in common_words_file for word in line.split()]
common_words_file.close()
#Select the top num_common_words words to remove from vocabulary
num_common_words = 600
#Update stopwrd set with common words
stop_words.update(common_words_list[0:num_common_words])
#Convert to unicode
if(platform.system() == 'Windows'):
    stop_words = [word for word in stop_words]
    tagger_path = tagger_path = 'C:\\Users\\Jona Q\\Documents\\GitHub\\Arxiv\\chunker.pickle'
else:
    stop_words = [unicode(word) for word in stop_words]
    tagger_path = "/home/jonathan/Documents/WordEmbedding/Arxiv/chunker2.pickle"

#
# Set up noun phrase (NP) grammar
#
tags = ['NN','NNS', 'JJ', 'RB', 'VBP', 'VB']
tag_filter = ['NN', 'NNS', 'JJ']

arxivManager = ArxivManager()
for c in arxivManager.categories():
    raw_data = arxivManager.read(c)
    for i, abstract in enumerate(raw_data):
        #Preprocess our raw text data
        abstract = [re.sub(r'[^\x00-\x7F]+','', sentence) for sentence in abstract]
        abstract = [re.sub("[^a-zA-Z0-9\-,.]", " ", sentence).lower() for sentence in abstract]
        print(abstract)
        abstract = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in abstract]
        with open(tagger_path, 'rb') as handle:
            chunker = pickle.load(handle)
        chunks = [chunker.parse(sentence) for sentence in abstract]

        tree_csm = ChunkTreeCSM(chunks, embeddings, tags)
        maxCsmOutput = tree_csm.selectKeywords(tags, 'min')
        print("maxCsmOutput")
        pprint.pprint(maxCsmOutput)
        if platform.system() == 'Windows':
            wait = input("PRESS ENTER TO CONTINUE.")
        else:
            wait = raw_input("PRESS ENTER TO CONTINUE.")

        print('\n')
