import numpy as np
from nltk.corpus import stopwords
import time
from collections import Counter
from TopicExtraction.KeywordExtractionModule import ChunkTreeCSM
from TopicExtraction.EmbeddingModule import EmbeddingsReader
from Chunking.ChunkingModule import ConsecutiveNPChunker
from Chunking import ChunkingModule

import pickle
import numpy as np
import time
from nltk.corpus import stopwords
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
from phpserialize import unserialize


import pprint
import platform

#
#Compute tuple [avgPrecision, avgRecall, avgFmeasure]
#output is a list of lists. Each sublist contains tokens i.e. words, candidate keywords
#
#true_keywords_map maps a word to a keyword index number e.g. true_keywords_map['comput'] -> [2, 3] means the token 'comput'
#is found in the 2nd and 3rd keywords
#
#true_keywords_lengths[i] gives the number of tokens in the ith keyword
#
#Assumes the candidate keywords have been Stemmed e.g. str(SnowballStemmer("english").stem(candidate_keyword)
def computePerformance(output, true_keywords_map, true_keywords_lengths):
    keyword_counter = {}
    keyword_index_mapping = {}

    for i, keywords in enumerate(output):

        #For the ith word, create a list of lists, each sublist corresponds to
        #that token's indexes of keywords in which the token can be found
        hits = [true_keywords_map[k] for k in keywords if k in true_keywords_map]
        #Flatten list
        hits = [item for sublist in hits for item in sublist]
        #Count all indexes
        C = Counter(hits)
        #Get the most common index i.e. most token "hits"
        keyword_hit = C.most_common(1)
        #If we actually made a hit
        if keyword_hit:
            hit = keyword_hit[0]
            #If we already had previous results for this keyword
            if(hit[0] in keyword_counter):
                #If we have more hits, update
                if hit[1] > keyword_counter[hit[0]]:
                    keyword_counter[hit[0]] = hit[1]
                    keyword_index_mapping[hit[0]] = i
            else:
                keyword_counter[hit[0]] = hit[1]
                keyword_index_mapping[hit[0]] = i

    #Compute precision, recall and fmeasure for each keyword
    #keyword_counter[k] -> greatest number of token hits from a single candidate keyword
    avgPrecision = 0.0
    avgRecall = 0.0
    avgFmeasure = 0.0


    for k in keyword_counter.keys():
        precision = keyword_counter[k]/len(output[keyword_index_mapping[k]])
        recall = keyword_counter[k]/true_keywords_lengths[k]
        fmeasure = 2*((precision*recall)/(precision+recall))

        avgPrecision += precision
        avgRecall += recall
        avgFmeasure += fmeasure

    n_keywords = len(true_keywords_lengths)
    avgPrecision = avgPrecision/n_keywords
    avgRecall = avgRecall/n_keywords
    avgFmeasure = avgFmeasure/n_keywords

    return [avgPrecision, avgRecall, avgFmeasure]

#
# Perform DFS on grammar-generated parse tree
# to find the noun phrase NP, if any, that contains the
# specified input word
#
def traverseTree(tree, word):
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            if subtree._label == 'NP':
                for leave in subtree.leaves():
                    if word == leave[0]:
                        return [l for l in subtree.leaves()]
                    else:
                        traverseTree(subtree, word)
    return []

#Filter a noun phrase by tags
#Returns a list of tokens in noun_phrase whose part of speech is found in tag list 'tags'
def filterByTags(noun_phrase, tags):
    return [token[0] for token in noun_phrase if token[1] in tags]

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

def get_topics(model, feature_names, n_top_words):
    topics_list = []
    for topic_idx, topic in enumerate(model.components_):
        topics_list = topics_list + [" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])]
    return topics_list


def empty(seq):
    try:
        return all(map(empty, seq))
    except TypeError:
        return False

#
# Clock in time
#
start = time.time()

if(platform.system() == 'Windows'):
    common_words_path = 'C:\\Users\\Jona Q\\Documents\GitHub\\Arxiv\\Data\\20k.txt'
    rake_common_path = "C:\\Users\\Jona Q\\Documents\GitHub\\Arxiv\\Data\\500common.txt"
    papers_path = "C:\\Users\\Jona Q\\Documents\GitHub\\Arxiv\\Data\\ACMSerializedCoPPubs\\serializedPubsCS.txt"
else:
    common_words_path = '/home/jonathan/Documents/WordEmbedding/Arxiv/Data/20k.txt'
    papers_path = "/home/jonathan/Documents/WordEmbedding/Arxiv/Data/ACMSerializedCoPPubs/serializedPubsCS.txt"
    rake_common_path = "/home/jonathan/Documents/WordEmbedding/Arxiv/Data/500common.txt"

#
# Get embeddings
#
embObj = EmbeddingsReader()
normalize = True
embeddings = embObj.readEmbeddings(normalize)
values_array = embObj.valuesArray()
keys_array = embObj.keysArray()


#
# Create stopwords set
#
stop_words = set(stopwords.words('english'))
if(platform.system() == 'Windows'):
    stop_words = [word for word in stop_words]
else:
    stop_words = [unicode(word) for word in stop_words]



#
# Set up noun phrase (NP) grammar
#
tags = ['NN','NNS', 'JJ', 'RB', 'VBP', 'VB']
tag_filter = ['NN', 'NNS', 'JJ']

#For each paper
if(platform.system() == 'Windows'):
    data = raw_data = open(papers_path, 'rb').read()
    data = unserialize(data, charset='gb18030')
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


for i in range(numPapers):
    if len(data[i][key_concept]) == 0:
        continue

    file_name = data[i][key_title]
    #Read text and keywords
    abstract = data[i][key_abstract]
    if(platform.system()=='Windows'):
        abstract = abstract.decode()
    if str(abstract) == '':
        continue


    if platform.system() == 'Windows':
        keywords = [(data[i][key_concept][j][0]).decode() for j in range(len(data[i][key_concept]))]
    else:
        keywords = [(data[i][key_concept][j][0]) for j in range(len(data[i][key_concept]))]
    keywords = [re.sub(r'[^\x00-\x7F]+','', k) for k in keywords]

    print(abstract)
    pprint.pprint(keywords)

    keywords = [re.sub("[^a-zA-Z0-9\-,]", " ", k).lower() for k in keywords]
    keywords = [k.split() for k in keywords]
    keywords = [[str(SnowballStemmer("english").stem(item)) for item in sublist] for sublist in keywords]

    true_keywords_lengths = [len(l) for l in keywords]

    #Create true kwywords map
    true_keywords_map = {}
    for i, keyword in enumerate(keywords):
        for k in keyword:
            if k in true_keywords_map:
                true_keywords_map[k] += [i]
            else:
                true_keywords_map[k] = [i]

    abstract = nltk.sent_tokenize(abstract)


    abstract = [re.sub(r'[^\x00-\x7F]+','', sentence) for sentence in abstract]
    abstract = [re.sub("[^a-zA-Z0-9\-,.]", " ", sentence).lower() for sentence in abstract]

    abstract = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in abstract]

    with open('C:\\Users\\Jona Q\\Documents\\GitHub\\Arxiv\\chunker.pickle', 'rb') as handle:
        chunker = pickle.load(handle)
    chunks = [chunker.parse(sentence) for sentence in abstract]

    tree_csm = ChunkTreeCSM(chunks, embeddings, tags)
    maxCsmOutput = tree_csm.selectKeywords(tags, 'max')
    print("maxCsmOutput")
    pprint.pprint(maxCsmOutput)
    maxCsmOutput = [[str(SnowballStemmer("english").stem(item)) for item in sublist if item not in stop_words]
                    for sublist in maxCsmOutput]

    [maxCsmPrecision, maxCsmRecall, maxCsmFmeasure] = computePerformance(maxCsmOutput, true_keywords_map, true_keywords_lengths)

    """print(file_name)
    print('keywords:')
    pprint.pprint(keywords)
    print('Min CSM output')
    pprint.pprint(minCsmOutput)
    print('Max CSM output')
    pprint.pprint(maxCsmOutput)


    print("Max-CSM: recall = ", maxCsmRecall, ", precision = ", maxCsmPrecision, ", f-measure = ", maxCsmFmeasure)"""
    wait = input("PRESS ENTER TO CONTINUE.")

    print('\n')


end = time.time()
print("Finished after ", end-start, "seconds")
