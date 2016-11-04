import numpy as np
from nltk.corpus import stopwords
from os import listdir
import time

from TopicExtraction.KeywordExtractionModule import AbstractLIP
from TopicExtraction.EmbeddingModule import EmbeddingsReader
from TopicExtraction.ArxivDataModule import ArxivReader
from TopicExtraction.ArxivDataModule import ArxivManager

from RAKE_tutorial import rake

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

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
                        return [l[0] for l in subtree.leaves()]
                    else:
                        traverseTree(subtree, word)

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

#Unused
lambda_ = .1

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
# Initialize our file reader object
#
arxivReader = ArxivReader()

#
# Create stopwords set
#
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation
common_words_file = open(common_words_path, 'r')
common_words_list = [word for line in common_words_file for word in line.split()]
common_words_file.close()
num_common_words = 500
stop_words.update(common_words_list[0:num_common_words])
if(platform.system() == 'Windows'):
    stop_words = [word for word in stop_words]
else:
    stop_words = [unicode(word) for word in stop_words]



#
# Set up noun phrase (NP) grammar
#
tags = ['NN','NNS', 'JJ', 'RB', 'VBP', 'VB']
grammar = r"""
          NP: {<PP\$>?<JJ>*<NN|NNS|NNP>+<VBZ>*<JJ><NN|NNS>+}   # chunk determiner/possessive, adjectives and noun
          {<JJ>*<NN|NNS>*<NN|NNS><IN><JJ>*<NN|NNS>+}
          {<JJ><CC><JJ><NN|NNS>}
          {<JJ>*<NN|NNS>+}
          {<NN|NNS><IN><DT><NN|NNS>}
"""
cp = nltk.RegexpParser(grammar)

#For each paper
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

min_num_chars = 3
max_n_gram_size = 3
min_occur = 1
rake_object = rake.Rake(rake_common_path, min_num_chars, max_n_gram_size, min_occur)

recallFile = open('recall_results.txt', 'w')
precisionFile = open('precision_results.txt', 'w')
fmeasureFile = open('fmeasure_results.txt', 'w')


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


    #
    # RAKE
    #
    rakeOutput = rake_object.run(abstract)
    rakeOutput = [(rakeOutput[j][0]).split() for j in range(len(nltk.tokenize.sent_tokenize(abstract)))]
    rakeRetrieved = len(rakeOutput)
    rakeOutput = [str(SnowballStemmer("english").stem(item)) for sublist in rakeOutput for item in sublist]


    keywords = [str(data[i][key_concept][j][0]) for j in range(len(data[i][key_concept]))]
    keywords = [re.sub(r'[^\x00-\x7F]+','', k) for k in keywords]
    keywords = [re.sub("[^a-zA-Z0-9\-,]", " ", k).lower() for k in keywords]
    keywords = [k.split() for k in keywords]

    abstract = nltk.sent_tokenize(abstract)

    #
    # lda
    #
    n = len(keywords)
    n_features = 1000
    n_topics = n
    n_top_words = 2

    tfidf_vectorizer = TfidfVectorizer(max_features=n_features,stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(abstract)
    #CountVectorizer
    #TfidfVectorizer
    tf_vectorizer = CountVectorizer(max_features=n_features,stop_words='english')
    tf = tf_vectorizer.fit_transform(abstract)

    nmf = NMF(n_components=n_topics, random_state=0,alpha=.1, l1_ratio=.5, init = 'random')
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=10,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    lda.fit(tf)
    nmf.fit(tfidf)

    tf_feature_names = tf_vectorizer.get_feature_names()

    ldaOutput = get_topics(lda, tf_feature_names, n_top_words)
    ldaOutput = [(ldaOutput[j]).split() for j in range(len(ldaOutput))]
    ldaRetrieved = len(ldaOutput)
    ldaOutput = [str(SnowballStemmer("english").stem(item)) for sublist in ldaOutput for item in sublist]

    nmfOutput = get_topics(nmf, tf_feature_names, n_top_words)
    nmfOutput = [(nmfOutput[j]).split() for j in range(len(nmfOutput))]
    nmfRetrieved = len(nmfOutput)
    nmfOutput = [str(SnowballStemmer("english").stem(item)) for sublist in nmfOutput for item in sublist]



    abstract = [re.sub(r'[^\x00-\x7F]+','', sentence) for sentence in abstract]
    abstract = [re.sub("[^a-zA-Z0-9\-,]", " ", sentence).lower() for sentence in abstract]
    abstract = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in abstract]




    #Create chunks using our grammar
    chunks = [cp.parse(sentence) for sentence in abstract]
    # Filter out common words and words without the specified PoS tag
    abstract = [[ word[0] for word in sentence if word[1] in tags] for sentence in abstract]
    #Map all words to their embeddings
    abstract = [[embeddings.get(word) for word in sentences if word not in stop_words and embeddings.get(word) is not None] for sentences in abstract]
    abstract = [sentence for sentence in abstract if sentence]
    if empty(abstract):
        continue
    #Extract keywords
    lip = AbstractLIP(abstract, embObj.embeddingSize(), lambda_, 'cosine')
    lip.selectKeywords()
    results = lip.getResults(values_array, keys_array)
    output = [[]]
    #Extract NP's, if any
    for i in range(len(results)):
        t = chunks[i]
        output = output + [traverseTree(t, results[i])]
    outputRetrieved = len(output)
    output = [l for l in output if l]
    output = [str(SnowballStemmer("english").stem(item)) for sublist in output for item in sublist]

    #
    #Compute Results
    #
    hit = 0.0
    rakeHit = 0.0
    ldaHit = 0.0
    nmfHit = 0.0
    for ki, k in enumerate(keywords):
        k = [SnowballStemmer("english").stem(item) for item in k]
        #   LIP
        for ii, item in enumerate(k):
            if(item in output):
                hit = hit + 1
                break
        #       RAKE
        for ii, item in enumerate(k):
            if(item in rakeOutput):
                rakeHit = rakeHit + 1
                break
        #       LDA
        for ii, item in enumerate(k):
            if(item in ldaOutput):
                ldaHit = ldaHit + 1
                break
        #       NMF
        for ii, item in enumerate(k):
            if(item in nmfOutput):
                nmfHit = nmfHit + 1
                break


    #Recall
    recall = hit/n
    rakeRecall = rakeHit/n
    ldaRecall = ldaHit/n
    nmfRecall = nmfHit/n
    #Precision
    precision = hit/outputRetrieved
    rakePrecision = rakeHit/rakeRetrieved
    ldaPrecision = ldaHit/ldaRetrieved
    nmfPrecision = nmfHit/nmfRetrieved
    #F-measure
    if recall + precision == 0:
        fmeasure = 0
    else:
        fmeasure = 2*((precision*recall)/(precision+recall))

    if rakePrecision + rakeRecall == 0:
        rakeFmeasure = 0
    else:
        rakeFmeasure = 2*((rakePrecision*rakeRecall)/(rakePrecision+rakeRecall))

    if ldaPrecision + ldaRecall == 0:
        ldaFmeasure = 0
    else:
        ldaFmeasure = 2*((ldaPrecision*ldaRecall)/(ldaPrecision+ldaRecall))

    if nmfRecall + nmfPrecision == 0:
        nmfFmeasure = 0
    else:
        nmfFmeasure = 2*((nmfPrecision*nmfRecall)/(nmfPrecision+nmfRecall))

    print(file_name)
    print("LIP", file_name, ": recall = ", recall, ", precision = ", precision, ", f-measure = ", fmeasure)
    print("RAKE: recall = ", rakeRecall, ", precision = ", rakePrecision, ", f-measure = ", rakeFmeasure)
    print("LDA: recall = ", ldaRecall, ", precision = ", ldaPrecision, ", f-measure = ", ldaFmeasure)
    print("NMF: recall = ", nmfRecall, ", precision = ", nmfPrecision, ", f-measure = ", nmfFmeasure)
    print('\n')

    recallFile.write(str(recall)+","+str(rakeRecall)+","+str(ldaRecall)+","+str(nmfRecall)+"\n")
    precisionFile.write(str(precision)+","+str(rakePrecision)+","+str(ldaPrecision)+","+str(nmfPrecision)+"\n")
    fmeasureFile.write(str(fmeasure)+","+str(rakeFmeasure)+","+str(ldaFmeasure)+","+str(nmfFmeasure)+"\n")


recallFile.close()
precisionFile.close()
fmeasureFile.close()

end = time.time()
print("Finished after ", end-start, "seconds")
