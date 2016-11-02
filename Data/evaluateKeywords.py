import numpy as np
from nltk.corpus import stopwords
from os import listdir
import time

from TopicExtraction.KeywordExtractionModule import AbstractLIP
from TopicExtraction.EmbeddingModule import EmbeddingsReader
from TopicExtraction.ArxivDataModule import ArxivReader
from TopicExtraction.ArxivDataModule import ArxivManager

import numpy as np
import time
from nltk.corpus import stopwords
import nltk
from nltk.stem.snowball import SnowballStemmer

#from EvaluationModule import ZeroOneEvaluator
#from EvaluationModule import MSEEvaluator
#from EvaluationModule import MaxCosineEvaluator

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

#Unused
lambda_ = .1

#
# Clock in time
#
start = time.time()

if(platform.system() == 'Windows'):
	common_words_path = 'C:\\Users\\Jona Q\\Documents\GitHub\\Arxiv\\Data\\20k.txt'
	papers_path = "C:\\Users\\Jona Q\\Documents\GitHub\\Arxiv\\Data\\Text\\Papers\\"

else:
	common_words_path = '/home/jonathan/Documents/WordEmbedding/Arxiv/Data/20k.txt'
	papers_path = "/home/jonathan/Documents/WordEmbedding/Arxiv/Data/Text/Papers/"

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
file_list = listdir(papers_path)
for i, file_name in enumerate(file_list):

	#Read text and keywords
	[abstract, keywords] = arxivReader.readAbstractFileAndKeywords(papers_path+file_name)

	#Create chunks using our grammar
	chunks = [cp.parse(sentence) for sentence in abstract]
	# Filter out common words and words without the specified PoS tag
	abstract = [[ word[0] for word in sentence if word[1] in tags] for sentence in abstract]
	#Map all words to their embeddings
	abstract = [[embeddings.get(word) for word in sentences if word not in stop_words and embeddings.get(word) is not None] for sentences in abstract]

	#Extract keywords
	lip = AbstractLIP(abstract, embObj.embeddingSize(), lambda_, 'cosine')
	lip.selectKeywords()
	results = lip.getResults(values_array, keys_array)
	output = [[]]
    #Extract NP's, if any
	for i in range(len(results)):
		t = chunks[i]
		output = output + [traverseTree(t, results[i])]
	output = [l for l in output if l]

	#Print results on console
	print(file_name)
	print("extracted keywords:")
	pprint.pprint(output)
	print("true keywords:")
	pprint.pprint(keywords)

	output = [str(SnowballStemmer("english").stem(item)) for sublist in output for item in sublist]
	recall = 0.0
	n = len(keywords)
	for ki, k in enumerate(keywords):
		print("keyword ", str(ki+1),"/",n)
		k = [SnowballStemmer("english").stem(item) for item in k]
		pprint.pprint(k)
		for ii, item in enumerate(k):
			print("item", str(ii+1),"/",str(len(k)))
			if(item in output):
				print(item)
				recall = recall + 1
				break

	recall = recall/len(keywords)
	print("len(keywords) = ", len(keywords))
	print("Recall for paper", file_name, ": ", recall)
    #keywords = [SnowballStemmer("english").stem(item) for sublist in keywords for item in sublist]


end = time.time()
print(end-start)
