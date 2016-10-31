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

#from EvaluationModule import ZeroOneEvaluator
#from EvaluationModule import MSEEvaluator
#from EvaluationModule import MaxCosineEvaluator

import pprint
import platform

lambda_ = .1

#Clock in time
start = time.time()
if(platform.system() == 'Windows'):
	common_words_path = 'C:\\Users\\Jona Q\\Documents\\Embeddings\\20k.txt'
	papers_path = "C:\\Users\\Jona Q\\Documents\GitHub\\Arxiv\\Data\\Text\\Papers\\"

else:
	common_words_path = '/home/jonathan/Documents/WordEmbedding/20Newsgroups/Data/20k.txt'
	papers_path = "/home/jonathan/Documents/WordEmbedding/Arxiv/Data/Text/Papers/"

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
num_common_words = 500
#Update stopwrd set with common words
stop_words.update(common_words_list[0:num_common_words])
#Convert to unicode
if(platform.system() == 'Windows'):
	stop_words = [word for word in stop_words]
else:
	stop_words = [unicode(word) for word in stop_words]


arxivReader = ArxivReader()
tags = ['NN','NNS', 'JJ', 'RB', 'VBP', 'VB']
#tags=['NN','NNS']
#tags = ['NN','NNS', 'VBP', 'VB']

grammar = r"""
		  NP: {<PP\$>?<JJ>*<NN|NNS|NNP>+<VBZ>*<JJ><NN|NNS>+}   # chunk determiner/possessive, adjectives and noun
		  {<JJ>*<NN|NNS>*<NN|NNS><IN><JJ>*<NN|NNS>+}
		  {<JJ><CC><JJ><NN|NNS>}
		  {<JJ>*<NN|NNS>+}
		  {<NN|NNS><IN><DT><NN|NNS>}
"""
cp = nltk.RegexpParser(grammar)

#For each paper
#Clock in time
start = time.time()

file_list = listdir(papers_path)
for i, file_name in enumerate(file_list):

	[text_data, keywords] = arxivReader.readAbstractFileAndKeywords(papers_path+file_name)

	print(keywords)
	results = []
	for i in range(len(text_data)):
		message = text_data[i]
		#Noun phrase chunking
		chunks = [cp.parse(sentence) for sentence in message]

		# Filter out common words and words without the specified PoS tag
		message = [[ word[0] for word in sentence if word[1] in tags] for sentence in message]
		#Map all words to their embeddings
		message = [[embeddings.get(word) for word in sentences if word not in stop_words and embeddings.get(word) is not None] for sentences in message]
		#Check for empty sentences (after common and stop word removal)
		#message = [sen for sen in message if len(sen) > 3]

		lip = AbstractLIP(message, embObj.embeddingSize(), lambda_, 'cosine')
		lip.computeBoundary()
		lip.selectKeywords()
		r = lip.getResults(values_array, keys_array)
		results += r

	print('Keywords extracted:\n', results)
	output = [[]]



	for i in range(len(results)):
		t = chunks[i]
		output = output + [traverseTree(t, results[i])]
	output = [l for l in output if l]
	print(output)


end = time.time()
print(end-start)

def traverseTree(tree, word):
	for subtree in tree:
		if type(subtree) == nltk.tree.Tree:
			if subtree._label == 'NP':
				for leave in subtree.leaves():
					if word == leave[0]:
						return [l[0] for l in subtree.leaves()]
					else:
						traverseTree(subtree)
