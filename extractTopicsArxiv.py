"""
Extract a set of Topics from Arxiv Data Repository
Programmed by Jonathan Quijas

"""
import numpy as np
import time
from nltk.corpus import stopwords

from TopicExtraction.KeywordExtractionModule import LIPGreedy
from TopicExtraction.KeywordExtractionModule import LIP
from TopicExtraction.EmbeddingModule import EmbeddingsReader
from TopicExtraction.ArxivDataModule import ArxivReader
from TopicExtraction.ArxivDataModule import ArxivManager
from TopicExtraction.TopicExtractor import TopicExtractorCount
from TopicExtraction.TopicExtractor import TopicExtractorKMeans

import pprint
import platform

from collections import Counter

lambda_ = .001

#Clock in time
start = time.time()
if(platform.system() == 'Windows'):
	common_words_path = 'C:\\Users\\Jona Q\\Documents\\Embeddings\\20k.txt'
	text_data_path = "C:\\Users\\Jona Q\\Documents\GitHub\\Arxiv\\Data\\Text\\voxelHashing.txt"

else:
	common_words_path = '/home/jonathan/Documents/WordEmbedding/20Newsgroups/Data/20k.txt'
	text_data_path = "/home/jonathan/Documents/WordEmbedding/Arxiv/Data/Text/donQuixote.txt"


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
else:
    stop_words = [unicode(word) for word in stop_words]

arxivManager = ArxivManager()
for c in arxivManager.categories():
	raw_data = arxivManager.read(c)
	for i, text_data in enumerate(raw_data):
		print(text_data)

		results = []
		for i in range(len(text_data)):
			print('Paragraph ', i, '\n')
			message = text_data[i]

			message = embObj.mapWords(message)

			lip = LIP(message, embObj.embeddingSize(), lambda_, 'cosine')
			lip.computeBoundary()
			lip.selectKeywords()
			r = lip.getResults(values_array, keys_array)
			results += r

		results.sort()

		#topic_extractor_type = 'count'
		topic_extractor_type = 'kmeans'

		if(topic_extractor_type == 'count'):
		    topicExtractor = TopicExtractorCount(results, k)
		    topicExtractor.extractTopics()
		elif(topic_extractor_type == 'kmeans'):
		    results = np.array([embeddings.get(word) for word in results])
		    k = 5
		    topicExtractor = TopicExtractorKMeans(results, k)
		    topicExtractor.extractTopics(values_array, keys_array)