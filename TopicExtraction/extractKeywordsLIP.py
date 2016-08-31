import numpy as np
import time
from nltk.corpus import stopwords

#from LinearInformationPropagation import LIP
from LinearInformationPropagation import LIPGreedy
from EmbeddingModule import EmbeddingsReader
from ArxivDataModule import ArxivReader
from ArxivDataModule import ArxivManager
import pprint

from collections import Counter

lambda_ = .3

#Clock in time
start = time.time()

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
stop_words = [unicode(word) for word in stop_words]


arxivReader = ArxivReader()
text_data = arxivReader.readFileSplitByParagraphs("/home/jonathan/Documents/WordEmbedding/Arxiv/Data/Text/colosseum.txt")
print len(text_data), ' paragraphs\n'
results = []
for i in range(len(text_data)):
	print 'Paragraph ', i, '\n'
	message = text_data[i]
	#print message
	#Map all words to their embeddings
	message = [[embeddings.get(word) for word in sentences if word not in stop_words and embeddings.get(word) is not None] for sentences in message]
	#Check for empty sentences (after common and stop word removal)
	message = [sen for sen in message if len(sen) > 0]

	lip = LIPGreedy(message, embObj.embeddingSize(), lambda_, 'cosine')
	lip.computeBoundary()
	lip.selectKeywords()
	r = lip.getResults(values_array, keys_array)
	results += r

results.sort()
print results
cnt = Counter(results)
print 'Extracted keywords'
k = 1
print [k for k, v in cnt.iteritems() if v > k]



"""arxivManager = ArxivManager()
for c in arxivManager.categories():
	text_data = arxivManager.read(c)
	for i, message in enumerate(text_data):
		print 'Message ', i+1
		#Map all words to their embeddings
		message = [[embeddings.get(word) for word in sentences if word not in stop_words and embeddings.get(word) is not None] for sentences in message]
		#Check for empty sentences (after common and stop word removal)
		message = [sen for sen in message if len(sen) > 0]

		#message = embObj.mapWords(message)
		lip = LIPGreedy(message, embObj.embeddingSize(), lambda_)
		lip.computeBoundary()
		lip.selectKeywords()
		lip.printResults(values_array, keys_array)"""
