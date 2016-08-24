import pandas as pd
import numpy as np
import csv 
"""Embeddings reader and manager class
   Programmed by Jonathan Quijas
   Last modified 7/26/2016
"""

class EmbeddingsReader(object):
	def __init__(self, embedding_size = 50):
		self.embedding_size = embedding_size
		self.embeddings_path = '/home/jonathan/Documents/WordEmbedding/glove.6B.'+str(embedding_size)+'d.txt'
		#Safeguard vector for out of vocabulary words
		self.zero_vec = np.zeros(embedding_size)
		self.read_embs = False
	
	#Read embeddings file and store embeddings in object
	#Returns embeddings dictionary <key, value>
	def readEmbeddings(self, normalize=True):
		embeddings_path = '/home/jonathan/Documents/WordEmbedding/glove.6B.'+str(self.embedding_size)+'d.txt'
		data = pd.read_csv(embeddings_path, header=None, delimiter=' ', quoting=csv.QUOTE_NONE).as_matrix()
		[n, p] = data.shape
		if normalize:
			norms = np.linalg.norm(np.float64(data[:, 1:p]), axis = 1)
			self.data = dict(zip(data[:,0], data[:,1:p]/ norms[:,np.newaxis]))
			self.read_embs = True
			return self.data
		else:
			self.data = dict(zip(data[:,0], data[:,1:p]))
			self.read_embs = True
			return self.data

	def embeddingSize(self):
		return self.embedding_size
	
	#Return array with embedding vectors
	def valuesArray(self):
		if self.read_embs:
			return np.array(self.data.values())
		else:
			raise RuntimeError('Embeddings must be read first!')

	#Return array with embedding keys
	def keysArray(self):
		if self.read_embs:
			return np.array(self.data.keys())
		else:			
			raise RuntimeError('Embeddings must be read first!')
			
	def mapWords(self, message):
		return [[self.data.get(word, self.zero_vec) for word in sentence]for sentence in message]
