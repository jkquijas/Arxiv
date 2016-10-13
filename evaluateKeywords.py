import numpy as np
from nltk.corpus import stopwords

from EvaluationModule import ZeroOneEvaluator
#from EvaluationModule import MSEEvaluator
#from EvaluationModule import MaxCosineEvaluator

import pprint
import platform

lambda_ = .1

#Clock in time
start = time.time()
if(platform.system() == 'Windows'):
	common_words_path = 'C:\\Users\\Jona Q\\Documents\\Embeddings\\20k.txt'
	text_data_path = "C:\\Users\\Jona Q\\Documents\GitHub\\Arxiv\\Data\\Text\\voxelHashing.txt"
    keywords_path = "C:\\Users\\Jona Q\\Documents\GitHub\\Arxiv\\Data\\Text\\voxelHashing.txt"

else:
	common_words_path = '/home/jonathan/Documents/WordEmbedding/20Newsgroups/Data/20k.txt'
	text_data_path = "/home/jonathan/Documents/WordEmbedding/Arxiv/Data/Text/donQuixote.txt"
    keywords_path = "/home/jonathan/Documents/WordEmbedding/Arxiv/Data/Text/donQuixote.txt"

evaluator  = ZeroOneEvaluator()
