import rake
import operator
from nltk.corpus import stopwords
from bs4 import BeautifulSoup as Soup
import csv

min_num_chars = 3
n_gram = 3
min_occur = 1
rake_object = rake.Rake("/home/jonathan/Documents/WordEmbedding/Arxiv/Data/500common.txt", min_num_chars, n_gram, min_occur)

text_path = '/home/jonathan/Documents/WordEmbedding/Arxiv/Data/Text/Papers/herdedGibbsSampling.txt'

"""#Read categories file
categories_path = '/home/jonathan/Documents/WordEmbedding/Arxiv/Data/categories.txt'
cats_file = open(categories_path, 'r')
cats = list(csv.reader(cats_file))
cats = [cat[0] for cat in cats]
cats_file.close()
soupTag = 'summary'"""

text_data = open(text_path, 'r').read()
text_data = str(text_data)
keywords = rake_object.run(str(text_data))
print "Keywords:", keywords

"""for c_i, c in enumerate(cats):
	raw_data = open(text_path+c+'.txt', 'r').read()
	soup = Soup(raw_data)
	text_data = soup.findAll(soupTag)
	n = len(text_data)
	print n, ' messages extracted...'
	#For each message extracted
	for i, message in enumerate(text_data):
		message = str(message)
		print type(message)
		keywords = rake_object.run(str(message))
		print "Keywords:", keywords
"""
#Close files
