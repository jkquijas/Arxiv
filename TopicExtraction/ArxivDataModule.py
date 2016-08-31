from nltk.tokenize import TreebankWordTokenizer
from bs4 import BeautifulSoup as Soup
import re
import nltk
import csv
import numpy as np
import urllib

class ArxivObj(object):

	#True if writing text data to files, False otherwise
	write_files = True
	#Data path
	text_path = '/home/jonathan/Documents/WordEmbedding/Arxiv/Data/Text/'
	#Path to text file specifying search categories e.g. Smart Cities, Cyber Security, etc. One category per line
	categories_path = '/home/jonathan/Documents/WordEmbedding/Arxiv/Data/categories.txt'
	soupTag = 'summary'

	def __init__(self):
		#Read categories file
		cats_file = open(ArxivObj.categories_path, 'r')
		cats = list(csv.reader(cats_file))
		cats = [cat[0] for cat in cats]
		cats_file.close()

		self.cats = cats



class ArxivCrawler(ArxivObj):
	def __init__(self, n_results=5, search_field='cat'):
		super(self.__class__, self).__init__()

		#Append URL string together
		url_start = 'http://export.arxiv.org/api/query?search_query='+search_field+':'
		url_middle = '&start=0&max_results='+str(n_results)
		url_end = '&sortBy=lastUpdatedDate&sortOrder=descending'

		#Create URL Strings
		self.urls = [url_start + c + url_middle + url_end for c in self.cats]


	def crawl(self):
		files = [open(ArxivObj.text_path+c+'.txt', 'w') for c in self.cats]
		#String in quotes is the field will extract
		event_data_location = lambda x: x.name == ArxivObj.soupTag
		#Read data
		for i, url in enumerate(self.urls):
			data = urllib.urlopen(url).read()
			#Create Soup XML parser
			soup = Soup(data)
			events = soup.findAll(event_data_location)
			print 'Processing search: ', url
			if(events):
				for j, e in enumerate(events):
					print j
					e = str(e)
					files[i].write(e)

		#Close file handlers
		for f in files:
			f.close()


class ArxivReader(ArxivObj):
	def __init__(self):
		super(self.__class__, self).__init__()

	#Return categories list
	def categories(self):
		return self.cats

	def readFile(self, category):
		raw_data = open(ArxivObj.text_path+category+'.txt', 'r').read()
		soup = Soup(raw_data)
		text_data = soup.findAll(ArxivObj.soupTag)
		text_data = [nltk.sent_tokenize(str(message).strip('<'+ArxivObj.soupTag+'></'+ArxivObj.soupTag+'>')) for message in text_data]
		#text_data = [[TreebankWordTokenizer().tokenize(re.sub("[-\/]", " ", sentence).lower()) for sentence in message] for message in text_data]
		text_data = [[TreebankWordTokenizer().tokenize(re.sub("[^a-zA-Z0-9\-]", " ", sentence).lower()) for sentence in message] for message in text_data]
		return text_data

	def readFileSplitByParagraphs(self, filePath):
		raw_data = open(filePath, 'r').read()
		text_data = re.split('\n\n',raw_data)
		text_data = [re.sub(r'[^\x00-\x7F]+','', message) for message in text_data]
		print text_data
		text_data = [nltk.sent_tokenize(str(message)) for message in text_data]
		#text_data = [re.sub(r'\$.*?\$', '', message) for message in text_data]
		#text_data = [re.sub(r'\[.*?\]', '', message) for message in text_data]
		text_data = [[TreebankWordTokenizer().tokenize(re.sub("[^a-zA-Z0-9\-]", " ", sentence).lower()) for sentence in message] for message in text_data]
		return text_data

class ArxivWriter(ArxivObj):
	data_path = '/home/jonathan/Documents/WordEmbedding/Arxiv/Data/Keywords/'

	def __init__(self):
		super(self.__class__, self).__init__()
		#Read data corresponding to the categories chosen
		self.files = [open(ArxivWriter.data_path+c+'.txt', 'w') for c in self.cats]

	def writeToFile(self, cat_ind, data):
		np.savetxt(self.files[cat_ind], data, delimiter=',')

	def close(self):
		#Close files
		for f in self.files:
			f.close()

class ArxivManager(object):
	def __init__(self):
		self.reader = ArxivReader()
		self.writer = ArxivWriter()
	def categories(self):
		return self.reader.categories()
	def read(self, category):
		return self.reader.readFile(category)
	def write(self, cat_ind, data):
		self.writer.writeToFile(cat_ind, data)
	def close(self):
		self.writer.close()
