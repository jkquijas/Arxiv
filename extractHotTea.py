import numpy as np
from nltk.corpus import stopwords
import time
from collections import Counter
from TopicExtraction.KeywordExtractionModule import ChunkTreeCSM
from TopicExtraction.EmbeddingModule import EmbeddingsReader
from Chunking.ChunkingModule import ConsecutiveNPChunker
from Chunking import ChunkingModule
from TopicExtraction.ArxivDataModule import ArxivReader
import pickle
import numpy as np
import time
from nltk.corpus import stopwords
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
from HotTea.HotTea import Extractor


import pprint
import platform
#from slate import PDF

from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

def convert(fname, pages=None):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)

    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    infile = open(fname, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close
    return text


#Filter a noun phrase by tags
#Returns a list of tokens in noun_phrase whose part of speech is found in tag list 'tags'
def filterByTags(noun_phrase, tags):
    return [token[0] for token in noun_phrase if token[1] in tags]


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
    papers_path = "C:\\Users\\Jona Q\\Documents\GitHub\\Arxiv\\Data\\PDF\\Text\\asynchronousRL.txt"
    tagger_path = 'C:\\Users\\Jona Q\\Documents\\GitHub\\Arxiv\\chunker.pickle'
else:
    common_words_path = '/home/jonathan/Documents/WordEmbedding/Arxiv/Data/20k.txt'
    papers_path = "/home/jonathan/Documents/WordEmbedding/Arxiv/Data/PDF/Text/atari.txt"
    rake_common_path = "/home/jonathan/Documents/WordEmbedding/Arxiv/Data/500common.txt"
    tagger_path = "/home/jonathan/Documents/WordEmbedding/Arxiv/chunker2.pickle"



#
# Get embeddings
#
embedding_size = 100
embObj = EmbeddingsReader(embedding_size = embedding_size)
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



#   Read paper data
reader = ArxivReader()
paper = reader.readFileSplitByParagraphs(papers_path)

#   Load trained chunker
with open(tagger_path, 'rb') as handle:
    chunker = pickle.load(handle)

#   Chunk our paper
chunks = [[chunker.parse(sentence) for sentence in section] for section in paper]
#   For each section in the paper
results = [[]]
for i, section in enumerate(chunks):
    tree_csm = ChunkTreeCSM(section, embeddings, tags, embedding_size = embedding_size)
    maxCsmOutput = tree_csm.selectKeywords(tags, 'min')
    print("maxCsmOutput")
    pprint.pprint(maxCsmOutput)
    """maxCsmOutput = [[str(SnowballStemmer("english").stem(item)) for item in sublist if item not in stop_words]
                for sublist in maxCsmOutput]"""

    #if not empty(maxCsmOutput):
    results += [maxCsmOutput]
pprint.pprint(results)

extractor = Extractor(results)
extractor.createTopicHeap()
extractor.printTopics()

end = time.time()
print("Finished after ", end-start, "seconds")
