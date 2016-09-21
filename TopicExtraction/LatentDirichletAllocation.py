from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')
#en_stop = set(stopwords.words('english'))
#en_stop = [str(i) for i in en_stop]



# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# create sample documents

# compile sample documents into a list
doc_set = [open("/home/jonathan/Documents/WordEmbedding/Arxiv/Data/Text/grapes.txt").read()]
# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:

    # clean and tokenize document string
    raw = i.lower()

    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop and i != '\xe2']
    print stopped_tokens
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    print stemmed_tokens


    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=4, id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=4, num_words=4))
