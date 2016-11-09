import pickle
import nltk
from Chunking.ChunkingModule import ConsecutiveNPChunker

tagger_path = "/home/jonathan/Documents/WordEmbedding/Arxiv/chunker.pickle"
f = open('chunker2.pickle', 'wb')
with open(tagger_path, 'rb') as handle:
    pickle.dump(pickle.load(handle), f, protocol=2)
handle.close()
f.close()
