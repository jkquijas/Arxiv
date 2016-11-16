import heapq
import math
import numpy as np
from nltk.stem.snowball import SnowballStemmer



class Extractor(object):
    def __init__(self, data, embedding_size=50):
        self.HM = {}
        self.min_occur = 3
        self.zero_vec = np.zeros(embedding_size)
        self.threshold = .5

        #Fill up our <Keyword, Section Index> hashmap
        for i, section in enumerate(data):
            #Map each keyword to its section of occurrance
            for j, keywords in enumerate(section):
                for k, keyword in enumerate(keywords):
                    #
                    stem_keyword = str(SnowballStemmer("english").stem(keyword))
                    if stem_keyword in self.HM:
                        self.HM[stem_keyword] = self.HM[stem_keyword] + [((i,j), keyword)]
                    else:
                        self.HM[stem_keyword] = [(i, keyword)]
                    #

                    """if keyword in self.HM:
                        self.HM[keyword] = self.HM[keyword] + [i]
                    else:
                        self.HM[keyword] = [i]"""


    def topicMatrix(self, embeddings):
        L = [v for K, V in self.HM.items() for v in V if len(V) >= self.min_occur]
        M = np.array([embeddings.get(v[1],self.zero_vec) for v in L])
        """L = [K for K, V in self.HM.items()]
        M = np.array([embeddings.get(v,self.zero_vec) for v in L])
        print(M)"""
        print("Computing CS matrix...")
        R = np.dot(M, M.transpose()).tolist()
        print(len(R))
        x = [[idx+i+1 for idx, val in enumerate(R[i][i+1:]) if abs(val) > self.threshold] for i in range(len(R)-1)]
        print(x)
        print(len(x))

    def createTopicHeap(self):
        self.heap = [(len(v)*-1, set([w for (c,w) in v])) for k, v in self.HM.items() if len(v) >= self.min_occur]
        heapq.heapify(self.heap)

    def printTopics(self):
        n = len(self.heap)
        print(n," items in heap")
        hor_space = int(n)
        ver_space = 3
        level = 0
        q = [0]
        while q:
            k = q.pop(0)
            if math.ceil(math.log(k+1,2)) != level:
                level += 1
                hor_space /= 2
                print("\n\n\n\n")

            #Print horizontal spaces
            for h in range(int(hor_space)):
                print(" ", end="")
            #Print word
            print(self.heap[k], end="")
            #Print horizontal spaces
            for h in range(int(hor_space)):
                print(" ", end="")


            l = 2*k+1
            r = 2*k+2
            if l < n:
                q += [l]
            if r < n:
                q += [r]


class FIPExtractor(object):
    def __init__(self, data, embeddings, embedding_size = 50, threshold = .5):
        self.threshold = threshold
        self.data = data
        self.zero_vec = np.zeros(embedding_size)
        self.embeddings_data = [[[]]]
        for section in self.data:
            self.embeddings_data += [[[embeddings.get(word,self.zero_vec) for word in keywords] for keywords in section]]
        self.embeddings_data = self.embeddings_data[1:]

    def buldTopicStructure(self):
        n_sections = len(self.embeddings_data)
        #Bottom up computing of topic relevance
        for i in range(n_sections-1, 0, -1):
            #x = np.array(self.embeddings_data[i][0])
            #y = np.array(self.embeddings_data[i-1][0])
            x = np.array([k for j in range(len(self.embeddings_data[i])) for k in self.embeddings_data[i][j]])
            y = np.array([k for j in range(len(self.embeddings_data[i-1])) for k in self.embeddings_data[i-1][j]])

            x_shape = x.shape
            y_shape = y.shape
            """print(x_shape)
            print(y_shape)
            print('@@@@@@@@')"""
            if x_shape[0] == 0 or y_shape[0] == 0:
                continue
            R = np.dot(x, y.transpose())
            [r,c] = R.shape
            for ii in range(r):
                for jj in range(c):
                    if abs(R[ii][jj]) >= self.threshold:
                        pass
