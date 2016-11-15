import heapq
import math
import numpy as np


class Extractor(object):
    def __init__(self, data):
        self.HM = {}
        #Fill up our <Keyword, Section Index> hashmap
        for i, section in enumerate(data):
            print(section)
            #Skip empty results
            """if empty(section):
                continue"""
            #Map each keyword to its section of occurrance
            for j, keywords in enumerate(section):
                for k, keyword in enumerate(keywords):
                    print(keyword)
                    if keyword in self.HM:
                        self.HM[keyword] = self.HM[keyword] + [i]
                    else:
                        self.HM[keyword] = [i]

    def createTopicHeap(self):
        self.heap = [(len(v)*-1, k) for k, v in self.HM.items() if len(v) > 4]
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
        self.embeddings_data = [[]]
        for i, section in enumerate(data):
            #Map each keyword to its section of occurrance
            for j, keywords in enumerate(section):
                self.embeddings_data += [[embeddings.get(word,self.zero_vec) for word in keywords]]

    def buldTopicStructure(self):
        n_sections = len(self.data)
        #Bottom up computing of topic relevance
        for i in range(n_sections-1, 0, -1):
            x = np.array(self.embeddings_data[i])
            y = np.array(self.embeddings_data[i-1])
            x_shape = x.shape
            y_shape = y.shape
            if x_shape[0] == 0 or y_shape[0] == 0:
                continue
            R = np.dot(x, y.transpose())
            [r,c] = R.shape


            for ii in range(r):
                for jj in range(c):
                    if abs(R[ii][jj]) >= self.threshold:
                        pass
