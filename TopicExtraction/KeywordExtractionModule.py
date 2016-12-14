import numpy as np

from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist

from operator import itemgetter
import itertools
import nltk

from functools import reduce # Valid in Python 2.6+, required in Python 3 6
import operator

"""Keyword Extraction Model Class
   Programmed by Jonathan Quijas
   Last modified 10/13/2016
"""
class LIP(object):
    def __init__(self, message, embedding_size, lambda_ = 0.2, dist_metric = 'cosine'):
        self.dist_metric = dist_metric
        self.embedding_size = embedding_size
        self.lambda_ = lambda_
        #Compute longest sentence length
        self.long_sent_len = float(max([len(sentence) for sentence in message]))
        #Compute similarity vectors, and zip with sentences
        if(self.dist_metric == 'cosine'):
            S = [list(zip(np.sum(np.dot(np.array(sentence), np.array(sentence).transpose()), axis=0), sentence)) for sentence in message]
        else:
            S = [list(zip(np.sum(cdist(np.array(sentence), np.array(sentence), 'euclidean'), axis=0), sentence)) for sentence in message]

        #Sort sentences by similarity
        if(self.dist_metric == 'cosine'):
            self.S = [sorted(s, key=itemgetter(0)) for s in S]
        else:
            self.S = [sorted(s, key=itemgetter(0), reverse=True) for s in S]

        self.n_sen = len(self.S)

        #Initialize search index to 0 (Minimum within sentence cosine similarity)
        self.search_idx_vec = [0 for s in message]
        #Compute length of sentences
        self.sen_len_vec = [float(len(s)) for s in message]


        #Compute gamma boundaries
        self.gamma_min = sum(s[0][0] for s in self.S)
        self.gamma_max = sum(s[len(s)-1][0] for s in self.S)


    def computeBoundary(self, debug = True):
        for k, s in enumerate(self.S):
            if(self.dist_metric == 'cosine'):
                budget = s[0][0] + self.lambda_*(s[len(s)-1][0]-s[0][0])
                while(s[self.search_idx_vec[k]][0] < budget):
                    self.search_idx_vec[k] = self.search_idx_vec[k] + 1
            else:
                budget = s[0][0] - self.lambda_*(s[0][0]-s[len(s)-1][0])
                while(s[self.search_idx_vec[k]][0] > budget):
                    self.search_idx_vec[k] = self.search_idx_vec[k] + 1

        if(debug):
            print('search indexes = ', list(zip([x+1 for x in self.search_idx_vec], self.sen_len_vec)))
            print('Reduced search space to ', 100.0*(float(sum(np.array(self.search_idx_vec)+1.0))/float(sum(self.sen_len_vec))),' % of original space')

		#Create indexing dictionary
        self.dots = {}
        for i in range(self.n_sen):
            for j in range(self.search_idx_vec[i]+1):
                self.dots[str(i)+str(j)] = {}

    def selectKeywords(self, debug=True):
        # Linear Information Propagation (LIP) Search
        sim = 0.0
        chosen_idxs = [0 for k in range(self.n_sen)]
        delta = 0.0

        for ii in range(0, self.search_idx_vec[0]+1):
            for jj in range(0, self.search_idx_vec[1]+1):
                sim = np.dot(self.S[0][ii][1], self.S[1][jj][1])
                if(sim > delta):
                    delta = sim
                    chosen_idxs[0] = ii
                    chosen_idxs[1] = jj

        #For all remaining sentences 3:self.n_sen
        for k in range(2, self.n_sen):
            delta = 0.0
            #For kth sentence search range
            for ii in range(0, self.search_idx_vec[k]+1):
                sim = np.dot(self.S[k-1][self.search_idx_vec[k-1]][1], self.S[k][ii][1])
                if(sim > delta):
                    delta = sim
                    chosen_idxs[k] = ii
        self.chosen_idxs = chosen_idxs

    def getResults(self, values_array, keys_array):
        #Check for empty sentences (after common and stop word removal)
        results = []
        for k in range(self.n_sen):
            emb = (self.S[k][self.chosen_idxs[k]][1])
            #idx = [np.array_equal(emb,x) for x in values_array].index(True)
            idx = np.where(values_array == emb)[0][0]
            results += [keys_array[idx]]
        return results

    def printResults(self, values_array, keys_array):
        #Check for empty sentences (after common and stop word removal)
        for k in range(self.n_sen):
            emb = (self.S[k][self.chosen_idxs[k]][1])
            idx = np.where(values_array == emb)[0][0]
            print('s', k, ': ', keys_array[idx])
            worst_emb = (self.S[k][self.worst_idx[k]][1])
            idx = np.where(values_array == worst_emb)[0][0]
            print('worst ', k, ':', )



class LIPGreedy(LIP):
    def __init__(self, message, embedding_size, lambda_ = 0.2, dist_metric = 'cosine'):
        super(self.__class__, self).__init__(message, embedding_size, lambda_, dist_metric)


    def computeBoundary(self, debug = True):
        for k, s in enumerate(self.S):
            if(self.dist_metric == 'cosine'):
                budget = s[0][0] + self.lambda_*(s[len(s)-1][0]-s[0][0])
                while(s[self.search_idx_vec[k]][0] < budget):
                    self.search_idx_vec[k] = self.search_idx_vec[k] + 1
            else:
                budget = s[0][0] - self.lambda_*(s[0][0]-s[len(s)-1][0])
                while(s[self.search_idx_vec[k]][0] > budget):
                    self.search_idx_vec[k] = self.search_idx_vec[k] + 1

        if(debug):
            print('search indexes = ', list(zip([x+1 for x in self.search_idx_vec], self.sen_len_vec)))
            print('Reduced search space to ', 100.0*(float(sum(np.array(self.search_idx_vec)+1.0))/float(sum(self.sen_len_vec))),' % of original space')

		#Create indexing dictionary
        self.dots = {}
        for i in range(self.n_sen):
            for j in range(self.search_idx_vec[i]+1):
                self.dots[str(i)+str(j)] = {}


    def selectKeywords(self, debug=True):
		#Produce list of index permutations
        permutations = [k for k in range(self.search_idx_vec[0]+1)]
        for i in range(1, self.n_sen):
            permutations = list(itertools.product(permutations, [(k) for k in range(self.search_idx_vec[i]+1)]))
        permutations = [flat_tuple(a) for a in permutations]
        print('Number of permutations needed: ', len(permutations))
        costs =  [self.computeCost(perm) for perm in permutations]

        min_ind = permutations[np.argmin(costs)]
        max_ind = permutations[np.argmax(costs)]
        self.chosen_idxs = list(min_ind)
        self.worst_idxs = list(max_ind)

        if(debug):
            costs.sort()
            print('Min cost = ', costs[0])
            print('Max cost = ', costs[len(costs) - 1])

    def computeCost(self, indexes):
        if(self.dist_metric == 'cosine'):
            gamma = sum([self.S[i][indexes[i]][0] for i in range(self.n_sen)])
        else:
            gamma = sum([self.S[i][indexes[i]][0] for i in range(self.n_sen)])
		#For each sentence
        delta = 0.0
        for i in range(self.n_sen):
            #Add all dot products
            delta += sum([self.computeDot(i, indexes[i], j, indexes[j]) for j, idx in enumerate(indexes)])
        if(self.dist_metric == 'cosine'):
            return gamma-delta
        else:
            return -1*gamma+delta


    def computeDot(self, s_i, i, s_j, j):
        if((str(s_j)+str(j)) in self.dots[str(s_i)+str(i)]):
            return self.dots[str(s_i)+str(i)][(str(s_j)+str(j))]

        if((str(s_i)+str(i)) in self.dots[str(s_j)+str(j)]):
            dot = self.dots[str(s_j)+str(j)][(str(s_i)+str(i))]
            self.dots[str(s_i)+str(i)][(str(s_j)+str(j))] = dot
            return dot
        if(self.dist_metric == 'cosine'):
            dot = np.dot(self.S[s_i][i][1], self.S[s_j][j][1])
        else:
            dot = euclidean(self.S[s_i][i][1], self.S[s_j][j][1])

        self.dots[str(s_i)+str(i)][(str(s_j)+str(j))] = dot
        return dot

class AbstractLIP(LIP):
    def __init__(self, message, embedding_size, lambda_ = 0.2, dist_metric = 'cosine'):
        super(self.__class__, self).__init__(message, embedding_size, lambda_, dist_metric)

    def selectKeywords(self, debug=True):
        self.chosen_idxs = [0 for self.s in self.S]
        self.keywordMap = {}
        for i, s in enumerate(self.S):
            #Avoid repeated words
            while(np.array_str(s[self.chosen_idxs[i]][1]) in self.keywordMap and self.chosen_idxs[i] + 1 < len(s)):
                self.chosen_idxs[i] =  self.chosen_idxs[i] + 1
            self.keywordMap[np.array_str(s[self.chosen_idxs[i]][1])] = True

class ChunkTreeCSM(object):
    def __init__(self, tree, embeddings, tags, embedding_size=50, use_zero_vec=True):
        self.use_zero_vec = use_zero_vec
        self.nps = []
        self.embeddings = embeddings
        self.zero_vec = np.zeros(embedding_size)
        self.embedding_size = embedding_size
        #For each sentence
        for t in tree:
            sentence_nps = []
            #For each subtree in the sentence
            for subtree in t:
                #If it is a NP, add to list
                if type(subtree) == nltk.tree.Tree:
                    sentence_nps = sentence_nps + [subtree]
            if sentence_nps:
                self.nps = self.nps + [sentence_nps]

        self.abstract = []
        for sentence in self.nps:
            result = [[[(embeddings.get(word[0],self.zero_vec), word[1]) for word in n_p.leaves()] for n_p in sentence]]
            #result = [[[(embeddings.get(word[0]), word[1]) for word in n_p.leaves() if embeddings.get(word[0])
            #                                  is not None and word[1] in tags] for n_p in sentence]]
            self.abstract = self.abstract + result
        #self.abstract = [sentence for sentence in self.abstract if sentence]
        self.abstract = [[n_p for n_p in sentence if n_p]for sentence in self.abstract]
        self.abstract = [sentence for sentence in self.abstract if sentence]

    def selectKeywords(self, tags, type, feature_extraction = False):
        results = []
        #For each sentence
        for i, sentence in enumerate(self.abstract):
            #print sentence
            #For each noun phrase in this sentence
            avg_cos_sim_list = []
            for noun_phrase in sentence:
                np_emb = np.array([np_i[0] for np_i in noun_phrase])
                mean_ = np.mean(np.mean(np.dot(np.array(np_emb), np.array(np_emb).transpose()), axis=0))
                #mean_ = np.mean(np.sum(np.dot(np.array(np_emb), np.array(np_emb).transpose()), axis=0))
                avg_cos_sim_list += [mean_]
            #find np with optimal mean within-phrase cosine similarity
            if type == 'max':
                cos_sim_np_idx = np.argmax(avg_cos_sim_list)
            else:
                cos_sim_np_idx = np.argmin(avg_cos_sim_list)
            selected_np = self.nps[i][cos_sim_np_idx]
            #Filter selected noun phrase to output only specified PoS
            if feature_extraction:
                r = [self.getEmbedding(leave[0]) for leave in selected_np.leaves() if leave[1] in tags]
            else:
                r = [leave[0] for leave in selected_np.leaves() if leave[1] in tags]
            if r:
                results += [r]
        return results

    def getEmbedding(self, word):
        if word in self.embeddings:
            return self.embeddings.get(word)

        if self.use_zero_vec:
            new_emb = self.zero_vec
            self.embeddings[word] = new_emb
        else:
            new_emb = np.random.rand(self.embedding_size)
            self.embeddings[word] = new_emb

        return new_emb



def flat_tuple(a):
    if(type(a) not in (tuple, list)):
        return (a,)
    if(len(a) == 0):
        return tuple(a)
    return flat_tuple(a[0]) + flat_tuple(a[1:])
