import numpy as np

from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist

from operator import itemgetter
import itertools

from functools import reduce # Valid in Python 2.6+, required in Python 3 6
import operator

"""Linear Information Propagation Model Class
   Programmed by Jonathan Quijas
   Last modified 09/25/2016
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
        """if(self.dist_metric == 'cosine'):
            self.gamma_min = sum(s[0][0] for s in self.S)
            self.gamma_max = sum(s[len(s)-1][0] for s in self.S)
        else:
            self.gamma_min = sum(s[0][0] for s in self.S)
            self.gamma_max = sum(s[len(s)-1][0] for s in self.S)"""


    def computeBoundary(self, debug=True):
        #Compute budget!
        budget = self.gamma_min + self.lambda_*(self.gamma_max-self.gamma_min)
        if debug:
            print('Budget = ', budget)

        gain = (budget)/self.gamma_min
        gamma_hat = self.gamma_min

        iter_count = 1.0
        R = np.zeros((self.n_sen, self.embedding_size))
        for k in range(self.n_sen):
            R[k,:] = self.S[k][0][1]

        delta_0 = np.sum(np.sum(np.dot(R, R.transpose())))
        """print 'delta_0 = ', delta_0"""


        #Compute search boundaries defined by gamma_hat
        while (gamma_hat <= budget and iter_count < self.long_sent_len):
            search_prcnt = iter_count/self.long_sent_len
            changes = [0 for k in range(self.n_sen)]
            #Determine new search index for each sentence
            for k in range(self.n_sen):
                if((self.search_idx_vec[k]+2.0) / self.sen_len_vec[k] <= search_prcnt):
                    self.search_idx_vec[k] = self.search_idx_vec[k]+1
                    changes[k] = -1
            #Compute new gamma_hat
            gamma_hat = 0
            gamma_hat = sum(s[self.search_idx_vec[k]][0] for k, s in enumerate(self.S))


            #If we passed our budget, revert changes and exit
            if gamma_hat > budget:
                gamma_hat = 0
                self.search_idx_vec = [sum(x) for x in zip(self.search_idx_vec, changes)]
                gamma_hat = sum(s[self.search_idx_vec[k]][0] for k, s in enumerate(self.S))
                break

            iter_count = iter_count + 1.0

        if debug:
            print('gamma_hat = ', gamma_hat)
            print('search indexes = ', zip(self.search_idx_vec, self.sen_len_vec))
            print('Reduced search space to ', 100.0*(float(sum(np.array(self.search_idx_vec)+1.0))/float(sum(self.sen_len_vec))), ' % of original space')

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

        R = np.zeros((self.n_sen, self.embedding_size))
        for k in range(self.n_sen):
            R[k,:] = self.S[k][chosen_idxs[k]][1]

        delta_hat = np.sum(np.sum(np.dot(R, R.transpose())))
        if debug:
            print('delta_hat = ', delta_hat)

        self.chosen_idxs = chosen_idxs

    def getResults(self, values_array, keys_array):
        #Check for empty sentences (after common and stop word removal)
        results = []
        for k in range(self.n_sen):
            emb = (self.S[k][self.chosen_idxs[k]][1])
            idx = np.where(values_array == emb)[0][0]
            results += [keys_array[idx]]
        return results

    def printResults(self, values_array, keys_array):
        #Check for empty sentences (after common and stop word removal)
        for k in range(self.n_sen):
            emb = (self.S[k][self.chosen_idxs[k]][1])
            idx = np.where(values_array == emb)[0][0]
            print('s', k, ': ', keys_array[idx])



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
        self.chosen_idxs = list(min_ind)

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
            if(euclidean(self.S[s_i][i][1], self.S[s_j][j][1]) == 0.0 ):
                dot = -100000000
            else:
                dot = np.dot(self.S[s_i][i][1], self.S[s_j][j][1])
        else:
            dot = euclidean(self.S[s_i][i][1], self.S[s_j][j][1])

        self.dots[str(s_i)+str(i)][(str(s_j)+str(j))] = dot
        return dot

def flat_tuple(a):
    if(type(a) not in (tuple, list)):
        return (a,)
    if(len(a) == 0):
        return tuple(a)
    return flat_tuple(a[0]) + flat_tuple(a[1:])
