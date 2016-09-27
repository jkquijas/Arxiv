import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import platform

class TopicExtractor(object):
    def __init__(self, keywords):
        self.keywords = keywords

class TopicExtractorCount(TopicExtractor):
    def __init__(self, keywords,k=1):
        super(self.__class__, self).__init__(keywords)
        self.k = k
    def extractTopics(self):
        print(self.keywords)
        cnt = Counter(self.keywords)
        if(platform.system() == 'Windows'):
        	print([word for word, occur in iter(cnt.items()) if occur > self.k])
        else:
            print([word for word, occur in cnt.iteritems() if occur > self.k])


class TopicExtractorKMeans(TopicExtractor):
    def __init__(self, keywords,k=5):
        super(self.__class__, self).__init__(keywords)
        self.k = min(k, len(keywords))
        self.kmeans = KMeans(n_clusters=self.k)
        self.keywords = keywords
        self.nbrs = NearestNeighbors(n_neighbors=1).fit(self.keywords)

    def extractTopics(self, values_array, keys_array):
        self.kmeans.fit(self.keywords)
        centroids = np.array(self.kmeans.cluster_centers_)
        self.results = list(self.nbrs.kneighbors(X=centroids, n_neighbors=1, return_distance=False))

        for k in range(len(self.results)):
            emb = self.keywords[self.results[k]]
            idx = np.where(values_array == emb)[0][0]
            print('topic ', k+1, ': ', keys_array[idx])
