import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

class TopicExtractor(object):
    def __init__(self, keywords):
        self.keywords = keywords

class TopicExtractorCount(TopicExtractor):
    def __init__(self, keywords,n):
        super(self.__class__, self).__init__(keywords)
        self.n = n
    def extractTopics(self):

class TopicExtractorKMeans(TopicExtractor):
    def __init__(self, keywords,k):
        super(self.__class__, self).__init__(keywords)
        self.k = k
        self.kmeans = KMeans(n_clusters=k)
        self.nbrs = NearestNeighbors(n_neighbors=1).fit(keywords)
    def extractTopics(self):
        self.kmeans.fit(self.keywords)
        centroids = np.array(self.kmeans.cluster_centers_).tolist()
        self.results = [self.nbrs.kneighbors(X=c, n_neighbors=1, return_distance=False) for c in centroids]
