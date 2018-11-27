from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def create_kmeans_dict(descriptors, k):
    """
    Creates a dictionary of descriptors using K Means clustering
    :param descriptors: a list of SIFT descriptors
    :param k: the number of clusters to generate
    :return: a descriptor dictionary of shape (k, d) where d is the descriptor size
    """
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(descriptors)
    return kmeans.cluster_centers_


def nearest_neghbors(descriptor_dict, descs):
    """
    Computes the nearest neighbors of a list of descriptors in a descriptor dictionary
    :param descriptor_dict: a descriptor dictionary like in create_kmeans_dict
    :param descs: a list of descriptors of an image
    :return: the indices of the nearest neighbors of descs in descriptor_dict
    """
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(descriptor_dict)
    return nn.kneighbors(descs, return_distance=False).reshape((len(descs),))


def histogram(descriptor_dict, visual_words):
    """
    Calculates a frequency histogram of a list of visual words for
    a given descriptor dictionary
    :param descriptor_dict: a descriptor dictionary like in create_kmeans_dict
    :param visual_words: a list of indices corresponding to descriptors in descriptor_dict
    :return: a list of visual word frequencies of length len(descriptor_dict)
    """
    nw = len(visual_words)
    freqs = [0 for _ in range(len(descriptor_dict))]
    for w in visual_words:
        freqs[w] += 1
    return [x / nw for x in freqs]
