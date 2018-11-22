import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def create_kmeans_dict(descriptors_file, k):
    """
    Creates a dictionary of descriptors using K Means clustering
    :param descriptors_file: the file where the descriptor set is stored
    :param k: the number of clusters to generate
    :return: a descriptor dictionary of shape (k, d) where d is the descriptor size
    """
    descriptors = np.load(descriptors_file)
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
