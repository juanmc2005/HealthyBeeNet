import numpy as np
from sklearn.cluster import KMeans

descriptors = np.load('bee_dataset/sift_descriptors.npy')
kmeans = KMeans(n_clusters=4, random_state=1)
kmeans.fit(descriptors)

print(kmeans.cluster_centers_)
