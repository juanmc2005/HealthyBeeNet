import pandas as pd
import numpy as np
from os import listdir
from os.path import join
import scipy.io as sio
import image


def read_metadataset(filename):
    """
    Reads the bee image dataset csv file
    :param filename: The name of the csv file
    :return: a pandas dataframe with the contents of the file, without the useless columns
    """
    data = pd.read_csv(filename, sep=',', header=0).drop(['pollen_carrying', 'caste'], axis=1)
    data['health'] = data['health'].astype('category')
    return data


def labels(data):
    """
    Extract the health labels from a bee dataset
    :param data: a pandas dataframe with bee data
    :return: a list of booleans indicating if the hive is healthy
    """
    return [value == 'healthy' for value in data['health']]


def compute_descriptors(dir, out_file=None):
    """
    Transform a bee image dataset into a matrix representing all key
    point descriptors detected by SIFT.
    :param dir: the source directory where all the images are located
    :param out_file: an optional file path to store the descriptors in binary format
    :return: a matrix of shape (k, 128), where k is the number of
        key points extracted by SIFT
    """
    files = listdir(dir)
    result = image.sift(join(dir, files[0]))
    for file in files[1:]:
        descriptors = image.sift(join(dir, file))
        if descriptors is not None:
            result = np.append(result, descriptors, axis=0)
        else:
            print("No descriptors found for " + file)
    if out_file is not None:
        np.save(out_file, result)
    return result


# print(np.mean(labels(read_metadataset('bee_dataset/bee_data.csv'))))
print(np.load('bee_dataset/sift_descriptors.npy').shape)
