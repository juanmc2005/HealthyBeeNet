import pandas as pd
import numpy as np


def read_dataset(filename):
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


print(np.mean(labels(read_dataset('bee_dataset/bee_data.csv'))))
