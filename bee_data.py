import pandas as pd
from os import listdir
from sklearn.model_selection import train_test_split


def read_metadataset(filename):
    """
    Reads the bee image dataset csv file
    :param filename: The name of the csv file
    :return: a pandas dataframe with the contents of the file, without the useless columns
    """
    data = pd.read_csv(filename, sep=',', header=0).drop(['pollen_carrying', 'caste'], axis=1)
    data['health'] = data['health'].astype('category')
    return data


def read_dataset(img_dir, metadata_file):
    """
    Read a bee dataset and associate each image to a beehive health label.
    Remove files that are present in the metadata but not in the actual image list
    :param img_dir: a directory where the bee images are located
    :param data: a pandas dataframe with bee data
    :return: a list of pairs (x, y) where x is a bee image file
        and y is a boolean with True indicating that the hive is healthy
        and False otherwise
    """
    data = read_metadataset(metadata_file)
    return [(file, label == 'healthy')
            for file, label in zip(data['file'], data['health'])
            if file in listdir(img_dir)]


def split(dataset):
    """
    Splits the given bee dataset into train and test sets
    :param dataset: a bee dataset obtained from read_dataset
    :return: train_x, test_x, train_y, test_y
    """
    xs = [x for x, _ in dataset]
    ys = [y for _, y in dataset]
    return train_test_split(xs, ys, test_size=0.2, random_state=1)