import pandas as pd
from os import listdir
from os.path import join
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


def read_x_split(filename):
    """
    Read previously written bee image data split into a list
    :param filename: the file with the image file names
    :return: a list of bee image file names
    """
    with open(filename, 'r') as file:
        return [line.strip() for line in file.readlines()]


def read_y_split(filename):
    """
    Read previously written bee image labels split into a list
    :param filename: the file with the image labels
    :return: a list of booleans indicating if the hive is healthy for a bee in an image
    """
    return [label == 'healthy' for label in read_x_split(filename)]


def read_y_split_binary(filename):
    """
    Read previously written bee image labels split into
    a list of one hot encoded vectors
    :param filename: the file with the image labels
    :return: a list of labels encoded as one hot vectors,
      where [1, 0] = healthy, and [0, 1] = unhealthy
    """
    return [1 if label == 'healthy' else 0 for label in read_x_split(filename)]


def write_dataset(dataset, filename):
    """
    Writes a specific dataset (list of bee image files) to a txt file
    in the specified file
    :param dataset: the bee dataset to store
    :param filename: the file where the dataset will be written
    :return: nothing
    """
    with open(filename, 'w') as file:
        for image in dataset:
            file.write(image + '\n')


def split(dataset, directory):
    """
    Splits the given bee dataset into train and test sets and stores them
    in the specified directory
    :param dataset: a bee dataset obtained from read_dataset
    :param directory: a directory
    :return: train_x, test_x, train_y, test_y
    """
    xs = [x for x, _ in dataset]
    ys = [y for _, y in dataset]
    train_x, test_x, train_y, test_y = train_test_split(xs, ys, test_size=0.2,
                                                        stratify=ys, random_state=1)
    write_dataset(train_x, join(directory, 'train_x'))
    write_dataset(test_x, join(directory, 'test_x'))
    write_dataset(['healthy' if y else 'unhealthy' for y in train_y], join(directory, 'train_y'))
    write_dataset(['healthy' if y else 'unhealthy' for y in test_y], join(directory, 'test_y'))
    return train_x, test_x, train_y, test_y
