from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np
import classification as clf


def evaluate(model, img_dir, test_x, test_y, desc_dict):
    """
    Evaluate a classifier using a test bee image dataset and a descriptor dictionary
    :param model: a previously fit model for beehive health classification
    :param img_dir: a directory where image files are located
    :param test_x: a list of bee image files for testing
    :param test_y: a list of booleans indicating if the hive is healthy for a bee image
    :param desc_dict: a list of descriptors representing a dictionary of visual words
    :return: an object with the precision and recall for each class
    """
    histograms = clf.histograms_by_image(img_dir, test_x, desc_dict)
    pred_y = model.predict(histograms)
    return {
        'healthy': {
            'precision': precision_score(test_y, pred_y, pos_label=1),
            'recall': recall_score(test_y, pred_y, pos_label=1)
        },
        'unhealthy': {
            'precision': precision_score(test_y, pred_y, pos_label=0),
            'recall': recall_score(test_y, pred_y, pos_label=0)
        }
    }


def cross_val(model, img_dir, train_x, train_y, desc_dict):
    """
    Evaluate a model using 10-fold cross validation
    :param model: the model to evaluate
    :param img_dir: the directory to look for the image files
    :param train_x: a list of image files
    :param train_y: a list of booleans indicating if the hive is healthy for a bee image
    :param desc_dict: a list of descriptors representing a dictionary of visual words
    :return: the average of cross validation scores for this model
    """
    histograms = clf.histograms_by_image(img_dir, train_x, desc_dict)
    return np.mean(cross_val_score(model, histograms, train_y, cv=10))


def accuracy(model, img_dir, test_x, test_y, desc_dict):
    """
    Return the accuracy score for the current model in the given test dataset
    :param model: a previously fit model for beehive health classification
    :param img_dir: a directory where image files are located
    :param test_x: a list of bee image files for testing
    :param test_y: a list of booleans indicating if the hive is healthy for a bee image
    :param desc_dict: a list of descriptors representing a dictionary of visual words
    :return: the accuracy of the model
    """
    return accuracy_score(test_y, model.predict(clf.histograms_by_image(img_dir, test_x, desc_dict)))


def recall(model, img_dir, test_x, test_y, desc_dict):
    """
    Return the recall score for the current model in the given test dataset
    :param model: a previously fit model for beehive health classification
    :param img_dir: a directory where image files are located
    :param test_x: a list of bee image files for testing
    :param test_y: a list of booleans indicating if the hive is healthy for a bee image
    :param desc_dict: a list of descriptors representing a dictionary of visual words
    :return: the recall of the model
    """
    return recall_score(test_y, model.predict(clf.histograms_by_image(img_dir, test_x, desc_dict)))
