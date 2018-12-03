import image as img
import clustering as clt
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB


def histograms_by_image(img_dir, xs, desc_dict):
    """
    Return a list of histograms associated to a list of images
    :param img_dir: the directory to look for the image files
    :param xs: a list of image files
    :param desc_dict: a list of descriptors representing a dictionary of visual words
    :return: a list of k size histograms for the descriptor dictionary used
    """
    return [clt.histogram(desc_dict, clt.nearest_neghbors(desc_dict, ds))
            for ds in img.compute_descriptors_by_image(img_dir, xs)]


def train_svm(img_dir, train_x, train_y, desc_dict):
    """
    Train an SVM to classify beehive health based on an image descriptor histogram
    :param img_dir: the directory to look for the image files
    :param train_x: a list of image files
    :param train_y: a list of booleans indicating if the hive is healthy for a bee image
    :param desc_dict: a list of descriptors representing a dictionary of visual words
    :return: a trained SVM
    """
    histograms = histograms_by_image(img_dir, train_x, desc_dict)
    svc = SVC(C=10, random_state=1)
    svc.fit(histograms, train_y)
    return svc


def train_naive_bayes(img_dir, train_x, train_y, desc_dict):
    """
    Train a Naive Bayes model to classify beehive health based on an image descriptor histogram
    :param img_dir: the directory to look for the image files
    :param train_x: a list of image files
    :param train_y: a list of booleans indicating if the hive is healthy for a bee image
    :param desc_dict: a list of descriptors representing a dictionary of visual words
    :return: a trained Naive Bayes model
    """
    histograms = histograms_by_image(img_dir, train_x, desc_dict)
    nb = BernoulliNB()
    nb.fit(histograms, train_y)
    return nb
