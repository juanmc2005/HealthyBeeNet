import image as img
import clustering as clt
from sklearn.svm import SVC


def train_svm(img_dir, train_x, train_y, desc_dict):
    """
    Train an SVM to classify beehive health based on an image descriptor histogram
    :param img_dir: the directory to look for the image files
    :param train_x: a list of image files
    :param train_y: a list of booleans indicating if the hive is healthy for a bee image
    :param desc_dict: a list of descriptors representing a dictionary of visual words
    :return: a trained SVM
    """
    descriptors = img.compute_descriptors_by_image(img_dir, train_x)
    histograms = [clt.histogram(desc_dict, clt.nearest_neghbors(desc_dict, ds)) for ds in descriptors]
    svc = SVC()
    svc.fit(histograms, train_y)
    return svc
