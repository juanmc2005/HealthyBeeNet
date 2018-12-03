import utils
import clustering as clst
import evaluation as evl
from operator import itemgetter


def find_best_k(model, img_dir, train_x, train_y, descriptors):
    """
    Find the best K value (for K Means) for a specific model using
    10-fold cross validation with K values in range [10, 600].
    Print intermediate results so there's feedback for the caller
    :param model: an sklearn model to evaluate
    :param img_dir: a directory where all the bee images are located
    :param train_x: a list of image file names corresponding to the training set
    :param train_y: a list of image labels, where True = healthy and False = unhealthy
    :param descriptors: a list of SIFT descriptors extracted from the training set
    :return: the best k for the model
    """
    ks = utils.sampled_range(10, 600, 20)
    print("Will try K = " + str(ks))
    results = []
    for k in ks:
        print("Trying K = " + str(k))
        desc_dict = clst.create_kmeans_dict(descriptors, k=k)
        accuracy = evl.cross_val(model, img_dir, train_x, train_y, desc_dict)
        print("Mean Accuracy: " + str(accuracy))
        results.append((k, accuracy))
    # First pair element of first pair in sorted result list by the errors
    print(sorted(results, key=itemgetter(1)))
    return sorted(results, key=itemgetter(1))[-1][0]