import utils
import clustering as clst
import evaluation as evl
from operator import itemgetter


def find_best_k(model, img_dir, train_x, train_y, descriptors):
    ks = utils.sampled_range(30, 40, 4)
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