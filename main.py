import bee_data as bees
import image as imgs
import clustering as clst
import classification as clf
from sklearn.model_selection import StratifiedKFold
import numpy as np
from operator import itemgetter
import math


def sampled_range(mini, maxi, num):
    if not num:
        return []
    lmini = math.log(mini)
    lmaxi = math.log(maxi)
    ldelta = (lmaxi - lmini) / (num - 1)
    out = [x for x in set([int(math.exp(lmini + i * ldelta)) for i in range(num)])]
    out.sort()
    return out


def find_best_k(img_dir, train_x, train_y):
    n_splits = 10
    ks = sampled_range(50, 250, 15)
    skf = StratifiedKFold(n_splits=n_splits, random_state=1)
    results = []
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    for k in ks:
        print("Trying K = " + str(k))
        accuracy = 0
        for train_index, test_index in skf.split(train_x, train_y):
            k_train_x, k_test_x = train_x[train_index], train_x[test_index]
            k_train_y, k_test_y = train_y[train_index], train_y[test_index]
            descriptors = imgs.compute_descriptors(img_dir, k_train_x)
            desc_dict = clst.create_kmeans_dict(descriptors, k)
            bayes = clf.train_naive_bayes(bee_image_dir, k_train_x, k_train_y, desc_dict)
            accuracy += clf.accuracy(bayes, img_dir, k_test_x, k_test_y, desc_dict)
        print("Mean Accuracy: ")
        print(accuracy / n_splits)
        results.append((k, accuracy / n_splits))
    # First pair element of first pair in sorted result list by the errors
    print(sorted(results, key=itemgetter(1)))
    return sorted(results, key=itemgetter(1))[-1][0]


dataset_dir = 'bee_dataset/'
bee_image_dir = dataset_dir + 'bee_imgs'
train_descriptors_file = dataset_dir + 'sift_descriptors_train.npy'

# print("Reading dataset")
# metadata = 'bee_dataset/bee_data.csv'
# dataset = bees.read_dataset(bee_image_dir, metadata)
# print("Done")

# print("Splitting dataset")
# train_x, test_x, train_y, test_y = bees.split(dataset, 'bee_dataset')
# print("Done")

# Read training data
train_x = bees.read_x_split(dataset_dir + 'train_x')
train_y = bees.read_y_split(dataset_dir + 'train_y')
# train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, stratify=train_y, random_state=1)
# test_x = bees.read_x_split(dataset_dir + 'test_x')
# test_y = bees.read_y_split(dataset_dir + 'test_y')

# Compute and save training descriptors
# img.compute_descriptors(bee_image_dir, train_x, out_file=train_descriptors_file)

# Create descriptor dictionary using k means. TODO find best k after we finish this first step
print(sampled_range(50, 250, 15))
k = find_best_k(bee_image_dir, train_x, train_y)
print('K = ' + str(k))

# Train Naive Bayes model and evaluate. TODO maybe try with gaussian NB if enough time
# bayes = clf.train_naive_bayes(bee_image_dir, train_x, train_y, desc_dict)
# print(clf.evaluate(bayes, bee_image_dir, val_x, val_y, desc_dict))

# TODO train SVM and evaluate (later try other kernels)

# print("Training SVM")
# clf.train_svm(bee_image_dir, train_x, train_y, desc_dict)
# print("Done")
