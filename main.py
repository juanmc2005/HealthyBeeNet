import bee_data as bees
import image as img
import clustering as clst
import classification as clf
from sklearn.model_selection import train_test_split

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
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, stratify=train_y, random_state=1)
# test_x = bees.read_x_split(dataset_dir + 'test_x')
# test_y = bees.read_y_split(dataset_dir + 'test_y')

# Compute and save training descriptors
# img.compute_descriptors(bee_image_dir, train_x, out_file=train_descriptors_file)

# Create descriptor dictionary using k means. TODO find best k after we finish this first step
desc_dict = clst.create_kmeans_dict(train_descriptors_file, k=5)

# Train Naive Bayes model and evaluate. TODO maybe try with gaussian NB if enough time
bayes = clf.train_naive_bayes(bee_image_dir, train_x, train_y, desc_dict)
print(clf.evaluate(bayes, bee_image_dir, val_x, val_y, desc_dict))

# TODO train SVM and evaluate (later try other kernels)

# print("Training SVM")
# clf.train_svm(bee_image_dir, train_x, train_y, desc_dict)
# print("Done")
