import bee_data as bees
import image as imgs
import clustering as clst
import evaluation as evl
import optimisation as opt
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB

dataset_dir = 'bee_dataset/'
bee_image_dir = dataset_dir + 'bee_imgs'
train_descriptors_file = dataset_dir + 'sift_descriptors_train.npy'

# Read training data
train_x = bees.read_x_split(dataset_dir + 'train_x')
train_y = bees.read_y_split(dataset_dir + 'train_y')

# Load training descriptors
descriptors = imgs.load_descriptors(train_descriptors_file)

# Find best K for Bernoulli Naive Bayes
opt.find_best_k(BernoulliNB(), bee_image_dir, train_x, train_y, descriptors)

# Find best K for SVM
k = opt.find_best_k(SVC(C=10, gamma='scale', random_state=1), bee_image_dir, train_x, train_y, descriptors)

# Use best SVM K to print its 10-fold cross validation score
desc_dict = clst.create_kmeans_dict(descriptors, k=k)
print(evl.cross_val(SVC(C=10, gamma='scale', random_state=1), bee_image_dir, train_x, train_y, desc_dict))
