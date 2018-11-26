import bee_data as bees
import image as img
import clustering
import classification as clf

# print("Reading dataset")
# bee_image_dir = 'bee_dataset/bee_imgs'
# metadata = 'bee_dataset/bee_data.csv'
# dataset = bees.read_dataset(bee_image_dir, metadata)
# print("Done")

# print("Splitting dataset")
# train_x, test_x, train_y, test_y = bees.split(dataset, 'bee_dataset')
# print("Done")

# TODO read training data
# TODO compute and save training descriptors
# TODO create descriptor dictionary using k means (find best k after we finish this first step)
# TODO train Naive Bayes model and evaluate (later try with gaussian NB if enough time)
# TODO train SVM and evaluate (later try other kernels)

# print("Creating Descriptor Dictionary")
# desc_dict = clustering.create_kmeans_dict('bee_dataset/sift_descriptors_train.npy', 10)
# print("Done")

# print("Training SVM")
# clf.train_svm(bee_image_dir, train_x, train_y, desc_dict)
# print("Done")
