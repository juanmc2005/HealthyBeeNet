import bee_data as bees
import image as img
import clustering
import classification as clf

bee_image_dir = 'bee_dataset/bee_imgs'
metadata = 'bee_dataset/bee_data.csv'
dataset = bees.read_dataset(bee_image_dir, metadata)

train_x, test_x, train_y, test_y = bees.split(dataset)

print("Creating Descriptor Dictionary")
desc_dict = clustering.create_kmeans_dict('bee_dataset/sift_descriptors_train.npy', 10)
print("Done")

print("Training SVM")
clf.train_svm(bee_image_dir, train_x, train_y, desc_dict)
print("Done")