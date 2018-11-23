import bee_data as bees
import clustering
import image

bee_image_dir = 'bee_dataset/bee_imgs'
metadata = 'bee_dataset/bee_data.csv'
train_x, test_x, train_y, test_y = bees.split(bees.read_dataset(metadata))

test_desc = image.compute_descriptors(bee_image_dir, [test_x[0]])

desc_dict = clustering.create_kmeans_dict('bee_dataset/sift_descriptors_train.npy', 10)
test_hist = clustering.nearest_neghbors(desc_dict, test_desc)

print(clustering.histogram(desc_dict, test_hist))
