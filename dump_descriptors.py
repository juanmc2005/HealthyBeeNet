import bee_data as bees
import image as img
import clustering as clst
import numpy as np
from scipy import misc

dataset_dir = 'bee_dataset/'
bee_image_dir = dataset_dir + 'bee_imgs'
train_descriptors_file = dataset_dir + 'sift_descriptors_train.npy'

# Read training data
train_x = bees.read_x_split(dataset_dir + 'train_x')
train_y = bees.read_y_split(dataset_dir + 'train_y')

# Load training descriptors
descriptors = img.load_descriptors(train_descriptors_file)

desc_dict = clst.create_kmeans_dict(descriptors, k=36)

print(np.shape(desc_dict))

for i, image in enumerate(desc_dict):
    misc.toimage(np.reshape(image, (16, 8)), cmin=0, cmax=255).save(dataset_dir + 'descriptor_dict/' + str(i) + '.jpg')

