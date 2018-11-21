import numpy as np
from os.path import join
import image
import bee_data as bees


def compute_descriptors(images_dir, images, out_file=None):
    """
    Transform a bee image dataset into a matrix representing all key
    point descriptors detected by SIFT.
    :param images_dir: the directory where the image files are located
    :param images: a list of bee image file names to apply SIFT
    :param out_file: an optional file path to store the descriptors in binary format
    :return: a matrix of shape (k, 128), where k is the number of
        key points extracted by SIFT
    """
    result = image.sift(join(images_dir, images[0]))
    for file in images[1:]:
        descriptors = image.sift(join(images_dir, file))
        if descriptors is not None:
            result = np.append(result, descriptors, axis=0)
        else:
            print("No descriptors found for " + file)
    if out_file is not None:
        np.save(out_file, result)
    return result


bee_image_dir = 'bee_dataset/bee_imgs'
metadata = 'bee_dataset/bee_data.csv'
train_x, test_x, train_y, test_y = bees.split(bees.read_dataset(metadata))

