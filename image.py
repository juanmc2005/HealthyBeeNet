import cv2
import numpy as np
from os.path import join


def compute_descriptors_by_image(img_dir, images):
    """
    Transform a list of bee images into their corresponding SIFT descriptors
    :param img_dir: the directory where the image files are located
    :param images: a list of bee image file names to apply SIFT
    :return: a list of descriptor matrices
    """
    return [sift(join(img_dir, file)) for file in images]


def compute_descriptors(img_dir, images, out_file=None):
    """
    Transform a bee image dataset into a matrix representing all key
    point descriptors detected by SIFT.
    :param img_dir: the directory where the image files are located
    :param images: a list of bee image file names to apply SIFT
    :param out_file: an optional file path to store the descriptors in binary format
    :return: a matrix of shape (k, 128), where k is the number of
        key points extracted by SIFT
    """
    result = sift(join(img_dir, images[0]))
    for file in images[1:]:
        descriptors = sift(join(img_dir, file))
        if descriptors is not None:
            result = np.append(result, descriptors, axis=0)
        else:
            print("No descriptors found for " + file)
    if out_file is not None:
        np.save(out_file, result)
    return result


def sift(image):
    """
    Applies the SIFT algorithm to an image
    :param image: the path to a bee image
    :return: a matrix of shape (k, 128), where k is the number
        of key points detected by SIFT, with the descriptors for every key point
    """
    image = cv2.imread(image)
    color = cv2.cvtColor(image, cv2.IMREAD_COLOR)
    op = cv2.xfeatures2d.SIFT_create()
    _, descriptors = op.detectAndCompute(color, None)
    return descriptors


def draw_key_points(image, out_image):
    """
    Applies the SIFT algorithm to the input image and draws them in the output image
    :param img: the source image path to look for key points
    :param out_image: the output image path to draw the key points
    :return: nothing
    """
    img = cv2.imread(image)
    color = cv2.cvtColor(img, cv2.IMREAD_COLOR)
    op = cv2.xfeatures2d.SIFT_create()
    kps, descriptors = op.detectAndCompute(color, None)
    cv2.drawKeypoints(color, kps, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(out_image, img)
