from os import listdir
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
import numpy as np
import cv2


def show_kernels(layer_weights, rows, columns, channels=True):
    """
    Shows the images corresponding to the learned kernels in a convolutional layer
    :param layer_weights: a tensor of layer weights of shape (width, height, channels, nkernels)
    :param rows: the number of rows in the grid shown
    :param columns: the number of columns in the grid shown
    :param channels: whether to interpret the 3rd tensor dimension as color channels,
      if False, the first one is used
    :return: nothing
    """
    nkernels = layer_weights.shape[3]
    fig = plt.figure(figsize=(10, 8))
    for i in range(nkernels):
        fig.add_subplot(rows, columns, i + 1)
        if channels:
            img = layer_weights[:, :, :, i]
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(layer_weights[:, :, 0, i])
    plt.axis('off')
    for axe in fig.axes:
        axe.get_xaxis().set_visible(False)
        axe.get_yaxis().set_visible(False)
    plt.show()


def show_descriptor_dictionary(img_dir, rows=4, columns=9):
    """
    Shows the images corresponding to a descriptor dictionary
    :param img_dir: a directory where all descriptor images are located
    :param rows: the number of rows in the grid shown
    :param columns: the number of columns in the grid shown
    :return: nothing
    """
    fig = plt.figure(figsize=(10, 8))
    for i, file in enumerate(listdir(img_dir)):
        img = mpimg.imread(join(img_dir, file))
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img)
    plt.axis('off')
    for axe in fig.axes:
        axe.get_xaxis().set_visible(False)
        axe.get_yaxis().set_visible(False)
    plt.show()


def dump_descriptor_dictionary(desc_dict, output_dir):
    """
    Saves a descriptor dictionary as JPEG images in a specified output directory
    :param desc_dict: a descriptor dictionary to dump
    :param output_dir: a directory where to dump all descriptor images
    :return: nothing
    """
    for i, image in enumerate(desc_dict):
        misc.toimage(np.reshape(image, (16, 8)), cmin=0, cmax=255)\
            .save(join(output_dir, str(i) + '.jpg'))
