"""Load MNIST dataset into memory.

From the MNIST train and test dataset, create three splits, viz.,
training, validation and test datasets. Some user-provided portion
of the train dataset is used for validation dataset. To extract
binary information from MNIST database, the description as
provided in http://yann.lecun.com/exdb/mnist/ is used:

    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):

    [offset] [type]          [value]          [description] 
    0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
    0004     32 bit integer  60000            number of items 
    0008     unsigned byte   ??               label 
    0009     unsigned byte   ??               label 
    ........ 
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):

    [offset] [type]          [value]          [description] 
    0000     32 bit integer  0x00000803(2051) magic number 
    0004     32 bit integer  60000            number of images 
    0008     32 bit integer  28               number of rows 
    0012     32 bit integer  28               number of columns 
    0016     unsigned byte   ??               pixel 
    0017     unsigned byte   ??               pixel 
    ........ 
    xxxx     unsigned byte   ??               pixel
"""

import sys
import os
import struct
import array
import numpy as np
import matplotlib.pyplot as plt
import pickle


def load_mnist(num_valid=5000, normalized=True):
    """Load train, valid and test MNIST datasets.
    
    Given that MNIST dataset binary files accompany the script
    in ./mnist/ folder, load the respective splits in either
    `normalized` [-0.5, 0.5] or unnormalized [0, 255] form.  
    """
    # Load train and test sets using helper functions
    # Make sure the paths stay intact when called from a parent directory
    train_images = load_mnist_images(os.path.join(
        os.path.dirname(__file__), "mnist/train-images-idx3-ubyte"))
    train_labels = load_mnist_labels(os.path.join(
        os.path.dirname(__file__), "mnist/train-labels-idx1-ubyte"))
    test_images = load_mnist_images(os.path.join(
        os.path.dirname(__file__), "mnist/t10k-images-idx3-ubyte"))
    test_labels = load_mnist_labels(os.path.join(
        os.path.dirname(__file__), "mnist/t10k-labels-idx1-ubyte"))

    # Split train into train and valid sets
    valid_images = train_images[-num_valid:]
    valid_labels = train_labels[-num_valid:]
    train_images = train_images[:-num_valid]
    train_labels = train_labels[:-num_valid]

    # Normalize all images to a standard [-0.5, 0.5] interval if needed
    if normalized:
        train_images = (train_images - np.mean(train_images)) / 255.0
        valid_images = (valid_images - np.mean(train_images)) / 255.0
        test_images = (test_images - np.mean(train_images)) / 255.0

    # Bundle return images and label sets into one object
    class MNISTDataSet(object):
        pass
    dataset = MNISTDataSet()

    dataset.train_images, dataset.train_labels = train_images, train_labels
    dataset.valid_images, dataset.valid_labels = valid_images, valid_labels
    dataset.test_images, dataset.test_labels = test_images, test_labels

    return dataset


def load_mnist_labels(path_to_mnist_labels):
    """Given a train/test MNIST dataset, return its labels."""
    assert os.path.exists(path_to_mnist_labels), "Invalid path {}".format(
        path_to_mnist_labels)
    with open(path_to_mnist_labels, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        assert magic == 2049, "Magic number not matching"
        label_data = array.array("B", f.read())

    labels = np.array(label_data)
    return labels


def load_mnist_images(path_to_mnist_images):
    """Given a train/test MNIST dataset, return its images."""
    assert os.path.exists(path_to_mnist_images), "Invalid path {}".format(
        path_to_mnist_images)
    with open(path_to_mnist_images, "rb") as f:
        magic, num_images, num_rows, num_columns = struct.unpack(
            ">IIII", f.read(16))
        assert magic == 2051, "Magic number not matching"
        image_data = array.array("B", f.read())

    images = np.ndarray([num_images, num_rows, num_columns])
    for i in xrange(num_images):
        images[i] = np.array(
            image_data[i * num_rows * num_columns:(i + 1) * num_rows * num_columns]).reshape(num_rows, num_columns)

    return images


def main():
    # Test if datasets were loaded properly.
    # Invoking script from command-line triggers the test.

    images = load_mnist_images(
        os.path.abspath("mnist/train-images-idx3-ubyte"))
    labels = load_mnist_labels(
        os.path.abspath("mnist/train-labels-idx1-ubyte"))
    for i in xrange(25):
        plt.subplot(5, 5, i + 1)
        index = np.random.randint(0, images.shape[0])
        plt.imshow(images[index])
        plt.title(labels[index])
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    sys.exit(int(main() or 0))
