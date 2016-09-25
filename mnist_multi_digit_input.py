"""Helper to manage dataset inputs for MNIST multi-digit trainer."""

import tensorflow as tf

from dataset.load_mnist import load_mnist as load_mnist_single
from dataset.load_mnist_multi import MNISTMulti

## Constants
# Random state (seeds) for
# the different datasets
TRAIN_SEED = 101010  # Life
VALID_SEED = 666     # Angel
TEST_SEED = 314151   # Pi

# Image parameters
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

# Default batch size
BATCH_SIZE = 128

# Size of each dataset
# Remember MNIST multi-digit is a generated dataset,
# and it can be as *big* as one pleases.
TRAIN_SIZE = 2**16
VALID_SIZE = 2**12
TEST_SIZE = 2**14


def create_multi_digit_datasets():
    """Wraps MNIST multi-digit dataset loader into a simple interface."""

    # Load unnormalized single digit (original MNIST) dataset into memory
    mnist_single = load_mnist_single(normalized=False)

    # Produce separate objects, and so their respective worker threads internally
    # for train, valid and test datasets
    train_data = MNISTMulti(
        mnist_single.train_images, mnist_single.train_labels, TRAIN_SEED, BATCH_SIZE, TRAIN_SIZE)
    valid_data = MNISTMulti(
        mnist_single.valid_images, mnist_single.valid_labels, VALID_SEED, BATCH_SIZE, VALID_SIZE)
    test_data = MNISTMulti(
        mnist_single.test_images, mnist_single.test_labels, TEST_SEED, BATCH_SIZE, TEST_SIZE)

    return train_data, valid_data, test_data


def generate_feed_dict(data, images_pl, length_labels_pl, digits_labels_pl, masks_pl):
    """Helper that deciphers batch data received from MNISTMulti.
    
    Populates batch data into feed dictionary for the trainer.
    """
    images, labels = data.next_batch()

    feed_dict = {
        images_pl: images[:, :, :, None],
        length_labels_pl: labels['length'],
        digits_labels_pl: labels['digits'],
        masks_pl: labels['mask']
    }

    return feed_dict
