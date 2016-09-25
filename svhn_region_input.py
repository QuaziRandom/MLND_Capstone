"""Helper to manage dataset inputs for SVHN region trainer."""

import tensorflow as tf

from dataset.load_svhn_region import SVHNRegion

## Constants
# Image parameters
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 80
IMAGE_DEPTH = 3

# Default batch size
BATCH_SIZE = 32

def load_svhn_datasets(valid_dataset_needed=False, train_dataset_needed=True):
    """Wraps SVHN region dataset loader into a simple interface.
    
    Optionally specify to load (or not) valid and train datasets.
    """
    if train_dataset_needed:
        train_data = SVHNRegion('train', batch_size=BATCH_SIZE, buffer_size=8)
    else:
        train_data = 0
    if valid_dataset_needed:
        valid_data = SVHNRegion('valid', batch_size=BATCH_SIZE, buffer_size=8)
    else:
        valid_data = 0
    test_data = SVHNRegion('test', batch_size=BATCH_SIZE, buffer_size=8)

    return train_data, valid_data, test_data

def generate_feed_dict(data, images_pl, bboxes_pl):
    """Helper fills batch data received from SVHNRegion to feed dictionary."""
    images, bboxes = data.next_batch()

    feed_dict = {
        images_pl: images,
        bboxes_pl: bboxes
    }

    return feed_dict