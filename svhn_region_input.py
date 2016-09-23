import tensorflow as tf

from dataset.load_svhn_region import SVHNRegion

# Global constants
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 160
IMAGE_DEPTH = 3
BATCH_SIZE = 32

def load_svhn_datasets(valid_dataset_needed=False, train_dataset_needed=True):
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
    images, bboxes = data.next_batch()

    feed_dict = {
        images_pl: images,
        bboxes_pl: bboxes
    }

    return feed_dict