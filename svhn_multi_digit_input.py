import tensorflow as tf

from dataset.load_svhn_digits import SVHNDigits

# Global constants
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_DEPTH = 3
BATCH_SIZE = 128

def load_svhn_datasets(valid_dataset_needed=False):
    train_data = SVHNDigits('train')
    if valid_dataset_needed:
        valid_data = SVHNDigits('valid')
    else:
        valid_data = 0
    test_data = SVHNDigits('test')

    return train_data, valid_data, test_data

def generate_feed_dict(data, images_pl, length_labels_pl, digits_labels_pl, masks_pl):
    images, labels = data.next_batch()

    feed_dict = {
        images_pl: images,
        length_labels_pl: labels['length'],
        digits_labels_pl: labels['digits'],
        masks_pl: labels['mask']
    }

    return feed_dict