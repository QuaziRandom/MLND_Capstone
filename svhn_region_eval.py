"""Runs SVHN region model for manual evaluation.

For the SVHN dataset, the predicted bounding boxes are output
and overlayed with original image. This can be used for manual 
validation, in that the 'worthiness' of the predicted bounding
boxes can be assessed.

NOTE: The eval script is neither efficient nor elegant.
"""
import sys
import os
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf

import svhn_region as model
import svhn_region_input as inputs

# Import constants
BATCH_SIZE = inputs.BATCH_SIZE
IMAGE_WIDTH = inputs.IMAGE_WIDTH
IMAGE_HEIGHT = inputs.IMAGE_HEIGHT
IMAGE_DEPTH = inputs.IMAGE_DEPTH

# Checkpoint directory of trained SVHN region model
cp_dir = 'saved_models/svhn_region/'


def main(argv):
    # Get SVHN region test dataset
    _, _, test_data = inputs.load_svhn_datasets(
        valid_dataset_needed=False, train_dataset_needed=False)

    with tf.Graph().as_default() as graph:
        # Placeholders to provide inputs to the inference graph
        images_pl = tf.placeholder(
            tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
        bboxes_pl = tf.placeholder(tf.float32, [BATCH_SIZE, 4])
        dropout_pl = tf.placeholder(tf.float32)

        # Construct inference graph
        conv_pool, num_conv_pool, last_conv_depth = model.conv_graph(images_pl)
        logits = model.fc_graph(conv_pool, num_conv_pool,
                                last_conv_depth, dropout_pl)

        # Restorer to import trained model
        restorer = tf.train.Saver()

        # Default session
        sess = tf.Session()

        # Restore model from the last saved checkpoint
        # Exit if checkpoint not found
        ckpt = tf.train.get_checkpoint_state(cp_dir)
        if ckpt and ckpt.model_checkpoint_path:
            restorer.restore(sess, ckpt.model_checkpoint_path)
        else:
            print "Valid checkpoint not found at {}".format(cp_dir)
            return

        # Loop on inference graph for some batches of test data
        for _ in range(50):  # 50 batches
            # Populate input placeholders
            feed_dict = inputs.generate_feed_dict(
                test_data, images_pl, bboxes_pl)
            feed_dict[dropout_pl] = 1.0

            # Obtain predicted bounding box
            pred = sess.run(logits, feed_dict=feed_dict)

            # Get the images fed into the graph for the current batch
            images = feed_dict[images_pl]

            # Display results for each image in the batch
            for i in range(BATCH_SIZE):
                ax = plt.axes()
                plt.imshow(images[i] + 0.5)
                plt.axis('off')
                bbox_left, bbox_top, bbox_width, bbox_height = pred[
                    i, 1], pred[i, 0], pred[i, 3], pred[i, 2]
                ax.add_patch(patches.Rectangle(
                    (bbox_left, bbox_top), bbox_width, bbox_height, fill=False))
                plt.show()

if __name__ == '__main__':
    tf.app.run()
