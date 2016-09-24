import sys, os, hashlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf

import svhn_region as model
import svhn_region_input as inputs  

BATCH_SIZE = inputs.BATCH_SIZE
IMAGE_WIDTH = inputs.IMAGE_WIDTH
IMAGE_HEIGHT = inputs.IMAGE_HEIGHT
IMAGE_DEPTH = inputs.IMAGE_DEPTH

cp_dir = 'checkpoints/svhn_region/fixed_loss_default'

def main(argv):
    _, _, test_data = inputs.load_svhn_datasets(valid_dataset_needed=False, train_dataset_needed=False)

    with tf.Graph().as_default() as graph:
        images_pl = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
        bboxes_pl = tf.placeholder(tf.float32, [BATCH_SIZE, 4])
        dropout_pl = tf.placeholder(tf.float32)

        conv_pool, num_conv_pool, last_conv_depth = model.conv_graph(images_pl)
        logits = model.fc_graph(conv_pool, num_conv_pool, last_conv_depth, dropout_pl)

        saver = tf.train.Saver()
        sess = tf.Session()
        
        ckpt = tf.train.get_checkpoint_state(cp_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print "Valid checkpoint not found at {}".format(cp_dir)
            return

        for _ in range(2): # Some steps
            feed_dict = inputs.generate_feed_dict(test_data, images_pl, bboxes_pl)
            feed_dict[dropout_pl] = 1.0

            pred = sess.run(logits, feed_dict=feed_dict)

            images = feed_dict[images_pl]

            for i in range(BATCH_SIZE):
                ax = plt.axes()
                plt.imshow(images[i] + 0.5)
                plt.axis('off')
                bbox_left, bbox_top, bbox_width, bbox_height = pred[i, 1], pred[i, 0], pred[i, 3], pred[i, 2]
                ax.add_patch(patches.Rectangle((bbox_left, bbox_top), bbox_width, bbox_height, fill=False))
                plt.show()

if __name__ == '__main__':
    tf.app.run()