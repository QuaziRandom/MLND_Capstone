"""Evaluates SVHN multi-digit model performance.

Follows similar strategy as that for MNIST multi-digit model.
For the test dataset, evaluate accuracy and write some 
positive and negative samples into disk. Eval-ing strictly works only
on the inference graph of the SVHN multi-digit model.

NOTE: The eval script is neither efficient nor elegant. Most of efficient eval-ing 
is already done during training, and so does not necessitate elegance here.
"""
import sys
import os
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import svhn_multi_digit as model
import svhn_multi_digit_input as inputs

# Import constants
BATCH_SIZE = inputs.BATCH_SIZE
IMAGE_WIDTH = inputs.IMAGE_WIDTH
IMAGE_HEIGHT = inputs.IMAGE_HEIGHT
IMAGE_DEPTH = inputs.IMAGE_DEPTH
MAX_DIGITS = model.MAX_DIGITS

# Checkpoint directory of trained SVHN multi-digit model
cp_dir = 'checkpoints/svhn_multi_digit/'


def main(argv):
    # Get SVHN multi-digit test dataset
    _, _, test_data = inputs.load_svhn_datasets(
        valid_dataset_needed=False, train_dataset_needed=False)

    with tf.Graph().as_default() as graph:
        # Placeholders to provide inputs to the inference graph
        images_pl = tf.placeholder(
            tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
        length_labels_pl = tf.placeholder(tf.int32, [BATCH_SIZE])
        digits_labels_pl = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_DIGITS])
        masks_pl = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_DIGITS])
        dropout_pl = tf.placeholder(tf.float32)

        # Construct inference graph
        conv_pool, num_conv_pool, last_conv_depth = model.conv_graph(images_pl)
        length_logits, digits_logits = model.fc_graph(
            conv_pool, num_conv_pool, last_conv_depth, masks_pl, dropout_pl)

        # Convert logits into log softmax outputs for easing confidence eval
        length_log_softmax = tf.nn.log_softmax(length_logits)
        digits_log_softmax = []
        for d in digits_logits:
            digits_log_softmax.append(tf.nn.log_softmax(d))

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
                test_data, images_pl, length_labels_pl, digits_labels_pl, masks_pl)
            feed_dict[dropout_pl] = 1.0

            # Obtain predicted length and digits (log softmax-ed)
            # by running graph with batch inputs
            length_log, digits_pred = sess.run(
                [length_log_softmax, digits_log_softmax], feed_dict=feed_dict)

            # Workout predicted length from the
            # length logits (log softmax-ed)
            length = np.argmax(length_log, axis=1)

            # Workout predicted digits from the
            # digits logits (log softmax-ed)
            # Additionally, obtain max log probabilities
            # for each digit
            digits = []
            digits_max_log = []
            for d in digits_pred:
                digits.append(np.argmax(d, axis=1))
                digits_max_log.append(np.max(d, axis=1))
            digits = np.array(digits)
            digits_max_log = np.array(digits_max_log)

            # Calculate total confidence for each sequence length by
            # adding up log probability of different predicted lengths for a
            # given image and max log probabilites of each predicted digit for
            # that image. Finally, exponentiate the added joint probabilities.
            total_confidence = length_log + \
                np.cumsum(
                    np.hstack([np.zeros([BATCH_SIZE, 1]), digits_max_log.T]), axis=1)
            total_confidence_antilog = np.exp(total_confidence)

            # Get the inputs fed into the graph for the current batch
            images = feed_dict[images_pl]
            length_labels = feed_dict[length_labels_pl]
            digits_labels = feed_dict[digits_labels_pl]

            # Find results for each image in the batch
            for i in range(BATCH_SIZE):
                # Populate the label information (actual)
                # namely, length and digits into a string
                # for an image in the batch
                if length_labels[i] == 0:
                    A = str(length_labels[i]) + ' length'
                elif length_labels[i] > MAX_DIGITS:
                    A = '>' + str(MAX_DIGITS) + ' length'
                else:
                    A = str(length_labels[
                            i]) + '|' + ''.join(digits_labels[i, :length_labels[i]].astype(np.character).tolist())

                # Populate the predicted labels
                # namely, length and digits into a string
                # for an image in the batch
                if length[i] == 0:
                    P = str(length[i]) + ' length'
                elif length[i] > MAX_DIGITS:
                    P = '>' + str(MAX_DIGITS) + ' length'
                else:
                    P = str(
                        length[i]) + '|' + ''.join(digits[:length[i], i].astype(np.character).tolist())

                # Find the relative probability of predicted length (and digits)
                # and convert that into a string
                C = str(np.round(total_confidence_antilog[
                        i, length[i]] / np.sum(total_confidence_antilog[i]) * 100.0, 2))

                # Append actual, predicted labels and confidence
                # into 'title' for the image
                title = 'A:' + A + ' ' + 'P:' + P + ' ' + 'C:' + C + '%'

                # If actual, and predicted labels don't match,
                # the string labels won't match too. Separate
                # the negative samples and positive samples into
                # different directories
                if A != P:
                    fig_savepath = 'eval/svhn_multi_digit/negative_samples/'
                    plt.imshow(images[i] + 0.5)
                    plt.axis('off')
                    plt.title(title)
                    # Save each image
                    plt.savefig(
                        fig_savepath + hashlib.sha1(images[i].copy(order='C')).hexdigest() + '.png')
                else:
                    fig_savepath = 'eval/svhn_multi_digit/positive_samples/'
                    # Not writing positive samples now. One run already over
                    # with many samples.


if __name__ == '__main__':
    tf.app.run()
