"""Evaluates multi-digit recognition model performance.

The multi-digit recognition model combines the SVHN multi-digit 
model and SVHN region model. Evaluate performance of this combined model,
on the SVHN test dataset. The test data produced by SVHN region is used
to input images to SVHN region model, which localizes the most likely
position of the multi-digits in the image. The recognized region is 
then cropped out after a bit of boundary expansion, and fed to the 
mutlti-digit model. The multi-digit model then predicts the digits
sequence in the cropped image. Evaluate this prediction with SVHN
generated test data, and output positive and negative samples.
"""
import os
import hashlib
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

import svhn_multi_digit as digits_model
import svhn_region as region_model

from dataset.load_svhn_eval import SVHNEval

# 'Clues' that help identify variables to restore
# from the separate digits and region models
region_model_vars_clues = ('hidden', 'bbox')
digits_model_vars_clues = ('hidden', 'readout')
conv_vars_clues = ('conv')

# Checkpoint directory of trained SVHN region model
region_model_dir = 'saved_models/svhn_region'

# Checkpoint directory of trained SVHN multi-digit model
digits_model_dir = 'saved_models/svhn_multi_digit/'

## Constants
# Image parameters for region model
BIG_IMAGE_WIDTH = 160
BIG_IMAGE_HEIGHT = 80

# Parameters for digit model
SMALL_IMAGE_WIDTH = 64
SMALL_IMAGE_HEIGHT = 64
MAX_DIGITS = 5

# Common parameters
IMAGE_DEPTH = 3
BATCH_SIZE = 32

# Bounding box expansion ratio
# The height and width of the
# bounding box predicted by the
# region model is expanded
# by (1 + BBOX_EXPAND_RATIO)
BBOX_EXPAND_RATIO = 0.3


def main(argv):
    # Get SVHN eval dataset
    svhn_eval = SVHNEval(BATCH_SIZE)

    with tf.Graph().as_default() as graph:
        # Placeholder to provide image input to the inference graph
        images_pl = tf.placeholder(
            tf.float32, [BATCH_SIZE, None, None, IMAGE_DEPTH])

        # Constants needed by the inference graph
        masks = tf.constant(1.0, shape=[BATCH_SIZE, MAX_DIGITS])
        dropout = tf.constant(1.0)

        # Construct graph
        # Inference - Convolutional
        conv_pool, num_conv_pool, last_conv_depth = digits_model.conv_graph(
            images_pl)

        # Inference - Region (taking input from convolutional)
        with tf.name_scope('region'):
            bbox_tensor = region_model.fc_graph(
                conv_pool, num_conv_pool, last_conv_depth, dropout)

        # Inference - Length and digits (taking input from convolutional)
        with tf.name_scope('digits'):
            length_tensor, digits_tensor = digits_model.fc_graph(
                conv_pool, num_conv_pool, last_conv_depth, masks, dropout)

            # Convert logits into log softmax outputs for easing confidence
            # eval
            length_log_softmax = tf.nn.log_softmax(length_tensor)
            digits_log_softmax = []
            for d in digits_tensor:
                digits_log_softmax.append(tf.nn.log_softmax(d))

        # Setup the variables to be properly restored from the
        # respectived trained models.
        # Get all variables constructed in the default graph (here)
        all_graph_vars = tf.all_variables()

        # Populate dictionaries to be fed into tf.train.Saver()
        # If the name scope in the current graph is 'region',
        # then restore those variables from the region model
        # If the name scope in the current graph is 'digits',
        # then restore those variables from the digits model
        # If there is no top name scope, but the variables
        # start with 'conv', restore those from the
        # digits model. The 'conv' variable can be restored from
        # regions model too, as they're in reality
        # themselves derived from the digits model.
        digits_model_vars = {}
        region_model_vars = {}
        conv_model_vars = {}
        for var in all_graph_vars:
            if var.name.replace('region/', '').startswith(region_model_vars_clues):
                region_model_vars[var.op.name.replace('region/', '')] = var
            elif var.name.replace('digits/', '').startswith(digits_model_vars_clues):
                digits_model_vars[var.op.name.replace('digits/', '')] = var
            elif var.name.startswith(conv_vars_clues):
                digits_model_vars[var.op.name] = var

        # Restorers to import the respective models
        region_model_restorer = tf.train.Saver(region_model_vars)
        digits_model_restorer = tf.train.Saver(digits_model_vars)

        # Default session
        sess = tf.Session()

        # Restore digits model from the respective specified directory
        # If model not found, exit
        digits_ckpt = tf.train.get_checkpoint_state(digits_model_dir)
        if digits_ckpt and digits_ckpt.model_checkpoint_path:
            digits_model_restorer.restore(
                sess, digits_ckpt.model_checkpoint_path)
            print "Digits model restored from {}".format(digits_ckpt.model_checkpoint_path)
        else:
            print "Did not find digits trained model. Exiting."
            return

        # Restore region model from the respective specified directory
        # If model not found, exit
        regions_ckpt = tf.train.get_checkpoint_state(region_model_dir)
        if regions_ckpt and regions_ckpt.model_checkpoint_path:
            region_model_restorer.restore(
                sess, regions_ckpt.model_checkpoint_path)
            print "Region model restored from {}".format(regions_ckpt.model_checkpoint_path)
        else:
            print "Did not find regions trained model. Exiting."
            return

        # Run a debug op that makes sure all variables in the graph
        # are successfully imported and initialized
        sess.run(tf.assert_variables_initialized())

        # For all batches in the SVHN eval dataset (in turn SVHN test dataset)
        for _ in range(svhn_eval.get_dataset_size() // BATCH_SIZE):
            # Get the next batch
            images, bboxes, labels = svhn_eval.next_batch()
            length_labels = labels['length']
            digits_labels = labels['digits']

            # Fill the feed dictionary with the input images
            region_feed_dict = {
                images_pl: images
            }

            # Predict output of the region model, given the images
            bbox_pred = sess.run(bbox_tensor, feed_dict=region_feed_dict)

            # Process the images to be made compatible for digits model
            repackaged_images = np.ndarray(
                [BATCH_SIZE, SMALL_IMAGE_HEIGHT, SMALL_IMAGE_WIDTH, IMAGE_DEPTH])
            for i in range(BATCH_SIZE):
                # Get predicted bounding box for each image in the batch
                bbox_left, bbox_top, bbox_width, bbox_height = bbox_pred[
                    i, 1], bbox_pred[i, 0], bbox_pred[i, 3], bbox_pred[i, 2]

                # Find the crop co-ordinates after applying bounding box
                # expansion
                current_image = images[i]
                crop_top = max(
                    0, int(bbox_top - bbox_height * BBOX_EXPAND_RATIO / 2))
                crop_bottom = min(BIG_IMAGE_HEIGHT, int(
                    (bbox_top + bbox_height) + bbox_height * BBOX_EXPAND_RATIO / 2))
                crop_left = max(
                    0, int(bbox_left - bbox_width * BBOX_EXPAND_RATIO / 2))
                crop_right = min(BIG_IMAGE_WIDTH, int(
                    (bbox_left + bbox_width) + bbox_width * BBOX_EXPAND_RATIO / 2))

                # Crop from the input (regions) image
                cropped_image = current_image[
                    crop_top:crop_bottom, crop_left:crop_right]

                # Resize the cropped image into fixed square format
                repackaged_images[i] = cv2.resize(
                    cropped_image, (SMALL_IMAGE_WIDTH, SMALL_IMAGE_HEIGHT))

            # Create feed dictionary of the cropped and resized images from the
            # batch
            digits_feed_dict = {
                images_pl: repackaged_images
            }

            # Predict length and digits (log softmax-ed) found in the
            # crop-resized images
            length_pred, digits_pred = sess.run(
                [length_log_softmax, digits_log_softmax], feed_dict=digits_feed_dict)

            # Workout predicted length from the
            # length logits (log softmax-ed)
            length = np.argmax(length_pred, axis=1)

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
            total_confidence = length_pred + \
                np.cumsum(
                    np.hstack([np.zeros([BATCH_SIZE, 1]), digits_max_log.T]), axis=1)
            total_confidence_antilog = np.exp(total_confidence)

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

                # Get the predicted bounding box to overlay upon the image tp
                # display
                bbox_left, bbox_top, bbox_width, bbox_height = bbox_pred[
                    i, 1], bbox_pred[i, 0], bbox_pred[i, 3], bbox_pred[i, 2]

                # Construct the image to be displayed/saved with all
                # information
                ax = plt.axes()
                current_image = images[i]
                plt.imshow(current_image + 0.5)
                ax.add_patch(patches.Rectangle(
                    (bbox_left, bbox_top), bbox_width, bbox_height, fill=False))
                plt.axis('off')
                plt.title(title)

                # If actual, and predicted labels don't match,
                # the string labels won't match too. Separate
                # the negative samples and positive samples into
                # different directories
                if A != P:
                    fig_savepath = 'eval/multi_digit_recognition/negative_samples/'
                else:
                    fig_savepath = 'eval/multi_digit_recognition/positive_samples/'

                # Save each image
                plt.savefig(fig_savepath +
                            hashlib.sha1(current_image).hexdigest() + '.png')
                plt.close()


if __name__ == '__main__':
    tf.app.run()
