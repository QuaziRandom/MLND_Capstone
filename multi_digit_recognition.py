"""Inference model to perform multi-digit recognition.

The multi-digit recognition model combines the SVHN multi-digit 
model and SVHN region model. Given a directory with images that have
digits to be recognized, these input images are fed to the SVHN region
model, which localizes the most likely position of the multi-digits in 
the image. The recognized region is then cropped out after a bit of 
boundary expansion, and fed to the mutlti-digit model. The multi-digit 
model then predicts the digits sequence in the cropped image. Finally,
all the information is combined and displayed to the user.
"""
import os
import argparse
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

import svhn_multi_digit as digits_model
import svhn_region as region_model

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

# Bounding box expansion ratio
# The height and width of the
# bounding box predicted by the
# region model is expanded
# by (1 + BBOX_EXPAND_RATIO)
BBOX_EXPAND_RATIO = 0.4


def main(argv):
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='directory where input images reside')
    args = parser.parse_args()

    # Check if directory exists
    if not os.path.exists(args.dir):
        print "Invalid path", args.dir
        return
    elif not os.path.isdir(args.dir):
        print "Not a directory", args.dir
        return

    # Extract all files from the directory
    files = os.listdir(args.dir)

    with tf.Graph().as_default() as graph:
        # Placeholder to provide image input to the inference graph
        image_pl = tf.placeholder(tf.float32, [1, None, None, IMAGE_DEPTH])

        # Constants needed by the inference graph
        masks = tf.constant(1.0, shape=[1, MAX_DIGITS])
        dropout = tf.constant(1.0)

        # Construct graph
        # Inference - Convolutional
        conv_pool, num_conv_pool, last_conv_depth = digits_model.conv_graph(
            image_pl)

        # When combined model is run, the convolution graph
        # is used twice for a single image. Once, for the
        # localization of the region/bounding box, and the
        # next time for the image that is cropped (and resized)
        # to the predicted (and expanded) region. In other words,
        # the fully connected graphs pertaining to the region and
        # digits prediction use the same convlution graph.

        # Inference - Region (taking input from convolutional)
        with tf.name_scope('region'):
            bbox_tensor = region_model.fc_graph(
                conv_pool, num_conv_pool, last_conv_depth, dropout)

        # Inference - Length and digits (taking input from convolutional)
        with tf.name_scope('digits'):
            length_tensor, digits_tensor = digits_model.fc_graph(
                conv_pool, num_conv_pool, last_conv_depth, masks, dropout)

            # Convert logits into log softmax outputs
            # for easing confidence prediction
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

        # Run inference for each image found in the input directory
        for input_file in files:
            # Skip directories
            if os.path.isdir(os.path.join(args.dir, input_file)):
                continue

            # Read input image and verify if it's a valid readable image
            image = cv2.imread(os.path.join(args.dir, input_file))
            if type(image) == type(None):
                print "Invalid file", input_file
                continue

            # Resize and normalize the image to be compatible for region
            # prediction
            compat_image = cv2.resize(
                image, (BIG_IMAGE_WIDTH, BIG_IMAGE_HEIGHT))
            compat_image = (compat_image / 255.0) - 0.5

            # Obtain the localized region of digits in the image
            bbox_pred = sess.run(bbox_tensor, feed_dict={
                                 image_pl: compat_image[None, :, :, :]})

            # Find the crop co-ordinates after region expansion
            bbox_left, bbox_top, bbox_width, bbox_height = bbox_pred[
                0, 1], bbox_pred[0, 0], bbox_pred[0, 3], bbox_pred[0, 2]
            crop_top = max(
                0, int(bbox_top - bbox_height * BBOX_EXPAND_RATIO / 2))
            crop_bottom = min(BIG_IMAGE_HEIGHT, int(
                (bbox_top + bbox_height) + bbox_height * BBOX_EXPAND_RATIO / 2))
            crop_left = max(
                0, int(bbox_left - bbox_width * BBOX_EXPAND_RATIO / 2))
            crop_right = min(BIG_IMAGE_WIDTH, int(
                (bbox_left + bbox_width) + bbox_width * BBOX_EXPAND_RATIO / 2))

            # Store expanded bbox co-ordinates for later use
            e_bbox_top = crop_top
            e_bbox_left = crop_left
            e_bbox_height = crop_bottom - crop_top
            e_bbox_width = crop_right - crop_left

            # Crop and resize the image to be compatible for recognizing digits
            cropped_image = compat_image[
                crop_top:crop_bottom, crop_left:crop_right]
            resized_image = cv2.resize(
                cropped_image, (SMALL_IMAGE_WIDTH, SMALL_IMAGE_HEIGHT))

            # Obtain the predicted length and digits (log softmax-ed) for the
            # crop-resized image
            length_pred, digits_pred = sess.run([length_log_softmax, digits_log_softmax], feed_dict={
                                                image_pl: resized_image[None, :, :, :]})

            # Workout predicted length from the
            # length logits (log softmax-ed)
            length = np.argmax(length_pred, axis=1)

            # Since batch size is only 1 (current image)
            # store the "predicted length" in a different
            # variable for better readability
            pred_length = length[0]

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

            # Since batch size is only 1 (current image)
            # store the "predicted digits" in a different
            # variable for better readability
            pred_digits = digits[:pred_length, 0]

            # Calculate total confidence for each sequence length by
            # adding up log probability of different predicted lengths for a
            # given image and max log probabilites of each predicted digit for
            # that image. Finally, exponentiate the added joint probabilities.
            # NOTE: The below approach with (1,1) shaped array and the like
            # might seem convoluted for a single prediction, but this generalizes
            # for predictions of any given batch size, where the shape then
            # changes to (BATCH_SIZE, 1)
            total_confidence = length_pred + \
                np.cumsum(
                    np.hstack([np.zeros([1, 1]), digits_max_log.T]), axis=1)
            total_confidence_antilog = np.exp(total_confidence)

            # Since batch size is only 1 (current image)
            # store the "confidence in prediction" in a different
            # variable for better readability
            pred_confidence = total_confidence_antilog[
                0, pred_length] / np.sum(total_confidence_antilog[0])

            # Populate the predicted labels
            # namely, length and digits into a string
            # for the current image
            if pred_length == 0:
                P = str(pred_length) + ' length'
            elif pred_length > MAX_DIGITS:
                P = '>' + str(MAX_DIGITS) + ' length'
            else:
                P = str(pred_length) + '|' + \
                    ''.join(pred_digits.astype(np.character).tolist())

            # Convert predicted confidence to a string
            C = str(np.round(pred_confidence * 100.0, 2))

            # Append predicted information as 'title' of the image
            title = 'P:' + P + ' ' + 'C:' + C + '%'

            # Display the image along with localized region, predicted 
            # information and confidence to the user
            ax = plt.axes()
            current_image = compat_image
            plt.imshow(current_image + 0.5)
            ax.add_patch(patches.Rectangle((e_bbox_left, e_bbox_top),
                                           e_bbox_width, e_bbox_height, fill=False))
            plt.axis('off')
            plt.title(title)
            plt.show()

if __name__ == '__main__':
    tf.app.run()
