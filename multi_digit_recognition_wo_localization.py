"""Inference model to perform multi-digit recognition without localization.

This is a subset of the multi-digit recognition model, where the 
model only runs on images directly without localization. So the image inputs
must be already localized/cropped images. The predicted length and the digits 
of the image along with confidence is displayed to the user.

Check multi_digit_recognition.py for complete solution.
"""
import os
import argparse
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

import svhn_multi_digit as digits_model

# Checkpoint directory of trained SVHN multi-digit model
digits_model_dir = 'saved_models/svhn_multi_digit/'

# Parameters for digit model
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_DEPTH = 3
MAX_DIGITS = 5


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

        # Inference - Length and digits (taking input from convolutional)
        length_tensor, digits_tensor = digits_model.fc_graph(
            conv_pool, num_conv_pool, last_conv_depth, masks, dropout)

        # Convert logits into log softmax outputs
        # for easing confidence prediction
        length_log_softmax = tf.nn.log_softmax(length_tensor)
        digits_log_softmax = []
        for d in digits_tensor:
            digits_log_softmax.append(tf.nn.log_softmax(d))

        # Restorers to import the digits model
        digits_model_restorer = tf.train.Saver()

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

            # Resize and normalize the image to be compatible
            # for digits prediction
            compat_image = cv2.resize(
                image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            compat_image = (compat_image / 255.0) - 0.5

            # Obtain the predicted length and digits (log softmax-ed) for the
            # input image
            length_pred, digits_pred = sess.run([length_log_softmax, digits_log_softmax], feed_dict={
                                                image_pl: compat_image[None, :, :, :]})

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

            # Display the image along with predicted
            # information and confidence to the user
            plt.imshow(image)
            plt.axis('off')
            plt.title(title)
            plt.show()

if __name__ == '__main__':
    tf.app.run()
