"""Model for MNIST multi-digit recognition.

Four convolutional layers, interspersed with max pooling,
followed by fully connected net with two hidden layers. The
final hidden layer branches into several softmax layers:
one for predicting the length of the sequence and the rest
for each digit.
"""

import tensorflow as tf
import mnist_multi_digit_input as inputs
from helpers import variable_summary, activation_summary

## Constants
# Model parameters
CONV_1_DEPTH = 32
CONV_2_DEPTH = 64
CONV_3_DEPTH = 128
CONV_4_DEPTH = 128
HIDDEN_1_NODES = 2048
HIDDEN_2_NODES = 1024
LENGTH_LAYER_NODES = 6
DIGIT_LAYER_NODES = 10
MAX_DIGITS = 4

# Input parameters
IMAGE_WIDTH = inputs.IMAGE_WIDTH
IMAGE_HEIGHT = inputs.IMAGE_HEIGHT


def conv_graph(images):
    """Defines the convolutional part of the inference network."""
    num_conv_pool = 0
    with tf.name_scope('conv1'):
        weights = tf.Variable(tf.truncated_normal(
            [5, 5, 1, CONV_1_DEPTH], stddev=0.1), name='weights')
        biases = tf.Variable(tf.zeros([CONV_1_DEPTH]), name='biases')
        variable_summary(weights.name, weights)
        variable_summary(biases.name, biases)
        conv1 = tf.nn.relu(tf.nn.conv2d(images, weights, strides=[
                           1, 1, 1, 1], padding='SAME', name='conv') + biases, name='relu')
        activation_summary(conv1.name, conv1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[
                               1, 2, 2, 1], padding='SAME', name='pool')
        num_conv_pool += 1
    with tf.name_scope('conv2'):
        weights = tf.Variable(tf.truncated_normal(
            [5, 5, CONV_1_DEPTH, CONV_2_DEPTH], stddev=0.1), name='weights')
        biases = tf.Variable(tf.zeros([CONV_2_DEPTH]), name='biases')
        variable_summary(weights.name, weights)
        variable_summary(biases.name, biases)
        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, weights, strides=[
                           1, 1, 1, 1], padding='SAME', name='conv') + biases, name='relu')
        activation_summary(conv2.name, conv2)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[
                               1, 2, 2, 1], padding='SAME', name='pool')
        num_conv_pool += 1
    with tf.name_scope('conv3'):
        weights = tf.Variable(tf.truncated_normal(
            [5, 5, CONV_2_DEPTH, CONV_3_DEPTH], stddev=0.1), name='weights')
        biases = tf.Variable(tf.zeros([CONV_3_DEPTH]), name='biases')
        variable_summary(weights.name, weights)
        variable_summary(biases.name, biases)
        conv3 = tf.nn.relu(tf.nn.conv2d(pool2, weights, strides=[
                           1, 1, 1, 1], padding='SAME', name='conv') + biases, name='relu')
        activation_summary(conv3.name, conv3)
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[
                               1, 2, 2, 1], padding='SAME', name='pool')
        num_conv_pool += 1
    with tf.name_scope('conv4'):
        weights = tf.Variable(tf.truncated_normal(
            [5, 5, CONV_3_DEPTH, CONV_4_DEPTH], stddev=5e-2), name='weights')
        biases = tf.Variable(tf.zeros([CONV_4_DEPTH]), name='biases')
        variable_summary(weights.name, weights)
        variable_summary(biases.name, biases)
        conv4 = tf.nn.relu(tf.nn.conv2d(pool3, weights, strides=[
                           1, 1, 1, 1], padding='SAME', name='conv') + biases, name='relu')
        activation_summary(conv4.name, conv4)
        pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[
                               1, 2, 2, 1], padding='SAME', name='pool')
        num_conv_pool += 1

    return pool4, num_conv_pool, CONV_4_DEPTH


def fc_graph(pool_layer, num_conv_pool_layers, last_conv_depth, masks, dropout_keep_prob):
    """Defines the fully connected part of the inference network.
    
    Inputs:
        Relevant information of previous convolutional layers.
        Masks defining the digits to be masked during training.
            Masks depend on the length of sequence, and are left aligned.
            For 4 maximum supported digits, and an input data with 3 digits,
            mask becomes [1, 1, 1, 0].
        Dropout keep probability to apply between the two hidden layers.
    """

    # Find the reduction in image dimensions due to the several
    # 2-stride, 2-size max poolings
    reduced_height = IMAGE_HEIGHT // (2**num_conv_pool_layers)
    reduced_width = IMAGE_WIDTH // (2**num_conv_pool_layers)

    with tf.name_scope('conv_fc_interface'):
        pool_flat = tf.reshape(
            pool_layer, [-1, reduced_height * reduced_width * last_conv_depth], name='flatten')

    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal(
            [reduced_height * reduced_width * last_conv_depth, HIDDEN_1_NODES], stddev=1e-3), name='weights')
        biases = tf.Variable(tf.zeros([HIDDEN_1_NODES]), name='biases')
        variable_summary(weights.name, weights)
        variable_summary(biases.name, biases)
        hidden1 = tf.nn.relu(
            tf.matmul(pool_flat, weights) + biases, name='relu')
        activation_summary(hidden1.name, hidden1)
        hidden1_drop = tf.nn.dropout(
            hidden1, dropout_keep_prob, name='dropout')

    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal(
            [HIDDEN_1_NODES, HIDDEN_2_NODES], stddev=1e-3), name='weights')
        biases = tf.Variable(tf.zeros([HIDDEN_2_NODES]), name='biases')
        variable_summary(weights.name, weights)
        variable_summary(biases.name, biases)
        hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, weights) + biases)
        activation_summary(hidden2.name, hidden2)

    # The softmax linear layer that outputs the length of the sequence
    # Output length is enumerated into 0 ... MAX_DIGITS, and >MAX_DIGITS
    with tf.name_scope('readout_length'):
        weights = tf.Variable(
            tf.truncated_normal([HIDDEN_2_NODES, LENGTH_LAYER_NODES], stddev=1e-1), name='weights')
        biases = tf.Variable(tf.zeros([LENGTH_LAYER_NODES]), name='biases')
        variable_summary(weights.name, weights)
        variable_summary(biases.name, biases)
        length_logits = tf.matmul(hidden2, weights) + biases

    # Helper to define same kind of graph for each
    # of the digit readout/softmax linear layer
    def readout_digit_graph(scope_name):
        with tf.name_scope(scope_name):
            weights = tf.Variable(
                tf.truncated_normal([HIDDEN_2_NODES, DIGIT_LAYER_NODES], stddev=1e-1), name='weights')
            biases = tf.Variable(tf.zeros([DIGIT_LAYER_NODES]), name='biases')
            variable_summary(weights.name, weights)
            variable_summary(biases.name, biases)
            logits = tf.matmul(hidden2, weights) + biases
        return logits

    # The readout for digits is masked at train time
    # If a particular digit is masked, all outputs of the readout [0..9] are multiplied by zero
    # If not, the logits are passed through untouched (multiplied by 1)
    # At test time, all digits are unmasked
    digits_logits = []
    digits_logits.append(readout_digit_graph(
        'readout_digit1') * tf.reshape(masks[:, 0], [-1, 1]))
    digits_logits.append(readout_digit_graph(
        'readout_digit2') * tf.reshape(masks[:, 1], [-1, 1]))
    digits_logits.append(readout_digit_graph(
        'readout_digit3') * tf.reshape(masks[:, 2], [-1, 1]))
    digits_logits.append(readout_digit_graph(
        'readout_digit4') * tf.reshape(masks[:, 3], [-1, 1]))

    return length_logits, digits_logits


def loss_graph(length_logits, digits_logits, label_length, label_digits):
    """Defines the loss function/methodology that is added to the graph."""
    # Typical of a logistic regression classifier, softmax cross entropy is
    # used for loss. Additionally, the cross entropy of each of the 'heads'
    # are added up -- this is analogous to joint probability of the
    # predicted length AND each of the predicted digit.
    with tf.name_scope('softmax_xentropy'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            length_logits, label_length, name='xentropy_length')
        for i in range(MAX_DIGITS):
            xentropy_digit = tf.nn.sparse_softmax_cross_entropy_with_logits(
                digits_logits[i], label_digits[:, i], name='xentropy_digit')
            xentropy += xentropy_digit
        xentropy_mean = tf.reduce_mean(xentropy, name='xentropy_combined')
    return xentropy_mean


def train_graph(loss, global_step, decay_steps=2000, init_lr=1e-3, lr_decay_rate=0.9, constant_lr=True):
    """Defines optimizer for the training with either a constant or exponentially decaying learning rate."""
    with tf.name_scope('train'):
        tf.scalar_summary('total_loss', loss)
        if not constant_lr:
            learning_rate = tf.train.exponential_decay(
                init_lr, global_step, decay_steps, lr_decay_rate, staircase=True)
        else:
            learning_rate = init_lr
        tf.scalar_summary('learning_rate', learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate, name='adam')
        train_op = optimizer.minimize(
            loss, global_step=global_step, name='minimize_loss')
    return train_op


def eval_graph(length_logits, digits_logits, labels_length, labels_digits):
    """Adds the evaluation metric to the graph."""
    # First the correct-ness of the length is found from the length labels
    with tf.name_scope('length_pred'):
        length_pred = tf.argmax(length_logits, 1)
        length_correct = tf.equal(length_pred, tf.to_int64(labels_length))

    # And the same for each digit is obtained from the digits labels
    with tf.name_scope('digits_pred'):
        digits_pred = []
        for i in range(MAX_DIGITS):
            digits_pred.append(tf.argmax(digits_logits[i], 1))
        digits_correct = []
        for i in range(MAX_DIGITS):
            digits_correct.append(
                tf.equal(digits_pred[i], tf.to_int64(labels_digits[:, i])))

    # Finally, they are combined with a logical AND. This ensures there
    # will be no partial credit, and correct-ness can only be deemed when
    # the length and all of the digits are predicted right.
    with tf.name_scope('combined_pred'):
        total_correct = tf.logical_and(length_correct, digits_correct[0])
        for i in range(1, MAX_DIGITS):
            total_correct = tf.logical_and(total_correct, digits_correct[i])
        accuracy = tf.reduce_mean(tf.cast(total_correct, tf.float32))

    return accuracy
