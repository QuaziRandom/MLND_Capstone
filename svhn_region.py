"""Model for SVHN region localization.

Model re-uses the same convolutional net as
the SVHN multi-digit model. It replaces the fully
connected net from multi-digit model with two 
hidden layers, followed by a linear layer that 
directly outputs the localized bounding box.
"""
import tensorflow as tf
import numpy as np

import svhn_region_input as inputs
import svhn_multi_digit as digits_model

from helpers import variable_summary, activation_summary

## Constants
# Import conv model parameters
CONV_1_DEPTH = digits_model.CONV_1_DEPTH
CONV_2_DEPTH = digits_model.CONV_2_DEPTH
CONV_3_DEPTH = digits_model.CONV_3_DEPTH
CONV_4_DEPTH = digits_model.CONV_4_DEPTH
CONV_5_DEPTH = digits_model.CONV_5_DEPTH
CONV_6_DEPTH = digits_model.CONV_6_DEPTH

# Specify fully connected model parameters
HIDDEN_1_NODES = 2048
HIDDEN_2_NODES = 1024
BBOX_LAYER_NODES = 4

# Import input parameters
IMAGE_WIDTH = inputs.IMAGE_WIDTH
IMAGE_HEIGHT = inputs.IMAGE_HEIGHT
IMAGE_DEPTH = inputs.IMAGE_DEPTH


def he_init_std(n):
    """Helper to initialize weights according to He et al. (2015)"""
    return np.sqrt(2.0 / n)


def conv_graph(images):
    """Imports convolutional network from SVHN multi-digits model."""
    return digits_model.conv_graph(images, trainable=False)


def fc_graph(pool_layer, num_stride_two_pool_layers, last_conv_depth, dropout_keep_prob):
    """Defines the fully connected part of the inference network.
    
    Inputs:
        Relevant information of previous convolutional layers.
        Dropout keep probability to apply between hidden layers.
    """

    # Find the reduction in image dimensions due to the several
    # 2-stride, 2-size max poolings in the conv graph
    reduced_height = IMAGE_HEIGHT // (2**num_stride_two_pool_layers)
    reduced_width = IMAGE_WIDTH // (2**num_stride_two_pool_layers)

    with tf.name_scope('conv_fc_interface'):
        pool_flat = tf.reshape(
            pool_layer, [-1, reduced_height * reduced_width * last_conv_depth], name='flatten')

    with tf.name_scope('hidden1'):
        init_std = he_init_std(
            reduced_height * reduced_width * last_conv_depth)
        weights = tf.Variable(tf.truncated_normal(
            [reduced_height * reduced_width * last_conv_depth, HIDDEN_1_NODES], stddev=init_std), name='weights')
        biases = tf.Variable(tf.zeros([HIDDEN_1_NODES]), name='biases')
        variable_summary(weights.name, weights)
        variable_summary(biases.name, biases)
        hidden1 = tf.nn.relu(
            tf.matmul(pool_flat, weights) + biases, name='relu')
        activation_summary(hidden1.name, hidden1)
        hidden1_drop = tf.nn.dropout(
            hidden1, dropout_keep_prob, name='dropout')

    with tf.name_scope('hidden2'):
        init_std = he_init_std(HIDDEN_1_NODES)
        weights = tf.Variable(tf.truncated_normal(
            [HIDDEN_1_NODES, HIDDEN_2_NODES], stddev=init_std), name='weights')
        biases = tf.Variable(tf.zeros([HIDDEN_2_NODES]), name='biases')
        variable_summary(weights.name, weights)
        variable_summary(biases.name, biases)
        hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, weights) + biases)
        activation_summary(hidden2.name, hidden2)

    with tf.name_scope('bbox_output'):
        init_std = he_init_std(HIDDEN_2_NODES)
        weights = tf.Variable(tf.truncated_normal(
            [HIDDEN_2_NODES, BBOX_LAYER_NODES], stddev=init_std), name='weights')
        biases = tf.Variable(tf.zeros([BBOX_LAYER_NODES]), name='biases')
        variable_summary(weights.name, weights)
        variable_summary(biases.name, biases)
        logits = tf.matmul(hidden2, weights) + biases

    return logits


def loss_graph(logits, bboxes):
    """Defines the loss function/methodology that is added to the graph."""
    # The loss is computed between the predicted and actual bounding boxes
    # in an almost l2 loss manner. However, the top and left features of the
    # bounding boxes are weighted more than the height and width features.
    # So the total loss becomes:
    # 2 * sum((difference in top/left)**2) + sum((difference in height/width)**2)
    # In addition, the square root used in vanilla l2 loss is also dispensed.
    with tf.name_scope('modified_l2_loss'):
        squared_difference = tf.squared_difference(logits, bboxes)
        bbox_weights = tf.constant([2, 2, 1, 1], tf.float32)
        weighted_difference = tf.mul(squared_difference, bbox_weights)
        distance = tf.reduce_sum(weighted_difference, 1)
        loss = tf.reduce_mean(distance)
    return loss


def train_graph(loss, global_step, decay_steps=2000, init_lr=1e-4, lr_decay_rate=0.9, constant_lr=True):
    """Defines optimizer for the training with either a constant or exponentially decaying learning rate."""
    with tf.name_scope('train'):
        tf.scalar_summary('loss/batch', loss)
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
