import tensorflow as tf

import svhn_multi_digit_input as inputs

from helpers import variable_summary, activation_summary

# Global constants
CONV_1_DEPTH = 48
CONV_2_DEPTH = 64
CONV_3_DEPTH = 128
CONV_4_DEPTH = 192
HIDDEN_1_NODES = 4096
HIDDEN_2_NODES = 2048
LENGTH_LAYER_NODES = 6
DIGIT_LAYER_NODES = 10
MAX_DIGITS = 5

IMAGE_WIDTH = inputs.IMAGE_WIDTH
IMAGE_HEIGHT = inputs.IMAGE_HEIGHT
IMAGE_DEPTH = inputs.IMAGE_DEPTH

def conv_graph(images):
    num_conv_pool = 0
    with tf.name_scope('conv1'):
        weights = tf.Variable(tf.truncated_normal([5, 5, IMAGE_DEPTH, CONV_1_DEPTH], stddev=0.1), name='weights')
        biases = tf.Variable(tf.zeros([CONV_1_DEPTH]), name='biases')
        variable_summary(weights.name, weights)
        variable_summary(biases.name, biases)
        conv1 = tf.nn.relu(tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME', name='conv') + biases, name='relu')
        activation_summary(conv1.name, conv1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')
        num_conv_pool += 1
    with tf.name_scope('conv2'):
        weights = tf.Variable(tf.truncated_normal([5, 5, CONV_1_DEPTH, CONV_2_DEPTH], stddev=0.05), name='weights')
        biases = tf.Variable(tf.zeros([CONV_2_DEPTH]), name='biases')
        variable_summary(weights.name, weights)
        variable_summary(biases.name, biases)
        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='SAME', name='conv') + biases, name='relu')
        activation_summary(conv2.name, conv2)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')
        num_conv_pool += 1
    with tf.name_scope('conv3'):
        weights = tf.Variable(tf.truncated_normal([5, 5, CONV_2_DEPTH, CONV_3_DEPTH], stddev=0.05), name='weights')
        biases = tf.Variable(tf.zeros([CONV_3_DEPTH]), name='biases')
        variable_summary(weights.name, weights)
        variable_summary(biases.name, biases)
        conv3 = tf.nn.relu(tf.nn.conv2d(pool2, weights, strides=[1, 1, 1, 1], padding='SAME', name='conv') + biases, name='relu')
        activation_summary(conv3.name, conv3)
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')
        num_conv_pool += 1
    with tf.name_scope('conv4'):
        weights = tf.Variable(tf.truncated_normal([5, 5, CONV_3_DEPTH, CONV_4_DEPTH], stddev=0.01), name='weights')
        biases = tf.Variable(tf.zeros([CONV_4_DEPTH]), name='biases')
        variable_summary(weights.name, weights)
        variable_summary(biases.name, biases)
        conv4 = tf.nn.relu(tf.nn.conv2d(pool3, weights, strides=[1, 1, 1, 1], padding='SAME', name='conv') + biases, name='relu')
        activation_summary(conv4.name, conv4)
        pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')
        num_conv_pool += 1
    
    return pool4, num_conv_pool, CONV_4_DEPTH

def fc_graph(pool_layer, num_conv_pool_layers, last_conv_depth, masks, dropout_keep_prob):
    reduced_height = IMAGE_HEIGHT // (2**num_conv_pool_layers)
    reduced_width = IMAGE_WIDTH // (2**num_conv_pool_layers)

    with tf.name_scope('conv_fc_interface'):
        pool_flat = tf.reshape(pool_layer, [-1, reduced_height * reduced_width * last_conv_depth], name='flatten')

    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal(
            [reduced_height * reduced_width * last_conv_depth, HIDDEN_1_NODES], stddev=5e-4), name='weights')
        biases = tf.Variable(tf.zeros([HIDDEN_1_NODES]), name='biases')
        variable_summary(weights.name, weights)
        variable_summary(biases.name, biases)
        hidden1 = tf.nn.relu(tf.matmul(pool_flat, weights) + biases, name='relu')
        activation_summary(hidden1.name, hidden1)
        hidden1_drop = tf.nn.dropout(hidden1, dropout_keep_prob, name='dropout')
    
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([HIDDEN_1_NODES, HIDDEN_2_NODES], stddev=1e-3), name='weights')
        biases = tf.Variable(tf.zeros([HIDDEN_2_NODES]), name='biases')
        variable_summary(weights.name, weights)
        variable_summary(biases.name, biases)
        hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, weights) + biases)
        activation_summary(hidden2.name, hidden2)
    
    with tf.name_scope('readout_length'):
        weights = tf.Variable(
            tf.truncated_normal([HIDDEN_2_NODES, LENGTH_LAYER_NODES], stddev=1e-1), name='weights')
        biases = tf.Variable(tf.zeros([LENGTH_LAYER_NODES]), name='biases')
        variable_summary(weights.name, weights)
        variable_summary(biases.name, biases)
        length_logits = tf.matmul(hidden2, weights) + biases

    def readout_digit_graph(scope_name):
        with tf.name_scope(scope_name):
            weights = tf.Variable(
                tf.truncated_normal([HIDDEN_2_NODES, DIGIT_LAYER_NODES], stddev=1e-1), name='weights')
            biases = tf.Variable(tf.zeros([DIGIT_LAYER_NODES]), name='biases')
            variable_summary(weights.name, weights)
            variable_summary(biases.name, biases)
            logits = tf.matmul(hidden2, weights) + biases
        return logits
    
    digits_logits = []
    digits_logits.append(readout_digit_graph('readout_digit1') * tf.reshape(masks[:,0], [-1,1]))
    digits_logits.append(readout_digit_graph('readout_digit2') * tf.reshape(masks[:,1], [-1,1]))
    digits_logits.append(readout_digit_graph('readout_digit3') * tf.reshape(masks[:,2], [-1,1]))
    digits_logits.append(readout_digit_graph('readout_digit4') * tf.reshape(masks[:,3], [-1,1]))
    digits_logits.append(readout_digit_graph('readout_digit5') * tf.reshape(masks[:,4], [-1,1]))

    return length_logits, digits_logits

def loss_graph(length_logits, digits_logits, label_length, label_digits):
    with tf.name_scope('softmax_xentropy'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(length_logits, label_length, name='xentropy_length')
        for i in range(MAX_DIGITS):
            xentropy_digit = tf.nn.sparse_softmax_cross_entropy_with_logits(digits_logits[i], label_digits[:, i], name='xentropy_digit')
            xentropy += xentropy_digit
        xentropy_mean = tf.reduce_mean(xentropy, name='xentropy_combined')
    return xentropy_mean

def train_graph(loss, global_step, decay_steps=2000, init_lr=1e-3, lr_decay_rate=0.9, constant_lr=True):
    with tf.name_scope('train'):
        tf.scalar_summary('total_loss', loss)
        if not constant_lr:
            learning_rate = tf.train.exponential_decay(init_lr, global_step, decay_steps, lr_decay_rate, staircase=True)
        else:
            learning_rate = init_lr
        tf.scalar_summary('learning_rate', learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate, name='adam')
        train_op = optimizer.minimize(loss, global_step=global_step, name='minimize_loss')
    return train_op

def eval_graph(length_logits, digits_logits, labels_length, labels_digits):
    with tf.name_scope('length_pred'):
        length_pred = tf.argmax(length_logits, 1)
        length_correct = tf.equal(length_pred, tf.to_int64(labels_length))
    
    with tf.name_scope('digits_pred'):
        digits_pred = []
        for i in range(MAX_DIGITS):
            digits_pred.append(tf.argmax(digits_logits[i], 1))
        digits_correct = []
        for i in range(MAX_DIGITS):
            digits_correct.append(tf.equal(digits_pred[i], tf.to_int64(labels_digits[:, i])))
    
    with tf.name_scope('combined_pred'):
        total_correct = tf.logical_and(length_correct, digits_correct[0])
        for i in range(1, MAX_DIGITS):
            total_correct = tf.logical_and(total_correct, digits_correct[i])
        accuracy = tf.reduce_mean(tf.cast(total_correct, tf.float32))

    return accuracy