import sys, os
import numpy as np
import tensorflow as tf

from dataset.load_mnist import load_mnist as load_mnist_single
from dataset.load_mnist_multi import MNISTMulti

TRAIN_SIZE = 2**16
VALID_SIZE = 2**12
TEST_SIZE = 2**14
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
BATCH_SIZE = 128

CONV_1_DEPTH = 32
CONV_2_DEPTH = 64
CONV_3_DEPTH = 128
CONV_4_DEPTH = 128
HIDDEN_1_NODES = 2048
LENGTH_LAYER_NODES = 6
DIGIT_LAYER_NODES = 10
MAX_DIGITS = 4

def main(_):
    mnist_single = load_mnist_single(normalized=False)
    
    train_data = MNISTMulti(
        mnist_single.train_images, mnist_single.train_labels, 101010, BATCH_SIZE, TRAIN_SIZE)
    valid_data = MNISTMulti(
        mnist_single.valid_images, mnist_single.valid_labels, 666, BATCH_SIZE, VALID_SIZE)
    test_data = MNISTMulti(
        mnist_single.test_images, mnist_single.test_labels, 314151, BATCH_SIZE, TEST_SIZE)

    def conv_graph(images):
        num_conv_pool = 0
        with tf.name_scope('conv1'):
            weights = tf.Variable(tf.truncated_normal([5, 5, 1, CONV_1_DEPTH], stddev=0.1), name='weights')
            biases = tf.Variable(tf.zeros([CONV_1_DEPTH]), name='biases')
            conv1 = tf.nn.relu(tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME') + biases)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            num_conv_pool += 1
        with tf.name_scope('conv2'):
            weights = tf.Variable(tf.truncated_normal([5, 5, CONV_1_DEPTH, CONV_2_DEPTH], stddev=0.1), name='weights')
            biases = tf.Variable(tf.zeros([CONV_2_DEPTH]), name='biases')
            conv2 = tf.nn.relu(tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='SAME') + biases)
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            num_conv_pool += 1
        with tf.name_scope('conv3'):
            weights = tf.Variable(tf.truncated_normal([5, 5, CONV_2_DEPTH, CONV_3_DEPTH], stddev=1e-2), name='weights')
            biases = tf.Variable(tf.zeros([CONV_3_DEPTH]), name='biases')
            conv3 = tf.nn.relu(tf.nn.conv2d(pool2, weights, strides=[1, 1, 1, 1], padding='SAME') + biases)
            pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            num_conv_pool += 1
        with tf.name_scope('conv4'):
            weights = tf.Variable(tf.truncated_normal([5, 5, CONV_3_DEPTH, CONV_4_DEPTH], stddev=1e-2), name='weights')
            biases = tf.Variable(tf.zeros([CONV_4_DEPTH]), name='biases')
            conv4 = tf.nn.relu(tf.nn.conv2d(pool3, weights, strides=[1, 1, 1, 1], padding='SAME') + biases)
            pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            num_conv_pool += 1
        
        return pool4, num_conv_pool, CONV_4_DEPTH

    def fc_graph(pool_layer, num_conv_pool_layers, last_conv_depth, masks):
        reduced_height = IMAGE_HEIGHT // (2**num_conv_pool_layers)
        reduced_width = IMAGE_WIDTH // (2**num_conv_pool_layers)
        pool_flat = tf.reshape(pool_layer, [-1, reduced_height * reduced_width * last_conv_depth])

        with tf.name_scope('hidden1'):
            weights = tf.Variable(tf.truncated_normal(
                [reduced_height * reduced_width * last_conv_depth, HIDDEN_1_NODES], stddev=1e-3), name='weights')
            biases = tf.Variable(tf.zeros([HIDDEN_1_NODES]), name='biases')
            hidden1 = tf.nn.relu(tf.matmul(pool_flat, weights) + biases)
        
        with tf.name_scope('readout_length'):
            weights = tf.Variable(
                tf.truncated_normal([HIDDEN_1_NODES, LENGTH_LAYER_NODES], stddev=1e-1), name='weights')
            biases = tf.Variable(tf.zeros([LENGTH_LAYER_NODES]))
            length_logits = tf.matmul(hidden1, weights) + biases

        def readout_digit_graph(scope_name):
            with tf.name_scope(scope_name):
                weights = tf.Variable(
                    tf.truncated_normal([HIDDEN_1_NODES, DIGIT_LAYER_NODES], stddev=1e-1), name='weights')
                biases = tf.Variable(tf.zeros([DIGIT_LAYER_NODES]))
                logits = tf.matmul(hidden1, weights) + biases
            return logits
        
        digits_logits = []
        digits_logits.append(readout_digit_graph('readout_digit1') * tf.reshape(masks[:,0], [-1,1]))
        digits_logits.append(readout_digit_graph('readout_digit2') * tf.reshape(masks[:,1], [-1,1]))
        digits_logits.append(readout_digit_graph('readout_digit3') * tf.reshape(masks[:,2], [-1,1]))
        digits_logits.append(readout_digit_graph('readout_digit4') * tf.reshape(masks[:,3], [-1,1]))

        return length_logits, digits_logits

    def loss_graph(length_logits, digits_logits, label_length, label_digits):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(length_logits, label_length)
        for i in range(MAX_DIGITS):
            xentropy += tf.nn.sparse_softmax_cross_entropy_with_logits(digits_logits[i], label_digits[:, i])
        xentropy_mean = tf.reduce_mean(xentropy)
        return xentropy_mean

    def train_graph(loss):
        optimizer = tf.train.AdamOptimizer(1e-4)
        train_op = optimizer.minimize(loss)
        return train_op

    def eval_graph(length_logits, digits_logits, labels_length, labels_digits):
        length_pred = tf.argmax(length_logits, 1)
        length_correct = tf.equal(length_pred, tf.to_int64(labels_length))
        
        digits_pred = []
        for i in range(MAX_DIGITS):
            digits_pred.append(tf.argmax(digits_logits[i], 1))
        digits_correct = []
        for i in range(MAX_DIGITS):
            digits_correct.append(tf.equal(digits_pred[i], tf.to_int64(labels_digits[:, i])))
        
        total_correct = tf.logical_and(length_correct, digits_correct[0])
        for i in range(1, MAX_DIGITS):
            total_correct = tf.logical_and(total_correct, digits_correct[i])

        return tf.reduce_mean(tf.cast(total_correct, tf.float32))

    def generate_feed_dict(data, images_pl, length_labels_pl, digits_labels_pl, masks_pl):
        images, labels = data.next_batch()
        length_labels = []
        digits_labels = []
        masks = []
        for l in labels:
            digits = np.zeros(MAX_DIGITS)
            mask = np.zeros(MAX_DIGITS)
            length = 0

            if l != '_':
                length = len(l)
                if length <= MAX_DIGITS:
                    digits[:length] = np.array([int(d) for d in l])
                    mask[:length] = 1
            
            length_labels.append(length)
            digits_labels.append(digits)
            masks.append(mask)
        
        feed_dict = {
            images_pl: images[:, :, :, None],
            length_labels_pl: length_labels,
            digits_labels_pl: digits_labels,
            masks_pl: masks
        }

        return feed_dict

    def test_valid_eval(sess, eval_batch, test_valid_data, images_pl, length_labels_pl, digits_labels_pl, masks_pl):
        num_batches = test_valid_data.get_dataset_size() // BATCH_SIZE
        correct = 0.0
        for _ in range(num_batches):
            feed_dict = generate_feed_dict(test_valid_data, images_pl, length_labels_pl, digits_labels_pl, masks_pl)
            correct += sess.run(eval_batch, feed_dict=feed_dict) * BATCH_SIZE
        accuracy = correct / (num_batches * BATCH_SIZE) * 100.0
        return accuracy

    with tf.Graph().as_default() as graph:
        images_pl = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        length_labels_pl = tf.placeholder(tf.int32, [BATCH_SIZE])
        digits_labels_pl = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_DIGITS])
        masks_pl = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_DIGITS])

        conv_pool, num_conv_pool, last_conv_depth = conv_graph(images_pl)
        length_logits, digits_logits = fc_graph(conv_pool, num_conv_pool, last_conv_depth, masks_pl)
        loss = loss_graph(length_logits, digits_logits, length_labels_pl, digits_labels_pl)
        train_step = train_graph(loss)
        batch_eval = eval_graph(length_logits, digits_logits, length_labels_pl, digits_labels_pl)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        for step in xrange(2000):
            feed_dict = generate_feed_dict(train_data, images_pl, length_labels_pl, digits_labels_pl, masks_pl)
            _, loss_value = sess.run([train_step, loss], feed_dict=feed_dict)

            if step % 20 == 0:
                batch_accuracy = sess.run(batch_eval, feed_dict=feed_dict) * 100.0
                print "Step {}: Loss = {}, Batch accuracy = {}%".format(step, loss_value, batch_accuracy)
            
            if step % 40 == 0:
                valid_accuracy = test_valid_eval(
                    sess, batch_eval, valid_data, images_pl, length_labels_pl, digits_labels_pl, masks_pl)
                print "Valid accuracy = {}%".format(valid_accuracy)


if __name__ == '__main__':
    tf.app.run()