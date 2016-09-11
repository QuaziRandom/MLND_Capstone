import sys

import numpy as np
import tensorflow as tf

from dataset.load_mnist import load_mnist

def main(_):
    mnist = load_mnist()

    def conv_graph(images):
        with tf.name_scope('conv1'):
            weights = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name='weights')
            biases = tf.Variable(tf.zeros([32]), name='biases')
            conv1 = tf.nn.relu(tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME') + biases)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        with tf.name_scope('conv2'):
            weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name='weights')
            biases = tf.Variable(tf.zeros([64]), name='biases')
            conv2 = tf.nn.relu(tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='SAME') + biases)
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        with tf.name_scope('hidden1'):
            pool2_flattened = tf.reshape(pool2, [-1, 7 * 7 * 64])
            weights = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.01), name='weights')
            biases = tf.Variable(tf.zeros([1024]), name='biases')
            hidden1 = tf.nn.relu(tf.matmul(pool2_flattened, weights) + biases)
        with tf.name_scope('readout'):
            weights = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.01), name='weights')
            biases = tf.Variable(tf.zeros([10]), name='biases')
            logits = tf.matmul(hidden1, weights) + biases
        
        return logits
    
    def loss_graph(logits, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
        return loss

    def train_graph(loss):
        optimizer = tf.train.AdamOptimizer(1e-4)
        train_op = optimizer.minimize(loss)
        return train_op

    def evaluate_graph(logits, labels):
        correct = tf.nn.in_top_k(logits, labels, 1)
        return tf.reduce_sum(tf.cast(correct, tf.float32))

    with tf.Graph().as_default():
        tf_train_images = tf.placeholder(tf.float32, [128, 28, 28, 1])
        tf_train_labels = tf.placeholder(tf.int32, [128])

        logits = conv_graph(tf_train_images)
        loss = loss_graph(logits, tf_train_labels)
        train_step = train_graph(loss)
        evaluation = evaluate_graph(logits, tf_train_labels)
    
        with tf.Session() as sess:
            tf.initialize_all_variables().run()

            for step in xrange(15000):
                offset = (step * 128) % (mnist.train_images.shape[0] - 128)
                images = mnist.train_images[offset:(offset + 128), :, :, None]
                labels = mnist.train_labels[offset:(offset + 128)]
                feed_dict = {tf_train_images: images, tf_train_labels: labels}

                _, loss_value = sess.run([train_step, loss], feed_dict=feed_dict)

                def evaluate_on_dataset(_images, _labels):
                    true_count = 0.0
                    for _step in xrange(_images.shape[0] // 128):
                        _offset = (_step * 128)
                        _images_batch = _images[_offset:(_offset + 128), :, :, None]
                        _labels_batch = _labels[_offset:(_offset + 128)]
                        _feed_dict = {tf_train_images: _images_batch, tf_train_labels: _labels_batch}
                        true_count += sess.run(evaluation, feed_dict=_feed_dict)
                    return true_count / _images.shape[0]

                if step % 100 == 0:
                    print "Step {}: loss = {}, valid_accuracy = {}".format(
                        step, loss_value, evaluate_on_dataset(mnist.valid_images, mnist.valid_labels))

                if (step + 1) % 1000 == 0:      
                    print "train_accuracy = {}".format(evaluate_on_dataset(mnist.train_images, mnist.train_labels))
                    print "test_accuracy = {}".format(evaluate_on_dataset(mnist.test_images, mnist.test_labels))

if __name__ == '__main__':
    tf.app.run()