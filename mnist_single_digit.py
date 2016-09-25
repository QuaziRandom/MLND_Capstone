"""A simple model to work on MNIST dataset.

This model is largely based on the 'Deep MNIST for Experts'
tutorial from TensorFlow documentation: Two convolutional
layers followed by one fully connected layer, ends with a 
softmax readout layer.

This script also integrates TF summary writers to log various
variables of the graph, and save/restore models in case of
a crash and/or for later use.
"""
import sys

import numpy as np
import tensorflow as tf

from dataset.load_mnist import load_mnist


def main(_):
    mnist = load_mnist()

    def make_summary(name, var):
        """Helper function for logging variable summaries."""
        with tf.name_scope('summary'):
            tf.scalar_summary('mean/' + name, tf.reduce_mean(var))
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)

    def conv_graph(images):
        """Adds the entire inference graph to the model."""
        with tf.name_scope('conv1'):
            weights = tf.Variable(tf.truncated_normal(
                [5, 5, 1, 32], stddev=0.1), name='weights')
            make_summary(weights.name, weights)
            biases = tf.Variable(tf.zeros([32]), name='biases')
            make_summary(biases.name, biases)
            conv1 = tf.nn.relu(tf.nn.conv2d(images, weights, strides=[
                               1, 1, 1, 1], padding='SAME') + biases)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[
                                   1, 2, 2, 1], padding='SAME')
        with tf.name_scope('conv2'):
            weights = tf.Variable(tf.truncated_normal(
                [5, 5, 32, 64], stddev=0.1), name='weights')
            make_summary(weights.name, weights)
            biases = tf.Variable(tf.zeros([64]), name='biases')
            make_summary(biases.name, biases)
            conv2 = tf.nn.relu(tf.nn.conv2d(pool1, weights, strides=[
                               1, 1, 1, 1], padding='SAME') + biases)
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[
                                   1, 2, 2, 1], padding='SAME')
        with tf.name_scope('hidden1'):
            pool2_flattened = tf.reshape(pool2, [-1, 7 * 7 * 64])
            weights = tf.Variable(tf.truncated_normal(
                [7 * 7 * 64, 1024], stddev=0.01), name='weights')
            make_summary(weights.name, weights)
            biases = tf.Variable(tf.zeros([1024]), name='biases')
            make_summary(biases.name, biases)
            hidden1 = tf.nn.relu(tf.matmul(pool2_flattened, weights) + biases)
        with tf.name_scope('readout'):
            weights = tf.Variable(tf.truncated_normal(
                [1024, 10], stddev=0.01), name='weights')
            make_summary(weights.name, weights)
            biases = tf.Variable(tf.zeros([10]), name='biases')
            make_summary(biases.name, biases)
            logits = tf.matmul(hidden1, weights) + biases
            tf.histogram_summary('logits', logits)

        return logits

    def loss_graph(logits, labels):
        """Adds loss function to the model."""
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels, name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
        tf.scalar_summary('cross_entropy_mean', loss)
        return loss

    def train_graph(loss):
        """Adds optimizer used for training the model."""
        optimizer = tf.train.AdamOptimizer(1e-4)
        train_op = optimizer.minimize(loss)
        return train_op

    def evaluate_graph(logits, labels):
        """Adds evaluation metric to the model."""
        correct = tf.nn.in_top_k(logits, labels, 1)
        return tf.reduce_sum(tf.cast(correct, tf.float32))

    with tf.Graph().as_default() as graph:
        # Placeholders to provide data into the graph
        tf_train_images = tf.placeholder(tf.float32, [128, 28, 28, 1])
        tf_train_labels = tf.placeholder(tf.int32, [128])

        # Graph construction
        # Inference
        logits = conv_graph(tf_train_images)

        # Train
        loss = loss_graph(logits, tf_train_labels)
        train_step = train_graph(loss)

        # Evaluation
        evaluation = evaluate_graph(logits, tf_train_labels)

        # Summary writer operations on the graph
        merged_summaries = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(
            'logs/mnist_single_digit', graph)

        # Saver to checkpoint models regularly during training
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # If checkpoint exists, resume training from last known point. Else
            # start anew.
            ckpt = tf.train.get_checkpoint_state(
                'checkpoints/mnist_single_digit')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                start_step = int(ckpt.model_checkpoint_path.split(
                    '/')[-1].split('-')[-1])
            else:
                tf.initialize_all_variables().run()
                start_step = 0

            # Main training loop
            for step in xrange(start_step, 20001):
                # Populate data into the feed dictionary
                offset = (step * 128) % (mnist.train_images.shape[0] - 128)
                images = mnist.train_images[offset:(offset + 128), :, :, None]
                labels = mnist.train_labels[offset:(offset + 128)]
                feed_dict = {tf_train_images: images, tf_train_labels: labels}

                # Run a single train step
                _, loss_value = sess.run(
                    [train_step, loss], feed_dict=feed_dict)

                def evaluate_on_dataset(_images, _labels):
                    """Runs evaluation graph for an entire set instead of for a batch."""
                    true_count = 0.0
                    for _step in xrange(_images.shape[0] // 128):
                        _offset = (_step * 128)
                        _images_batch = _images[
                            _offset:(_offset + 128), :, :, None]
                        _labels_batch = _labels[_offset:(_offset + 128)]
                        _feed_dict = {tf_train_images: _images_batch,
                                      tf_train_labels: _labels_batch}
                        true_count += sess.run(evaluation,
                                               feed_dict=_feed_dict)
                    return true_count / _images.shape[0]

                # Log training statistics regularly
                if step % 100 == 0:
                    print "Step {}: loss = {}, valid_accuracy = {}".format(
                        step, loss_value, evaluate_on_dataset(mnist.valid_images, mnist.valid_labels))
                    summary = sess.run(merged_summaries, feed_dict=feed_dict)
                    summary_writer.add_summary(summary, step)

                # Compute accuracy on entire test and training dataset once in
                # a while
                if step != 0 and step % 1000 == 0:
                    print "train_accuracy = {}".format(
                        evaluate_on_dataset(mnist.train_images, mnist.train_labels))
                    print "test_accuracy = {}".format(
                        evaluate_on_dataset(mnist.test_images, mnist.test_labels))
                    saver_path = saver.save(
                        sess, 'checkpoints/mnist_single_digit/model.ckpt', global_step=step)
                    print "New checkpoint created at {}".format(saver_path)

if __name__ == '__main__':
    tf.app.run()
