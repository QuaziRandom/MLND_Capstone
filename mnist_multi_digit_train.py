"""Trainer for MNIST multi-digit model."""

import sys
import os
import numpy as np
import tensorflow as tf

import mnist_multi_digit as model
import mnist_multi_digit_input as inputs

from helpers import variable_summary, activation_summary
from helpers import parse_cmd_options

## Constants
# Dataset parameters
BATCH_SIZE = inputs.BATCH_SIZE
IMAGE_WIDTH = inputs.IMAGE_WIDTH
IMAGE_HEIGHT = inputs.IMAGE_HEIGHT
MAX_DIGITS = model.MAX_DIGITS

# Dropout probability
DROPOUT_KEEP_PROB = 0.5

# Learning rate parameters
LR_INIT_VALUE = 8e-4
LR_DECAY_RATE = 0.9
LR_DECAY_STEPS = 1000  # Roughly two epochs

# Max number of training steps
MAX_STEPS = 15000 + 1

# Default work directories
DEFAULT_LOG_DIR = 'logs/mnist_multi_digit'
DEFAULT_CP_DIR = 'checkpoints/mnist_multi_digit'


def test_valid_eval(sess, eval_batch, test_valid_data, images_pl, length_labels_pl, digits_labels_pl, masks_pl, dropout_pl):
    """Evaluates accuracy for the entire set of validation or test dataset."""
    num_batches = test_valid_data.get_dataset_size() // BATCH_SIZE
    correct = 0.0
    # Run evaluation op for all batches in dataset
    for _ in range(num_batches):
        feed_dict = inputs.generate_feed_dict(
            test_valid_data, images_pl, length_labels_pl, digits_labels_pl, masks_pl)
        feed_dict[dropout_pl] = 1.0
        correct += sess.run(eval_batch, feed_dict=feed_dict) * BATCH_SIZE
    accuracy = correct / (num_batches * BATCH_SIZE) * 100.0
    return accuracy


def main(argv):
    # Parse options from command-line
    args = parse_cmd_options(argv)

    run_name = args.run_name
    log_dir = os.path.join(args.logdir, run_name) if args.logdir else os.path.join(
        DEFAULT_LOG_DIR, run_name)
    cp_dir = os.path.join(args.cpdir, run_name) if args.cpdir else os.path.join(
        DEFAULT_CP_DIR, run_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    elif args.reset_logdir and os.listdir(log_dir) != []:
        # Better safe than sorry
        print "Please manually remove all files in {}".format(log_dir)
        return

    if not os.path.exists(cp_dir):
        os.makedirs(cp_dir)
    elif args.reset_cpdir and os.listdir(cp_dir) != []:
        # Better safe than sorry
        print "Please manually remove all files in {}".format(cp_dir)
        return

    # Load MNIST multi-digit train, valid and test datasets
    train_data, valid_data, test_data = inputs.create_multi_digit_datasets()

    with tf.Graph().as_default() as graph:
        # Placeholders to provide inputs to the computation graph
        images_pl = tf.placeholder(
            tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        length_labels_pl = tf.placeholder(tf.int32, [BATCH_SIZE])
        digits_labels_pl = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_DIGITS])
        masks_pl = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_DIGITS])
        dropout_pl = tf.placeholder(tf.float32)

        # Global step to keep track and for use in exponential 
        # decayed learning rate
        global_step = tf.Variable(0, trainable=False)

        # Graph construction
        # Inference
        conv_pool, num_conv_pool, last_conv_depth = model.conv_graph(images_pl)
        length_logits, digits_logits = model.fc_graph(
            conv_pool, num_conv_pool, last_conv_depth, masks_pl, dropout_pl)

        # Train
        loss = model.loss_graph(
            length_logits, digits_logits, length_labels_pl, digits_labels_pl)
        train_step = model.train_graph(
            loss, global_step, LR_DECAY_STEPS, LR_INIT_VALUE, LR_DECAY_RATE, False)

        # Evaluation
        batch_eval = model.eval_graph(
            length_logits, digits_logits, length_labels_pl, digits_labels_pl)

        # Summary writer op on the graph
        merged_summaries = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(log_dir, graph)

        # Placeholders to keep track of accuracy
        # Since accuracy values are not directly produced by graph
        # for entire datasets, the calculated values need to be fed
        # back into the graph summary ops
        with tf.name_scope('accuracy_summary'):
            batch_accuracy_pl = tf.placeholder(tf.float32)
            batch_accuracy_summary = tf.scalar_summary(
                'accuracy/batch', batch_accuracy_pl, collections='accuracies')
            valid_accuracy_pl = tf.placeholder(tf.float32)
            valid_accuracy_summary = tf.scalar_summary(
                'accuracy/valid', valid_accuracy_pl, collections='accuracies')
            test_accuracy_pl = tf.placeholder(tf.float32)
            test_accuracy_summary = tf.scalar_summary(
                'accuracy/test', test_accuracy_pl, collections='accuracies')

        # Saver to regularly save and/or restore checkpoints
        saver = tf.train.Saver()

        # Default session
        sess = tf.Session()

        # If checkpoint already exists, restore graph from that point.
        # Else, start everything fresh.
        ckpt = tf.train.get_checkpoint_state(cp_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            start_step = int(ckpt.model_checkpoint_path.split(
                '/')[-1].split('-')[-1]) + 1
            if start_step >= (MAX_STEPS - 1):
                print "Model already trained to {} steps".format(MAX_STEPS - 1)
                return
            global_step.assign(start_step)
            print "Restoring from {} at step {}".format(ckpt.model_checkpoint_path, start_step)
        else:
            sess.run(tf.initialize_all_variables())
            start_step = 0

        # Main training loop
        for step in xrange(start_step, MAX_STEPS):
            # Populate data into feed dictionary
            feed_dict = inputs.generate_feed_dict(
                train_data, images_pl, length_labels_pl, digits_labels_pl, masks_pl)
            feed_dict[dropout_pl] = DROPOUT_KEEP_PROB

            # Run a single train step
            _, loss_value = sess.run([train_step, loss], feed_dict=feed_dict)

            # Log training statistics, variable summaries regularly
            if step % 50 == 0:
                batch_accuracy, summary = sess.run(
                    [batch_eval, merged_summaries], feed_dict=feed_dict)
                print "Step {}: Loss = {}, Batch accuracy = {}%".format(step, loss_value, batch_accuracy * 100.0)
                summary_writer.add_summary(summary, step)
                accuracy_summary = sess.run(batch_accuracy_summary, feed_dict={
                                            batch_accuracy_pl: batch_accuracy})
                summary_writer.add_summary(accuracy_summary, step)

            # Compute accuracy on valid dataset once in a while
            # and log that using summary writer
            if step != 0 and step % 500 == 0:
                valid_accuracy = test_valid_eval(
                    sess, batch_eval, valid_data, images_pl, length_labels_pl, digits_labels_pl, masks_pl, dropout_pl)
                print "Valid accuracy = {}%".format(valid_accuracy)
                accuracy_summary = sess.run(valid_accuracy_summary, feed_dict={
                                            valid_accuracy_pl: valid_accuracy})
                summary_writer.add_summary(accuracy_summary, step)

            # Compute accuracy on valid dataset once in a while
            # and log that using summary writer.
            # Also checkpoint the model at this step.
            if step != 0 and step % 1000 == 0:
                test_accuracy = test_valid_eval(
                    sess, batch_eval, test_data, images_pl, length_labels_pl, digits_labels_pl, masks_pl, dropout_pl)
                print "Test accuracy = {}%".format(test_accuracy)
                accuracy_summary = sess.run(test_accuracy_summary, feed_dict={
                                            test_accuracy_pl: test_accuracy})
                summary_writer.add_summary(accuracy_summary, step)
                saver_path = saver.save(sess, os.path.join(
                    cp_dir, 'model.ckpt'), global_step=step)
                print "Model checkpoint created at {}".format(saver_path)


if __name__ == '__main__':
    tf.app.run()
