"""Trainer for SVHN region model."""

import sys
import os
import numpy as np
import tensorflow as tf

import svhn_region as model
import svhn_region_input as inputs

from helpers import variable_summary, activation_summary
from helpers import parse_cmd_options

## Constants
# Dataset parameters
BATCH_SIZE = inputs.BATCH_SIZE
IMAGE_WIDTH = inputs.IMAGE_WIDTH
IMAGE_HEIGHT = inputs.IMAGE_HEIGHT
IMAGE_DEPTH = inputs.IMAGE_DEPTH

# Dropout probability
DROPOUT_KEEP_PROB = 0.5

# Other parameters
# Part 1 - Train data
LR_INIT_VALUE = 1e-4
MAX_STEPS = 3000 + 1

# Part 2 - Test data (Refer to doc for more info)
# LR_INIT_VALUE = 2e-5
# MAX_STEPS = 6000 + 1

# Default work directories
DEFAULT_LOG_DIR = 'logs/svhn_region'
DEFAULT_CP_DIR = 'checkpoints/svhn_region'

# Checkpoint directory of trained digits model
# from which the convolution layer weights are used
TRAINED_CONV_CP_DIR = 'saved_models/svhn_multi_digit/'


def test_valid_eval(sess, loss, test_valid_data, images_pl, bboxes_pl, dropout_pl):
    """Evaluates average loss for the entire set of validation or test dataset."""
    num_batches = test_valid_data.get_dataset_size() // BATCH_SIZE
    total_loss = 0.0
    for _ in range(num_batches):
        feed_dict = inputs.generate_feed_dict(
            test_valid_data, images_pl, bboxes_pl)
        feed_dict[dropout_pl] = 1.0
        total_loss += sess.run(loss, feed_dict=feed_dict) * BATCH_SIZE
    avg_loss = total_loss / (num_batches * BATCH_SIZE)
    return avg_loss


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

    # Load SVHN region train, valid and test datasets
    # Additionally, making validation optional can help save memory
    # This helps in running this on PC with limited resources
    need_validation = args.validation
    train_data, valid_data, test_data = inputs.load_svhn_datasets(
        need_validation)

    with tf.Graph().as_default() as graph:
        # Placeholders to provide inputs to the computation graph
        images_pl = tf.placeholder(
            tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
        bboxes_pl = tf.placeholder(tf.float32, [BATCH_SIZE, 4])
        dropout_pl = tf.placeholder(tf.float32)

        # Global step to keep track and for use in exponential
        # decayed learning rate
        global_step = tf.Variable(0, trainable=False)

        # Graph construction
        # Inference
        conv_pool, num_conv_pool, last_conv_depth = model.conv_graph(images_pl)
        logits = model.fc_graph(conv_pool, num_conv_pool,
                                last_conv_depth, dropout_pl)

        # Train
        loss = model.loss_graph(logits, bboxes_pl)
        train_step = model.train_graph(
            loss, global_step, init_lr=LR_INIT_VALUE)

        # Summary writer op on the graph
        merged_summaries = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(log_dir, graph)

        # Placeholders to keep track of average loss
        # Since average loss values are not directly produced by graph
        # for entire datasets, the Python calculated values need to
        # be fed back into the graph summary ops
        with tf.name_scope('loss_summary'):
            valid_loss_pl = tf.placeholder(tf.float32)
            valid_loss_summary = tf.scalar_summary(
                'loss/valid', valid_loss_pl, collections='loss')
            test_loss_pl = tf.placeholder(tf.float32)
            test_loss_summary = tf.scalar_summary(
                'loss/test', test_loss_pl, collections='loss')

        # Get conv variables
        all_graph_vars = tf.all_variables()
        conv_variables = [
            var for var in all_graph_vars if var.name.startswith('conv')]

        # Spin up default and conv graph saver/restorer
        default_saver = tf.train.Saver()
        digits_conv_model = tf.train.Saver(conv_variables)

        # Default session
        sess = tf.Session()

        # If checkpoint already exists, restore graph from that point.
        # Else, restore conv variables from already trained multi-digits model
        # and initialize the rest of the variables
        ckpt = tf.train.get_checkpoint_state(cp_dir)
        if ckpt and ckpt.model_checkpoint_path:
            default_saver.restore(sess, ckpt.model_checkpoint_path)
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
            digits_ckpt = tf.train.get_checkpoint_state(TRAINED_CONV_CP_DIR)
            if digits_ckpt and digits_ckpt.model_checkpoint_path:
                digits_conv_model.restore(
                    sess, digits_ckpt.model_checkpoint_path)
                print "Trained conv restored from {}".format(digits_ckpt.model_checkpoint_path)
            else:
                print "Did not find digits trained conv model. Exiting."
                return

        # Add a debug op if debug options specified
        if args.debug:
            # Add an op that raises assertion if any op in the graph returns
            # inf or nan
            check_numerics = tf.add_check_numerics_ops()

        # Main training loop
        for step in xrange(start_step, MAX_STEPS):
            # Populate data into feed dictionary
            feed_dict = inputs.generate_feed_dict(
                train_data, images_pl, bboxes_pl)
            feed_dict[dropout_pl] = DROPOUT_KEEP_PROB

            # Run a single train step
            if args.debug:
                # If debug option specified, run debug op additionally for each
                # step
                _, loss_value, check = sess.run(
                    [train_step, loss, check_numerics], feed_dict=feed_dict)
                print step, check
            else:
                _, loss_value = sess.run(
                    [train_step, loss], feed_dict=feed_dict)

            # Log training statistics, variable summaries regularly
            if step % 50 == 0:
                summary = sess.run(merged_summaries, feed_dict=feed_dict)
                print "Step {}: Batch loss = {}".format(step, loss_value)
                summary_writer.add_summary(summary, step)

            # Compute accuracy on valid dataset (if needed)
            # once in a while and log that using summary writer
            if need_validation:
                if step != 0 and step % 500 == 0:
                    valid_avg_loss = test_valid_eval(
                        sess, loss, valid_data, images_pl, bboxes_pl, dropout_pl)
                    print "Valid average loss = {}".format(valid_avg_loss)
                    loss_summary = sess.run(valid_loss_summary, feed_dict={
                                            valid_loss_pl: valid_avg_loss})
                    summary_writer.add_summary(loss_summary, step)

            # Compute accuracy on valid dataset once in a while
            # and log that using summary writer.
            # Also checkpoint the model at this step.
            if step != 0 and step % 1000 == 0:
                test_avg_loss = test_valid_eval(
                    sess, loss, test_data, images_pl, bboxes_pl, dropout_pl)
                print "Test average loss = {}".format(test_avg_loss)
                loss_summary = sess.run(test_loss_summary, feed_dict={
                                        test_loss_pl: test_avg_loss})
                summary_writer.add_summary(loss_summary, step)

                saver_path = default_saver.save(sess, os.path.join(
                    cp_dir, 'model.ckpt'), global_step=step)
                print "Model checkpoint created at {}".format(saver_path)


if __name__ == '__main__':
    tf.app.run()
