import sys, os
import numpy as np
import tensorflow as tf

import svhn_region as model
import svhn_region_input as inputs

from helpers import variable_summary, activation_summary
from helpers import parse_cmd_options

# Some global constants
BATCH_SIZE = inputs.BATCH_SIZE
IMAGE_WIDTH = inputs.IMAGE_WIDTH
IMAGE_HEIGHT = inputs.IMAGE_HEIGHT
IMAGE_DEPTH = inputs.IMAGE_DEPTH
DROPOUT_KEEP_PROB = 0.5

LR_INIT_VALUE = 1e-4
MAX_STEPS = 13000 + 1

DEFAULT_LOG_DIR = 'logs/svhn_region'
DEFAULT_CP_DIR = 'checkpoints/svhn_region'

TRAINED_CONV_CP_DIR = 'checkpoints/svhn_multi_digit/final_part_2'

def test_valid_eval(sess, loss, test_valid_data, images_pl, bboxes_pl, dropout_pl):
    num_batches = test_valid_data.get_dataset_size() // BATCH_SIZE
    total_loss = 0.0
    for _ in range(num_batches):
        feed_dict = inputs.generate_feed_dict(test_valid_data, images_pl, bboxes_pl)
        feed_dict[dropout_pl] = 1.0
        total_loss += sess.run(loss, feed_dict=feed_dict) * BATCH_SIZE
    avg_loss = total_loss / (num_batches * BATCH_SIZE)
    return avg_loss

def main(argv):
    args = parse_cmd_options(argv)

    run_name = args.run_name
    log_dir = os.path.join(args.logdir, run_name) if args.logdir else os.path.join(DEFAULT_LOG_DIR, run_name)
    cp_dir = os.path.join(args.cpdir, run_name) if args.cpdir else os.path.join(DEFAULT_CP_DIR, run_name)  

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    elif args.reset_logdir and os.listdir(log_dir) != []:
        # TODO: Add automatic (intelligent) reset later.
        print "Remove all files in {}".format(log_dir)
        return

    if not os.path.exists(cp_dir):
        os.makedirs(cp_dir)
    elif args.reset_cpdir and os.listdir(cp_dir) != []:
        # TODO: Add automatic (intelligent) reset later.
        print "Remove all files in {}".format(cp_dir)
        return
    
    # Making validation optional can help save memory
    # This helps in running this on PC with limited resources
    need_validation = args.validation
    train_data, valid_data, test_data = inputs.load_svhn_datasets(need_validation)

    with tf.Graph().as_default() as graph:
        images_pl = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
        bboxes_pl = tf.placeholder(tf.float32, [BATCH_SIZE, 4])
        dropout_pl = tf.placeholder(tf.float32)

        global_step = tf.Variable(0, trainable=False)

        conv_pool, num_conv_pool, last_conv_depth = model.conv_graph(images_pl)
        print conv_pool.get_shape()
        logits = model.fc_graph(conv_pool, num_conv_pool, last_conv_depth, dropout_pl)
        loss = model.loss_graph(logits, bboxes_pl)
        train_step = model.train_graph(loss, global_step, init_lr=LR_INIT_VALUE)

        merged_summaries = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(log_dir, graph)

        with tf.name_scope('loss_summary'):
            valid_loss_pl = tf.placeholder(tf.float32)
            valid_loss_summary = tf.scalar_summary('loss/valid', valid_loss_pl, collections='loss')
            test_loss_pl = tf.placeholder(tf.float32)
            test_loss_summary = tf.scalar_summary('loss/test', test_loss_pl, collections='loss')

        # Get conv variables
        all_graph_vars = tf.all_variables()
        conv_variables = [var for var in all_graph_vars if var.name.startswith('conv')]
        
        # Spin up default and conv graph saver/restorer
        default_saver = tf.train.Saver()
        digits_conv_model = tf.train.Saver(conv_variables) 

        sess = tf.Session()
        
        ckpt = tf.train.get_checkpoint_state(cp_dir)
        if ckpt and ckpt.model_checkpoint_path:
            default_saver.restore(sess, ckpt.model_checkpoint_path)
            start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]) + 1
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
                digits_conv_model.restore(sess, digits_ckpt.model_checkpoint_path)
                print "Trained conv restored from {}".format(digits_ckpt.model_checkpoint_path)
            else:
                print "Did not find digits trained conv model. Exiting."
                return

        if args.debug:
            # Add an op that raises assertion if any op in the graph returns inf or nan
            check_numerics = tf.add_check_numerics_ops()

        for step in xrange(start_step, MAX_STEPS):
            feed_dict = inputs.generate_feed_dict(train_data, images_pl, bboxes_pl)
            feed_dict[dropout_pl] = DROPOUT_KEEP_PROB
            
            if args.debug:
                _, loss_value, check = sess.run([train_step, loss, check_numerics], feed_dict=feed_dict)
                print step, check
            else:
                _, loss_value = sess.run([train_step, loss], feed_dict=feed_dict)

            if step % 50 == 0:
                summary = sess.run(merged_summaries, feed_dict=feed_dict)
                print "Step {}: Batch loss = {}".format(step, loss_value)
                summary_writer.add_summary(summary, step)
            
            if need_validation:
                if step != 0 and step % 500 == 0:
                    valid_avg_loss = test_valid_eval(sess, loss, valid_data, images_pl, bboxes_pl, dropout_pl)
                    print "Valid average loss = {}%".format(valid_avg_loss)
                    loss_summary = sess.run(valid_loss_summary, feed_dict={valid_loss_pl: valid_avg_loss})
                    summary_writer.add_summary(loss_summary, step)
            
            if step != 0 and step % 1000 == 0:
                test_avg_loss = test_valid_eval(sess, loss, test_data, images_pl, bboxes_pl, dropout_pl)
                print "Test average loss = {}%".format(test_loss)
                loss_summary = sess.run(test_loss_summary, feed_dict={test_loss_pl: test_loss})
                summary_writer.add_summary(loss_summary, step)

                saver_path = default_saver.save(sess, os.path.join(cp_dir, 'model.ckpt'), global_step=step)
                print "Model checkpoint created at {}".format(saver_path)


if __name__ == '__main__':
    tf.app.run()