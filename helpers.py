import tensorflow as tf
import argparse

def variable_summary(name, var):
    with tf.name_scope('variable_summary'):
        mean = tf.reduce_mean(var)
        with tf.name_scope('std_dev'):
            std = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('mean/' + name, mean)
        tf.scalar_summary('std/' + name, std)
        tf.histogram_summary('variable/' + name, var)

def activation_summary(name, var):
    with tf.name_scope('activation_summary'):
        tf.scalar_summary('sparsity/' + name, tf.nn.zero_fraction(var))
        tf.histogram_summary('activation/' + name, var)

def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', help='name for this particular run', nargs='?', default='')
    parser.add_argument('-l', '--logdir', help='relative path to log directory')
    parser.add_argument('-c', '--cpdir', help='relative path to checkpoint directory')
    parser.add_argument('-r', '--reset-logdir', help='reset (erase) all logs at the log directory', action='store_true')
    parser.add_argument('-R', '--reset-cpdir', help='reset (erase) all checkpoints at the checkpoints directory', action='store_true')
    args = parser.parse_args()
    return args