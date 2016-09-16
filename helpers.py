import tensorflow as tf

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