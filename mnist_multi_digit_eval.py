import sys, os, hashlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import mnist_multi_digit as model
import mnist_multi_digit_input as inputs

# NOTE: The eval script is neither efficient or elegant. Most of efficient eval-ing 
# is already done during training, and so doesn't necessitate elegance too. This
# script writes positive/negative samples from evaluation of a given train checkpoint.   

BATCH_SIZE = inputs.BATCH_SIZE
IMAGE_WIDTH = inputs.IMAGE_WIDTH
IMAGE_HEIGHT = inputs.IMAGE_HEIGHT
MAX_DIGITS = model.MAX_DIGITS

cp_dir = 'checkpoints/mnist_multi_digit/exponential_lr/conv3_1e-1_conv4_5e-2/lrinit_8e-4_lrdecay_9e-1'

def main(argv):
    _, _, test_data = inputs.create_multi_digit_datasets()

    with tf.Graph().as_default() as graph:
        images_pl = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        length_labels_pl = tf.placeholder(tf.int32, [BATCH_SIZE])
        digits_labels_pl = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_DIGITS])
        masks_pl = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_DIGITS])
        dropout_pl = tf.placeholder(tf.float32)

        conv_pool, num_conv_pool, last_conv_depth = model.conv_graph(images_pl)
        length_logits, digits_logits = model.fc_graph(conv_pool, num_conv_pool, last_conv_depth, masks_pl, dropout_pl)

        saver = tf.train.Saver()

        sess = tf.Session()
        
        ckpt = tf.train.get_checkpoint_state(cp_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print "Valid checkpoint not found at {}".format(cp_dir)

        for _ in range(100): # 100 batches
            feed_dict = inputs.generate_feed_dict(test_data, images_pl, length_labels_pl, digits_labels_pl, masks_pl)
            feed_dict[dropout_pl] = 1.0
            
            length_pred, digits_pred = sess.run([length_logits, digits_logits], feed_dict=feed_dict)
            length = np.argmax(length_pred, axis=1)
            digits = []
            for d in digits_pred:
                digits.append(np.argmax(d, axis=1))
            digits = np.array(digits)
            
            images = feed_dict[images_pl]
            length_labels = feed_dict[length_labels_pl]
            digits_labels = feed_dict[digits_labels_pl]

            for i in range(BATCH_SIZE):         

                if length_labels[i] == 0:
                    A = str(length_labels[i]) + ' length'
                elif length_labels[i] > MAX_DIGITS:
                    A = '>' + str(MAX_DIGITS) + ' length'
                else:
                    A = str(length_labels[i]) + '|' + ''.join(digits_labels[i, :length_labels[i]].astype(np.character).tolist())
                
                if length[i] == 0:
                    P = str(length[i]) + ' length'
                elif length[i] > MAX_DIGITS:
                    P = '>' + str(MAX_DIGITS) + ' length'
                else:
                    P = str(length[i]) + '|' + ''.join(digits[:length[i], i].astype(np.character).tolist())

                title = 'A:' + A + ' ' + 'P:' + P

                if A != P:
                    fig_savepath = 'eval/mnist_multi_digit/negative_samples/'
                    plt.imshow(np.reshape(images[i], [IMAGE_HEIGHT, IMAGE_WIDTH]))
                    plt.axis('off')
                    plt.title(title)
                    plt.savefig(fig_savepath + hashlib.sha1(images[i].copy(order='C')).hexdigest() + '.png')
                else:
                    fig_savepath = 'eval/scratch/'
                    # Not writing positive samples now. One run already over with many samples.           
                

if __name__ == '__main__':
    tf.app.run()