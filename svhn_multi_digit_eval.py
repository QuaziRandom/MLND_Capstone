# Follows similar approach to MNIST multi-digit eval

import sys, os, hashlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import svhn_multi_digit as model
import svhn_multi_digit_input as inputs  

BATCH_SIZE = inputs.BATCH_SIZE
IMAGE_WIDTH = inputs.IMAGE_WIDTH
IMAGE_HEIGHT = inputs.IMAGE_HEIGHT
IMAGE_DEPTH = inputs.IMAGE_DEPTH
MAX_DIGITS = model.MAX_DIGITS

cp_dir = 'checkpoints/svhn_multi_digit/final_part_2'

def main(argv):
    _, _, test_data = inputs.load_svhn_datasets(valid_dataset_needed=False, train_dataset_needed=False)

    with tf.Graph().as_default() as graph:
        images_pl = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
        length_labels_pl = tf.placeholder(tf.int32, [BATCH_SIZE])
        digits_labels_pl = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_DIGITS])
        masks_pl = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_DIGITS])
        dropout_pl = tf.placeholder(tf.float32)

        conv_pool, num_conv_pool, last_conv_depth = model.conv_graph(images_pl)
        length_logits, digits_logits = model.fc_graph(conv_pool, num_conv_pool, last_conv_depth, masks_pl, dropout_pl)
        length_log_softmax = tf.nn.log_softmax(length_logits)
        digits_log_softmax = []
        for d in digits_logits:
            digits_log_softmax.append(tf.nn.log_softmax(d))

        saver = tf.train.Saver()

        sess = tf.Session()
        
        ckpt = tf.train.get_checkpoint_state(cp_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print "Valid checkpoint not found at {}".format(cp_dir)

        for _ in range(50): # 50 batches
            feed_dict = inputs.generate_feed_dict(test_data, images_pl, length_labels_pl, digits_labels_pl, masks_pl)
            feed_dict[dropout_pl] = 1.0
            
            length_log, digits_pred = sess.run([length_log_softmax, digits_log_softmax], feed_dict=feed_dict)
            length = np.argmax(length_log, axis=1)
            digits = []
            digits_max_log = []
            for d in digits_pred:
                digits.append(np.argmax(d, axis=1))
                digits_max_log.append(np.max(d, axis=1))
            digits = np.array(digits)
            digits_max_log = np.array(digits_max_log)
            total_confidence = length_log + np.cumsum(np.hstack([np.zeros([BATCH_SIZE,1]), digits_max_log.T]), axis=1)
            total_confidence_antilog = np.exp(total_confidence)
            
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

                C = str(np.round(total_confidence_antilog[i, length[i]] / np.sum(total_confidence_antilog[i]) * 100.0, 2))

                title = 'A:' + A + ' ' + 'P:' + P + ' ' + 'C:' + C + '%'

                if A != P:
                    fig_savepath = 'eval/svhn_multi_digit/negative_samples/'
                    plt.imshow(images[i] + 0.5)
                    plt.axis('off')
                    plt.title(title)
                    plt.savefig(fig_savepath + hashlib.sha1(images[i].copy(order='C')).hexdigest() + '.png')
                else:
                    fig_savepath = 'eval/scratch/'
                    # Not writing positive samples now. One run already over with many samples.           
                

if __name__ == '__main__':
    tf.app.run()