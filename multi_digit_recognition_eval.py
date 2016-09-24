import os, hashlib
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

import svhn_multi_digit as digits_model
import svhn_region as region_model

from dataset.load_svhn_eval import SVHNEval

region_model_vars_clues = ('hidden', 'bbox')
digits_model_vars_clues = ('hidden', 'readout')
conv_vars_clues = ('conv')

region_model_dir = 'saved_models/svhn_region'
digits_model_dir = 'saved_models/svhn_multi_digit/'

BIG_IMAGE_WIDTH = 160
BIG_IMAGE_HEIGHT = 80
SMALL_IMAGE_WIDTH = 64
SMALL_IMAGE_HEIGHT = 64
IMAGE_DEPTH = 3
BATCH_SIZE = 32
MAX_DIGITS = 5

BBOX_EXPAND_RATIO = 0.3

def process_length_digits_logits(length_logit, digits_logit):
    pass

def main(argv):
    svhn_eval = SVHNEval(BATCH_SIZE)

    with tf.Graph().as_default() as graph:
        images_pl = tf.placeholder(tf.float32, [BATCH_SIZE, None, None, IMAGE_DEPTH])
        masks = tf.constant(1.0, shape=[BATCH_SIZE, MAX_DIGITS])
        dropout = tf.constant(1.0)

        conv_pool, num_conv_pool, last_conv_depth = digits_model.conv_graph(images_pl)
        
        with tf.name_scope('region'):
            bbox_tensor = region_model.fc_graph(conv_pool, num_conv_pool, last_conv_depth, dropout)

        with tf.name_scope('digits'):    
            length_tensor, digits_tensor = digits_model.fc_graph(conv_pool, num_conv_pool, last_conv_depth, masks, dropout)
            length_log_softmax = tf.nn.log_softmax(length_tensor)
            digits_log_softmax = []
            for d in digits_tensor:
                digits_log_softmax.append(tf.nn.log_softmax(d))

        all_graph_vars = tf.all_variables()

        digits_model_vars = {}
        region_model_vars = {}
        conv_model_vars = {}
        for var in all_graph_vars:           
            if var.name.replace('region/','').startswith(region_model_vars_clues):
                region_model_vars[var.op.name.replace('region/','')] = var
            elif var.name.replace('digits/','').startswith(digits_model_vars_clues):
                digits_model_vars[var.op.name.replace('digits/','')] = var
            elif var.name.startswith(conv_vars_clues):
                digits_model_vars[var.op.name] = var                
        
        region_model_restorer = tf.train.Saver(region_model_vars)          
        digits_model_restorer = tf.train.Saver(digits_model_vars)

        sess = tf.Session()

        digits_ckpt = tf.train.get_checkpoint_state(digits_model_dir)
        if digits_ckpt and digits_ckpt.model_checkpoint_path:            
            digits_model_restorer.restore(sess, digits_ckpt.model_checkpoint_path)
            print "Digits model restored from {}".format(digits_ckpt.model_checkpoint_path)
        else:
            print "Did not find digits trained model. Exiting."
            return

        regions_ckpt = tf.train.get_checkpoint_state(region_model_dir)
        if regions_ckpt and regions_ckpt.model_checkpoint_path:            
            region_model_restorer.restore(sess, regions_ckpt.model_checkpoint_path)
            print "Region model restored from {}".format(regions_ckpt.model_checkpoint_path)
        else:
            print "Did not find regions trained model. Exiting."
            return

        sess.run(tf.assert_variables_initialized())

        for _ in range(svhn_eval.get_dataset_size() // BATCH_SIZE):
            images, bboxes, labels = svhn_eval.next_batch()
            length_labels = labels['length']
            digits_labels = labels['digits']

            region_feed_dict = {
                images_pl: images
            }

            bbox_pred = sess.run(bbox_tensor, feed_dict=region_feed_dict)

            repackaged_images = np.ndarray([BATCH_SIZE, SMALL_IMAGE_HEIGHT, SMALL_IMAGE_WIDTH, IMAGE_DEPTH])
            for i in range(BATCH_SIZE):                
                bbox_left, bbox_top, bbox_width, bbox_height = bbox_pred[i, 1], bbox_pred[i, 0], bbox_pred[i, 3], bbox_pred[i, 2]
                
                current_image = images[i]
                crop_top = max(0, int(bbox_top - bbox_height * BBOX_EXPAND_RATIO/2))
                crop_bottom = min(BIG_IMAGE_HEIGHT, int((bbox_top + bbox_height) + bbox_height * BBOX_EXPAND_RATIO/2))
                crop_left = max(0, int(bbox_left - bbox_width * BBOX_EXPAND_RATIO/2))
                crop_right = min(BIG_IMAGE_WIDTH, int((bbox_left + bbox_width) + bbox_width * BBOX_EXPAND_RATIO/2))

                cropped_image = current_image[crop_top:crop_bottom, crop_left:crop_right]

                repackaged_images[i] = cv2.resize(cropped_image, (SMALL_IMAGE_WIDTH, SMALL_IMAGE_HEIGHT))

            digits_feed_dict = {
                images_pl: repackaged_images
            }

            length_pred, digits_pred = sess.run([length_log_softmax, digits_log_softmax], feed_dict=digits_feed_dict)

            length = np.argmax(length_pred, axis=1)
            digits = []
            digits_max_log = []
            for d in digits_pred:
                digits.append(np.argmax(d, axis=1))
                digits_max_log.append(np.max(d, axis=1))
            digits = np.array(digits)
            digits_max_log = np.array(digits_max_log)
            total_confidence = length_pred + np.cumsum(np.hstack([np.zeros([BATCH_SIZE,1]), digits_max_log.T]), axis=1)
            total_confidence_antilog = np.exp(total_confidence)

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

                bbox_left, bbox_top, bbox_width, bbox_height = bbox_pred[i, 1], bbox_pred[i, 0], bbox_pred[i, 3], bbox_pred[i, 2]
                ax = plt.axes()
                current_image = images[i]
                plt.imshow(current_image + 0.5)
                ax.add_patch(patches.Rectangle((bbox_left, bbox_top), bbox_width, bbox_height, fill=False))
                plt.axis('off')
                plt.title(title)

                if A != P:
                    fig_savepath = 'eval/scratch/negative/'
                else:
                    fig_savepath = 'eval/scratch/positive/'

                plt.savefig(fig_savepath + hashlib.sha1(current_image).hexdigest() + '.png')
                plt.close()


if __name__ == '__main__':
    tf.app.run()