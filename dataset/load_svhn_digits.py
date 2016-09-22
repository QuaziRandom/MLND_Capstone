# TODO: Documentation on this is pretty low, and it looks the 
# ones that most needs it. Fill this up later.

import sys, os
from scipy.io import loadmat
import numpy as np
import cv2
import matplotlib.pyplot as plt

from threading import Thread
from collections import deque
from Queue import Queue
from time import sleep

BBOX_EXPAND_RATIO = 0.3 # 30%

PREIMG_RESIZE_HEIGHT = 76
PREIMG_RESIZE_WIDTH = 76
FINAL_IMAGE_HEIGHT = 64
FINAL_IMAGE_WIDTH = 64
IMAGE_DEPTH = 3

MAX_LEGAL_DIGITS = 5

TRAIN_DIR = os.path.join(os.path.dirname(__file__),'svhn/train/')
EXTRA_DIR = os.path.join(os.path.dirname(__file__),'svhn/extra/')
TEST_DIR = os.path.join(os.path.dirname(__file__),'svhn/test/')
LABEL_MAT_FILE = 'digitStructv7.mat'

TRAIN_LABEL_MAT = os.path.join(TRAIN_DIR, LABEL_MAT_FILE)
EXTRA_LABEL_MAT = os.path.join(EXTRA_DIR, LABEL_MAT_FILE)
TEST_LABEL_MAT = os.path.join(TEST_DIR, LABEL_MAT_FILE)

VALID_TO_TRAIN_EXTRA_RATIO = 0.05                           # 5%
TRAIN_TO_TRAIN_EXTRA_RATIO = 1 - VALID_TO_TRAIN_EXTRA_RATIO # 95%

# This constant limits the number of train images being loaded.
# For some reasons unknown tensorflow messes up some variables
# if all (the last bits) of the training data is considered.
TRAIN_DATASET_LIMIT = 25000

# Extra dataset limit to be used in part 2. Refer to 
# svhn_multi_digit_train.py for more details about training phases.
EXTRA_DATASET_LIMIT = 5000 

class SVHNDigits(object):
    def __init__(self, type_data, random_state=101010, batch_size=128, buffer_size=128, num_threads=8):
        if type_data == 'train' or type_data == 'valid':
            train_mat = loadmat(TRAIN_LABEL_MAT)['digitStruct'][:, :TRAIN_DATASET_LIMIT]
            extra_mat = loadmat(EXTRA_LABEL_MAT)['digitStruct'] # [:, :EXTRA_DATASET_LIMIT] # part 2
            train_portion = int(train_mat['name'][0].shape[0] * TRAIN_TO_TRAIN_EXTRA_RATIO)
            extra_portion = int(extra_mat['name'][0].shape[0] * TRAIN_TO_TRAIN_EXTRA_RATIO)
            total_train_size = train_portion + extra_portion
            if type_data == 'train':
                self._train_names = train_mat['name'][0][:train_portion]
                self._train_bboxes = train_mat['bbox'][0][:train_portion]
                self._extra_names = extra_mat['name'][0][:extra_portion]
                self._extra_bboxes = extra_mat['bbox'][0][:extra_portion]
                self._dataset_size = total_train_size
            else:
                self._train_names = train_mat['name'][0][train_portion:]
                self._train_bboxes = train_mat['bbox'][0][train_portion:]
                self._extra_names = extra_mat['name'][0][extra_portion:]
                self._extra_bboxes = extra_mat['bbox'][0][extra_portion:]
                self._dataset_size = (train_mat['name'][0].shape[0] + extra_mat['name'][0].shape[0]) - total_train_size
            self._train_index = 0
            self._extra_index = 0
            self._train_to_whole_ratio = float(self._train_names.shape[0]) / float(self._extra_names.shape[0] + self._train_names.shape[0])
            self._train_extra_random_state = 1618
            self._train_extra_random = np.random.RandomState(self._train_extra_random_state)            
            self._working_mode = 1 # Sample from both train and extra
        elif type_data == 'test':
            test_mat = loadmat(TEST_LABEL_MAT)['digitStruct']
            self._test_names = test_mat['name'][0]
            self._test_bboxes = test_mat['bbox'][0]
            self._dataset_size = test_mat['name'][0].shape[0]
            self._test_index = 0
            self._working_mode = 0 # Sample only from test data
        else:
            assert 'Invalid dataset type'

        self._type_data = type_data
        self._distortions_random_state = random_state
        self._distortions_random = np.random.RandomState(self._distortions_random_state)
        self._batch_size = batch_size

        self._max_threads = num_threads
        self._delivery_queue_size = buffer_size + 2 * num_threads # Some safety margin
        self._delivery_queue = Queue(self._delivery_queue_size)
        self._task_queue = Queue(buffer_size)
        self.initialize_threads()

    def initialize_threads(self):
        for i in range(self._max_threads):
            worker = Thread(target=self.fill_delivery_queue)
            worker.setDaemon(True)
            worker.start()

        producer = Thread(target=self.fill_queue)
        producer.setDaemon(True)
        producer.start()

        # Prefetch and block for the first time
        print "Prefetching SVHN data for {}".format(self._type_data)
        while self._delivery_queue.qsize() < self._task_queue.qsize(): # Will be approximate size
            pass
        print "Prefetched SVHN data for {}".format(self._type_data)

    TRAIN_DATA_ENUM = 1
    EXTRA_DATA_ENUM = 2
    TEST_DATA_ENUM = 3

    def sample_from_train_or_extra(self):
        return self.TRAIN_DATA_ENUM if self._train_extra_random.random_sample() < self._train_to_whole_ratio else self.EXTRA_DATA_ENUM

    def fill_queue(self):
        if self._working_mode == 1: # Work on train and extra datasets
            while True:
                # Gather indices
                job = []
                for _ in xrange(self._batch_size):
                    t_or_e = self.sample_from_train_or_extra()
                    if t_or_e == self.TRAIN_DATA_ENUM:
                        index = self._train_index
                        names = self._train_names
                        bboxes = self._train_bboxes
                        work_dir = TRAIN_DIR
                        self._train_index += 1
                    else:
                        index = self._extra_index
                        names = self._extra_names
                        bboxes = self._extra_bboxes
                        work_dir = EXTRA_DIR
                        self._extra_index += 1

                    job.append((index, names, bboxes, work_dir))
                    
                    if self._train_index == self._train_names.shape[0]:
                        self._train_index = 0

                    if self._extra_index == self._extra_names.shape[0]:
                        self._extra_index = 0

                self._task_queue.put(job, block=True)
        else:
            while True:
                job = []
                for _ in xrange(self._batch_size):
                    job.append((self._test_index, self._test_names, self._test_bboxes, TEST_DIR))
                    self._test_index += 1
                    if self._test_index == self._test_names.shape[0]:
                        self._test_index = 0
                        
                self._task_queue.put(job, block=True)

    class DeckRecord(object):
        pass

    def fill_delivery_queue(self):
        while True:
            job = self._task_queue.get(block=True)
            images, labels = self.next_batch_queue(job)
            entry = self.DeckRecord()
            entry.images = images
            entry.labels = labels
            self._delivery_queue.put(entry)
            self._task_queue.task_done()

    def next_batch(self):
        while self._delivery_queue.empty():
            print "load_svhn_digits num_threads {} or buffer_size {} too low".format(
                self._max_threads, self._delivery_queue_size)
            sleep(1)
        item = self._delivery_queue.get()
        self._delivery_queue.task_done()
        return item.images, item.labels
    
    def next_batch_queue(self, job):        
        images = np.ndarray([self._batch_size, FINAL_IMAGE_HEIGHT, FINAL_IMAGE_WIDTH, IMAGE_DEPTH])
        labels = np.ndarray([self._batch_size], dtype=[
            ('string', 'S5'), ('length', 'i4'), ('digits', 'i4', MAX_LEGAL_DIGITS), ('mask', 'i4', MAX_LEGAL_DIGITS)])
        
        count = 0
        for i, name, bbox, wdir in job:
            filename = os.path.join(wdir, name[i][0])
            img = cv2.imread(filename)

            crop_top = bbox[i][0]['top'].astype(int).min()
            crop_left = bbox[i][0]['left'].astype(int).min()
            crop_bottom = (bbox[i][0]['height'].astype(int) + bbox[i][0]['top'].astype(int)).max()
            crop_right = (bbox[i][0]['width'].astype(int) +bbox[i][0]['left'].astype(int)).max()
            
            big_box_height = crop_bottom - crop_top
            big_box_width = crop_right - crop_left

            crop_top = max(0, int(crop_top - big_box_height * BBOX_EXPAND_RATIO/2))
            crop_bottom = min(img.shape[0], int(crop_bottom + big_box_height * BBOX_EXPAND_RATIO/2))
            crop_left = max(0, int(crop_left - big_box_width * BBOX_EXPAND_RATIO/2))
            crop_right = min(img.shape[1], int(crop_right + big_box_width * BBOX_EXPAND_RATIO/2))

            cropped_img = img[crop_top:crop_bottom, crop_left:crop_right]

            # Disabled random crop/distortion for now.            
            # resized_img = cv2.resize(cropped_img, (PREIMG_RESIZE_WIDTH, PREIMG_RESIZE_HEIGHT))

            # if self._working_mode == 1: # Test and valid mode
            #     random_crop_top = self._distortions_random.randint(0, PREIMG_RESIZE_HEIGHT - FINAL_IMAGE_HEIGHT)
            #     random_crop_left = self._distortions_random.randint(0, PREIMG_RESIZE_WIDTH - FINAL_IMAGE_WIDTH)
            # else: # For test data, crop only center
            #     random_crop_top = (PREIMG_RESIZE_HEIGHT - FINAL_IMAGE_HEIGHT) / 2
            #     random_crop_left = (PREIMG_RESIZE_WIDTH - FINAL_IMAGE_WIDTH) / 2
            # random_cropped_img = img[random_crop_top:FINAL_IMAGE_HEIGHT, random_crop_left:FINAL_IMAGE_WIDTH]

            final_img = cv2.resize(cropped_img, (FINAL_IMAGE_WIDTH, FINAL_IMAGE_HEIGHT))

            label_digits_raw = bbox[i][0]['label'].astype(int)
            label_digits_raw[label_digits_raw == 10] = 0
            label_length = label_digits_raw.shape[0]

            if label_length == 0:
                label_string = '_'
                label_digits = np.zeros(MAX_LEGAL_DIGITS)
                label_mask = np.zeros(MAX_LEGAL_DIGITS)
            else:
                # Generate label data
                label_string = ''.join(label_digits_raw.astype(np.character).tolist())
                label_digits = np.zeros(MAX_LEGAL_DIGITS)                
                label_mask = np.zeros(MAX_LEGAL_DIGITS)
                if label_length <= MAX_LEGAL_DIGITS:
                    label_digits[:label_length] = label_digits_raw
                    label_mask[:label_length] = 1

            # Fill in the return values
            images[count] = final_img
            labels[count] = (label_string, label_length, label_digits, label_mask)
            count += 1
        
        # Perform a simple normalization
        images = (images / 255.0) - 0.5

        return images, labels

    def get_dataset_size(self):
        return self._dataset_size
        
    def get_batch_size(self):
        return self._batch_size

def main():
    svhn_digits = SVHNDigits('test', batch_size=2, buffer_size=8, num_threads=2)

    for i in range(0, 32, 2):
        images, labels = svhn_digits.next_batch()
        plt.subplot(4,8,i+1)
        plt.imshow(images[0])
        plt.axis('off')
        plt.title(labels['string'][0])
        plt.subplot(4,8,i+2)
        plt.imshow(images[1])
        plt.axis('off')
        plt.title(labels['string'][1])

    plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))