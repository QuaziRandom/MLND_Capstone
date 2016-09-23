import sys, os
from scipy.io import loadmat
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from threading import Thread
from collections import deque
from Queue import Queue
from time import sleep

TRAIN_DIR = os.path.join(os.path.dirname(__file__),'svhn/train/')
EXTRA_DIR = os.path.join(os.path.dirname(__file__),'svhn/extra/')
TEST_DIR = os.path.join(os.path.dirname(__file__),'svhn/test/')
LABEL_MAT_FILE = 'digitStructv7.mat'

TRAIN_LABEL_MAT = os.path.join(TRAIN_DIR, LABEL_MAT_FILE)
EXTRA_LABEL_MAT = os.path.join(EXTRA_DIR, LABEL_MAT_FILE)
TEST_LABEL_MAT = os.path.join(TEST_DIR, LABEL_MAT_FILE)

BBOX_EXPAND_RATIO = 0.3 # 30%

FINAL_IMAGE_HEIGHT = 180
FINAL_IMAGE_WIDTH = 320
IMAGE_DEPTH = 3

# This constant limits the number of train images being loaded.
# For some reasons unknown tensorflow messes up some variables
# if all (the last bits) of the training data is considered.
TRAIN_DATASET_LIMIT = 25000

VALID_RATIO = 0.1             # 10%
TRAIN_RATIO = 1 - VALID_RATIO # 90%

class SVHNRegion(object):
    def __init__(self, type_data, random_state=101010, batch_size=128, buffer_size=64, num_threads=4):
        if type_data == 'train' or type_data == 'valid':
            train_mat = loadmat(TRAIN_LABEL_MAT)['digitStruct'][:, :TRAIN_DATASET_LIMIT]
            train_portion = int(train_mat['name'][0].shape[0] * TRAIN_RATIO)
            if type_data == 'train':
                self._train_names = train_mat['name'][0][:train_portion]
                self._train_bboxes = train_mat['bbox'][0][:train_portion]
                self._dataset_size = train_portion
            else:
                self._train_names = train_mat['name'][0][train_portion:]
                self._train_bboxes = train_mat['bbox'][0][train_portion:]
                self._dataset_size = train_mat['name'][0].shape[0] - train_portion
            self._train_index = 0
            self._train_random_state = 1618
            self._train_random = np.random.RandomState(self._train_random_state)            
            self._working_mode = 1 # For train and valid
        elif type_data == 'test':
            test_mat = loadmat(TEST_LABEL_MAT)['digitStruct']
            self._test_names = test_mat['name'][0]
            self._test_bboxes = test_mat['bbox'][0]
            self._dataset_size = test_mat['name'][0].shape[0]
            self._test_index = 0
            self._working_mode = 0 # For only test
        else:
            assert 'Invalid dataset type'

        self._type_data = type_data
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

    def fill_queue(self):
        if self._working_mode == 1: # For train and valid dataset
            while True:
                job = []
                for _ in xrange(self._batch_size):
                    job.append((self._train_index, self._train_names, self._train_bboxes, TRAIN_DIR))
                    self._train_index += 1
                    if self._train_index == self._train_names.shape[0]:
                        self._train_index = 0
                        
                self._task_queue.put(job, block=True)
        else:
            while True:             # For test dataset
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
            images, bboxes = self.next_batch_queue(job)
            entry = self.DeckRecord()
            entry.images = images
            entry.bboxes = bboxes
            self._delivery_queue.put(entry)
            self._task_queue.task_done()

    def next_batch(self):
        while self._delivery_queue.empty():
            print "load_svhn_region num_threads {} or buffer_size {} too low".format(
                self._max_threads, self._delivery_queue_size)
            sleep(1)
        item = self._delivery_queue.get()
        self._delivery_queue.task_done()
        return item.images, item.bboxes

    def next_batch_queue(self, job):        
        images = np.ndarray([self._batch_size, FINAL_IMAGE_HEIGHT, FINAL_IMAGE_WIDTH, IMAGE_DEPTH])
        bboxes = np.ndarray([self._batch_size, 4])
        
        count = 0
        for i, name, bbox, wdir in job:
            filename = os.path.join(wdir, name[i][0])
            img = cv2.imread(filename)

            # Get the encompassing bounding box from mat file
            top = bbox[i][0]['top'].astype(int).min()
            left = bbox[i][0]['left'].astype(int).min()
            bottom = (bbox[i][0]['height'].astype(int) + bbox[i][0]['top'].astype(int)).max()
            right = (bbox[i][0]['width'].astype(int) +bbox[i][0]['left'].astype(int)).max()
            
            # Extract original bounding box height and width
            box_height = bottom - top
            box_width = right - left

            # Expand the bounding box to the specified margin
            top = max(0, int(top - box_height * BBOX_EXPAND_RATIO/2))
            height = min(img.shape[0], int(bottom + box_height * BBOX_EXPAND_RATIO/2)) - top
            left = max(0, int(left - box_width * BBOX_EXPAND_RATIO/2))
            width = min(img.shape[1], int(right + box_width * BBOX_EXPAND_RATIO/2)) - left

            # Resize bounding box to conform to final image
            # Note: It might be downscale or upscale
            vertical_scaling = img.shape[0] / float(FINAL_IMAGE_HEIGHT)
            horizontal_scaling = img.shape[1] / float(FINAL_IMAGE_WIDTH)

            # Finalize bounding box
            bbox_top = top // vertical_scaling
            bbox_left = left // horizontal_scaling
            bbox_height = height // vertical_scaling
            bbox_width = width // horizontal_scaling

            # Resize input image
            resized_img = cv2.resize(img, (FINAL_IMAGE_WIDTH, FINAL_IMAGE_HEIGHT))

            # Fill in the return values
            images[count] = resized_img
            bboxes[count] = (bbox_top, bbox_left, bbox_height, bbox_width)
            count += 1
        
        # Perform a simple normalization
        images = (images / 255.0) - 0.5

        return images, bboxes

    def get_dataset_size(self):
        return self._dataset_size
        
    def get_batch_size(self):
        return self._batch_size


def main():
    svhn_region = SVHNRegion('train', batch_size=2, buffer_size=8, num_threads=2)

    for i in range(0, 32, 2):
        images, bboxes = svhn_region.next_batch()
        ax = plt.subplot(4,8,i+1)
        plt.imshow(images[0])
        plt.axis('off')
        bbox_left, bbox_top, bbox_width, bbox_height = bboxes[0, 1], bboxes[0, 0], bboxes[0, 3], bboxes[0, 2]
        ax.add_patch(patches.Rectangle((bbox_left, bbox_top), bbox_width, bbox_height, fill=False))
        ax = plt.subplot(4,8,i+2)
        plt.imshow(images[1])
        plt.axis('off')
        bbox_left, bbox_top, bbox_width, bbox_height = bboxes[1, 1], bboxes[1, 0], bboxes[1, 3], bboxes[1, 2]
        ax.add_patch(patches.Rectangle((bbox_left, bbox_top), bbox_width, bbox_height, fill=False))

    plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))