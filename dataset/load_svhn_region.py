"""Produce batches of SVHN (region) dataset.

From the SVHN full format dataset, apply some preprocessing
and produce train, valid and test 'region' datasets. 
From the provided labels, find the biggest bounding box 
(or smallest rectangle) enclosing all digits in an image.
After scaling the input images and corresponding big bounding
box to a fixed size, return the resized images and bounding 
boxes as data and labels respectively. A slight expansion of 
the bounding box maybe specified if needed.

NOTE: Unlike SVHN digits dataset loader, this 'region' loader
does not input SVHN extra dataset, as the extra dataset contains
very simple and non-challenging scenes.

Additionally supports multi-threading for speediness.
"""
import sys
import os
from scipy.io import loadmat
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from threading import Thread
from collections import deque
from Queue import Queue
from time import sleep

## Constants
# Bounding box expansion ratio
# The height and width of the
# big bounding box found from
# the SVHN labels are expanded
# by (1 + BBOX_EXPAND_RATIO)
BBOX_EXPAND_RATIO = 0.0  # No expansion

# Directories containing uncompressed train and test
# datasets (images) as directly obtained
TRAIN_DIR = os.path.join(os.path.dirname(__file__), 'svhn/train/')
TEST_DIR = os.path.join(os.path.dirname(__file__), 'svhn/test/')

# The label (mat) file that is converted to a MATLAB v7
# version as opposed to the orignal v7.3 vesion
# This should be available in the corresponding
# directories as above
LABEL_MAT_FILE = 'digitStructv7.mat'
TRAIN_LABEL_MAT = os.path.join(TRAIN_DIR, LABEL_MAT_FILE)
TEST_LABEL_MAT = os.path.join(TEST_DIR, LABEL_MAT_FILE)

# Resized image parameters
FINAL_IMAGE_HEIGHT = 80
FINAL_IMAGE_WIDTH = 160
IMAGE_DEPTH = 3

# This constant limits the number of train images being loaded.
# For some reasons unknown, tensorflow messes up some variables
# if all (the last bits) of the training data is considered.
# So consider almost all images from train dataset except the
# last few thousand
TRAIN_DATASET_LIMIT = 25000

# Considering the SVHN train dataset as a whole, specify
# the split for validation data in this group.
VALID_RATIO = 0.1              # 10%
TRAIN_RATIO = 1 - VALID_RATIO  # 90%


class SVHNRegion(object):
    """Encapsulates SVHN region dataset loader."""

    def __init__(self, type_data, batch_size=128, buffer_size=32, num_threads=4):
        """Constructor for SVHNRegion.
        
        Instantiate object with 
            Type specifying 'train', 'valid' or 'test' dataset.
            Batch size indicating number of images and labels to be delivered per batch.
            Buffer size indicating number of batches to queue up.
            Number of worker threads filling up the queue.
        """
        # Load different label (mat) files based on the input type
        # If type is train or valid, load train labels
        # Else, load test labels
        if type_data == 'train' or type_data == 'valid':
            train_mat = loadmat(TRAIN_LABEL_MAT)['digitStruct'][
                :, :TRAIN_DATASET_LIMIT]
            train_portion = int(train_mat['name'][0].shape[0] * TRAIN_RATIO)
            # Perform the appropriate split for train or valid datasets
            if type_data == 'train':
                self._train_names = train_mat['name'][0][:train_portion]
                self._train_bboxes = train_mat['bbox'][0][:train_portion]
                self._dataset_size = train_portion
            else:
                self._train_names = train_mat['name'][0][train_portion:]
                self._train_bboxes = train_mat['bbox'][0][train_portion:]
                self._dataset_size = train_mat[
                    'name'][0].shape[0] - train_portion
            self._train_index = 0
            self._working_mode = 1  # For train and valid
        elif type_data == 'test':
            test_mat = loadmat(TEST_LABEL_MAT)['digitStruct']
            self._test_names = test_mat['name'][0]
            self._test_bboxes = test_mat['bbox'][0]
            self._dataset_size = test_mat['name'][0].shape[0]
            self._test_index = 0
            self._working_mode = 0  # For only test
        else:
            assert 'Invalid dataset type'

        self._type_data = type_data
        self._batch_size = batch_size

        self._max_threads = num_threads
        self._delivery_queue_size = buffer_size + 2 * num_threads  # Some safety margin
        self._delivery_queue = Queue(self._delivery_queue_size)
        self._task_queue = Queue(buffer_size)
        self.initialize_threads()

    def initialize_threads(self):
        # Start worker threads
        for i in range(self._max_threads):
            worker = Thread(target=self.fill_delivery_queue)
            worker.setDaemon(True)
            worker.start()

        # Start the producer thread
        producer = Thread(target=self.fill_queue)
        producer.setDaemon(True)
        producer.start()

        # Prefetch and block for the first time
        print "Prefetching SVHN region data for {}".format(self._type_data)
        while self._delivery_queue.qsize() < self._task_queue.qsize():  # Will be approximate size
            pass
        print "Prefetched SVHN region data for {}".format(self._type_data)

    # Producer thread
    # Produces appropriate batches of jobs to work on.
    def fill_queue(self):
        if self._working_mode == 1:  # For train and valid dataset
            while True:
                job = []
                # Gather relevant information for the next batch
                for _ in xrange(self._batch_size):
                    job.append((self._train_index, self._train_names,
                                self._train_bboxes, TRAIN_DIR))
                    self._train_index += 1

                    # Roll over when reaching end of train dataset
                    if self._train_index == self._train_names.shape[0]:
                        self._train_index = 0

                # Blocking request to put new job into task queue
                self._task_queue.put(job, block=True)
        else:
            while True:             # For test dataset
                job = []
                # Gather relevant information for the next batch
                for _ in xrange(self._batch_size):
                    job.append((self._test_index, self._test_names,
                                self._test_bboxes, TEST_DIR))
                    self._test_index += 1

                    # Roll over when reaching end of test dataset
                    if self._test_index == self._test_names.shape[0]:
                        self._test_index = 0

                # Blocking request to put new job into task queue
                self._task_queue.put(job, block=True)

    # Object to bundle a batch of data
    class DeliverRecord(object):
        pass

    # Worker thread
    def fill_delivery_queue(self):
        while True:
            # Block until task is available
            job = self._task_queue.get(block=True)

            # Get next batch
            images, bboxes = self.next_batch_queue(job)

            # Bundle up and fill the delivery queue
            # with the latest batch
            entry = self.DeliverRecord()
            entry.images = images
            entry.bboxes = bboxes
            self._delivery_queue.put(entry)

            # Notify completion of task
            self._task_queue.task_done()

    # Process the next batch
    def next_batch(self):
        """Provide the next batch of data from the delivery queue."""
        # Wait (block) if queue is empty
        while self._delivery_queue.empty():
            print "load_svhn_region num_threads {} or buffer_size {} too low".format(
                self._max_threads, self._delivery_queue_size)
            sleep(1)

        # Get processed batch from queue and return
        item = self._delivery_queue.get()
        self._delivery_queue.task_done()
        return item.images, item.bboxes

    def next_batch_queue(self, job):
        # Initialize arrays to store images and labels in
        images = np.ndarray(
            [self._batch_size, FINAL_IMAGE_HEIGHT, FINAL_IMAGE_WIDTH, IMAGE_DEPTH])
        bboxes = np.ndarray([self._batch_size, 4])

        # Counter to fill in the arrays sequentially
        count = 0

        # Process for each record in the job
        for i, name, bbox, wdir in job:
            # Read the image file specified in the job
            # NOTE: There is no error handling when a provided
            # filename can't be read either because of file not
            # existing or because it is corrupt, since defining
            # error reaction is hard. Instead the object will
            # crash on performing operations (down) on NoneType.
            filename = os.path.join(wdir, name[i][0])
            img = cv2.imread(filename)

            # Get the encompassing bounding box from mat file
            top = bbox[i][0]['top'].astype(int).min()
            left = bbox[i][0]['left'].astype(int).min()
            bottom = (bbox[i][0]['height'].astype(int) +
                      bbox[i][0]['top'].astype(int)).max()
            right = (bbox[i][0]['width'].astype(int) +
                     bbox[i][0]['left'].astype(int)).max()

            # Extract original bounding box height and width
            box_height = bottom - top
            box_width = right - left

            # Expand the bounding box to the specified margin
            top = max(0, int(top - box_height * BBOX_EXPAND_RATIO / 2))
            height = min(img.shape[0], int(
                bottom + box_height * BBOX_EXPAND_RATIO / 2)) - top
            left = max(0, int(left - box_width * BBOX_EXPAND_RATIO / 2))
            width = min(img.shape[1], int(
                right + box_width * BBOX_EXPAND_RATIO / 2)) - left

            # Resize bounding box to conform to final image
            # NOTE: It might be downscale or upscale
            vertical_scaling = img.shape[0] / float(FINAL_IMAGE_HEIGHT)
            horizontal_scaling = img.shape[1] / float(FINAL_IMAGE_WIDTH)

            # Finalize bounding box
            bbox_top = top // vertical_scaling
            bbox_left = left // horizontal_scaling
            bbox_height = height // vertical_scaling
            bbox_width = width // horizontal_scaling

            # Resize input image
            resized_img = cv2.resize(
                img, (FINAL_IMAGE_WIDTH, FINAL_IMAGE_HEIGHT))

            # Fill in the return values
            images[count] = resized_img
            bboxes[count] = (bbox_top, bbox_left, bbox_height, bbox_width)

            # Increment counter in the current job
            count += 1

        # Perform a simple normalization
        images = (images / 255.0) - 0.5

        return images, bboxes

    def get_dataset_size(self):
        """Getter for dataset size."""
        return self._dataset_size

    def get_batch_size(self):
        """Getter for batch size."""
        return self._batch_size


def main():
    # Test if SVHNRegion works properly.
    # Invoking script from command-line triggers the test.
    svhn_region = SVHNRegion('train', batch_size=2,
                             buffer_size=8, num_threads=2)

    for i in range(0, 32, 2):
        images, bboxes = svhn_region.next_batch()
        ax = plt.subplot(4, 8, i + 1)
        plt.imshow(images[0])
        plt.axis('off')
        bbox_left, bbox_top, bbox_width, bbox_height = bboxes[
            0, 1], bboxes[0, 0], bboxes[0, 3], bboxes[0, 2]
        ax.add_patch(patches.Rectangle((bbox_left, bbox_top),
                                       bbox_width, bbox_height, fill=False))
        ax = plt.subplot(4, 8, i + 2)
        plt.imshow(images[1])
        plt.axis('off')
        bbox_left, bbox_top, bbox_width, bbox_height = bboxes[
            1, 1], bboxes[1, 0], bboxes[1, 3], bboxes[1, 2]
        ax.add_patch(patches.Rectangle((bbox_left, bbox_top),
                                       bbox_width, bbox_height, fill=False))

    plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))
