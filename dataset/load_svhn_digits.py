"""Produce batches of SVHN (digits) dataset.

From the SVHN full format dataset, apply some preprocessing
and produce train, valid and test datasets. From the provided
labels, find the biggest bounding box (or smallest rectangle)
enclosing all digits in an image, and crop to this bounding
box. Additionally, a slight expansion of the bounding box
maybe specified before cropping. Finally, the cropped image
is resized to fit a fixed square dimension. The labels
follow similar structure as the MNIST multi-digit dataset:
i.e., labels are categorized into length, individual digits 
and masks. When the dataset is seen as supporting some maximum
number of digits, masks help to identify digits that are not
present in the label. That is, a 2 digit label corresponds
to a mask of [1, 1, 0, 0, 0] if 5 maximum digits are supported.

Additionally supports multi-threading for speediness.
"""

import sys
import os
from scipy.io import loadmat
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
BBOX_EXPAND_RATIO = 0.3  # 30%

# Cropped-resized image parameters
FINAL_IMAGE_HEIGHT = 64
FINAL_IMAGE_WIDTH = 64
IMAGE_DEPTH = 3

# Maximum number of digits allowed in the label.
# This allows length of sequence in the label
# to be enumerated finitely: (0, 1, ... N, >N).
MAX_LEGAL_DIGITS = 5

# Directories containing uncompressed train, extra and test
# datasets (images) as directly obtained
TRAIN_DIR = os.path.join(os.path.dirname(__file__), 'svhn/train/')
EXTRA_DIR = os.path.join(os.path.dirname(__file__), 'svhn/extra/')
TEST_DIR = os.path.join(os.path.dirname(__file__), 'svhn/test/')

# The label (mat) file that is converted to a MATLAB v7
# version as opposed to the orignal v7.3 vesion
# This should be available in the corresponding
# directories as above
LABEL_MAT_FILE = 'digitStructv7.mat'
TRAIN_LABEL_MAT = os.path.join(TRAIN_DIR, LABEL_MAT_FILE)
EXTRA_LABEL_MAT = os.path.join(EXTRA_DIR, LABEL_MAT_FILE)
TEST_LABEL_MAT = os.path.join(TEST_DIR, LABEL_MAT_FILE)

# Considering train and extra datasets as one bunch, specify
# the split for valid and train datasets from this bunch
VALID_TO_TRAIN_EXTRA_RATIO = 0.05                           # 5%
TRAIN_TO_TRAIN_EXTRA_RATIO = 1 - VALID_TO_TRAIN_EXTRA_RATIO  # 95%

# This constant limits the number of train images being loaded.
# For some reasons unknown, tensorflow messes up some variables
# if all (the last bits) of the training data is considered.
# So consider almost all images from train dataset except the
# last few thousand
TRAIN_DATASET_LIMIT = 25000

# Extra dataset limit to be used in part 2. Refer to
# svhn_multi_digit_train.py for more details about training phases.
EXTRA_DATASET_LIMIT = 5000


class SVHNDigits(object):
    """Encapsulates SVHN digits dataset loader."""

    def __init__(self, type_data, random_state=101010, batch_size=128, buffer_size=128, num_threads=8):
        """Constructor for SVHNDigits.
        
        Instantiate object with 
            Type specifying 'train', 'valid' or 'test' dataset.
            Random state (seed) for mixing samples from train and extra datasets.
            Batch size indicating number of images and labels to be delivered per batch.
            Buffer size indicating number of batches to queue up.
            Number of worker threads filling up the queue.
        """
        # Load different label (mat) files based on the input type
        # If type is train or valid, load both train and extra labels
        # Else if, type is test, load only test labels
        if type_data == 'train' or type_data == 'valid':
            train_mat = loadmat(TRAIN_LABEL_MAT)['digitStruct'][
                :, :TRAIN_DATASET_LIMIT]
            extra_mat = loadmat(EXTRA_LABEL_MAT)['digitStruct'] #[
                #:, :EXTRA_DATASET_LIMIT] # part 2
            train_portion = int(train_mat['name'][0].shape[
                                0] * TRAIN_TO_TRAIN_EXTRA_RATIO)
            extra_portion = int(extra_mat['name'][0].shape[
                                0] * TRAIN_TO_TRAIN_EXTRA_RATIO)
            total_train_size = train_portion + extra_portion
            # Perform the appropriate split for train or valid datasets
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
                self._dataset_size = (train_mat['name'][0].shape[
                                      0] + extra_mat['name'][0].shape[0]) - total_train_size
            self._train_index = 0
            self._extra_index = 0
            self._train_to_whole_ratio = float(self._train_names.shape[
                                               0]) / float(self._extra_names.shape[0] + self._train_names.shape[0])
            self._train_extra_random_state = 1618
            self._train_extra_random = np.random.RandomState(
                self._train_extra_random_state)
            self._working_mode = 1  # Sample from both train and extra
        elif type_data == 'test':
            test_mat = loadmat(TEST_LABEL_MAT)['digitStruct']
            self._test_names = test_mat['name'][0]
            self._test_bboxes = test_mat['bbox'][0]
            self._dataset_size = test_mat['name'][0].shape[0]
            self._test_index = 0
            self._working_mode = 0  # Sample only from test data
        else:
            assert 'Invalid dataset type'

        self._type_data = type_data
        self._distortions_random_state = random_state
        self._distortions_random = np.random.RandomState(
            self._distortions_random_state)
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
        print "Prefetching SVHN digits data for {}".format(self._type_data)
        while self._delivery_queue.qsize() < self._task_queue.qsize():  # Will be approximate size
            pass
        print "Prefetched SVHN digits data for {}".format(self._type_data)

    # Constants to help in mixing
    # train and extra datasets
    TRAIN_DATA_ENUM = 1
    EXTRA_DATA_ENUM = 2
    TEST_DATA_ENUM = 3

    # Sampler that simulates biased coin, to either sample
    # from train dataset, or extra dataset
    def sample_from_train_or_extra(self):
        return self.TRAIN_DATA_ENUM if self._train_extra_random.random_sample() < self._train_to_whole_ratio else self.EXTRA_DATA_ENUM

    # Producer thread
    # Produces appropriate batches of jobs to work on.
    # When 'train' or 'valid' type is specified at creation,
    # the producer samples for choice between train or
    # extra dataset, where the probability of picking
    # a sample is directly proportional to the ratio of
    # total number of samples available in each of the
    # train or extra datasets. The job is then filled
    # with information required by the worker to to
    # actually load images and label data.
    # In case of the 'test' type, the same as above applies
    # except that all samples are taken from the test dataset.
    def fill_queue(self):
        if self._working_mode == 1:  # Work on train and extra datasets
            while True:
                job = []
                # Gather relevant information for the next batch
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

                    # Roll over when reaching end of train dataset
                    if self._train_index == self._train_names.shape[0]:
                        self._train_index = 0

                    # Roll over when reaching end of extra dataset
                    if self._extra_index == self._extra_names.shape[0]:
                        self._extra_index = 0

                # Blocking request to put new job into task queue
                self._task_queue.put(job, block=True)
        else:
            while True:
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
            images, labels = self.next_batch_queue(job)

            # Bundle up and fill the delivery queue
            # with the latest batch
            entry = self.DeliverRecord()
            entry.images = images
            entry.labels = labels
            self._delivery_queue.put(entry)

            # Notify completion of task
            self._task_queue.task_done()

    def next_batch(self):
        """Provide the next batch of data from the delivery queue."""
        # Wait (block) if queue is empty
        while self._delivery_queue.empty():
            print "load_svhn_digits num_threads {} or buffer_size {} too low".format(
                self._max_threads, self._delivery_queue_size)
            sleep(1)

        # Get processed batch from queue and return
        item = self._delivery_queue.get()
        self._delivery_queue.task_done()
        return item.images, item.labels

    # Process the next batch
    def next_batch_queue(self, job):
        # Initialize arrays to store images and labels in
        images = np.ndarray(
            [self._batch_size, FINAL_IMAGE_HEIGHT, FINAL_IMAGE_WIDTH, IMAGE_DEPTH])
        labels = np.ndarray([self._batch_size], dtype=[('string', 'S5'), ('length', 'i4'), (
            'digits', 'i4', MAX_LEGAL_DIGITS), ('mask', 'i4', MAX_LEGAL_DIGITS)])

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

            # Obtain the co-ordinates of the box that
            # covers all the single digits
            crop_top = bbox[i][0]['top'].astype(int).min()
            crop_left = bbox[i][0]['left'].astype(int).min()
            crop_bottom = (bbox[i][0]['height'].astype(
                int) + bbox[i][0]['top'].astype(int)).max()
            crop_right = (bbox[i][0]['width'].astype(
                int) + bbox[i][0]['left'].astype(int)).max()

            # Calculate the height and width of the big bounding box
            # to aid in bounding box expansion
            big_box_height = crop_bottom - crop_top
            big_box_width = crop_right - crop_left

            # Expand the big bounding box by the 'expand ratio' specified
            # Also, limit the expansion at image borders
            crop_top = max(
                0, int(crop_top - big_box_height * BBOX_EXPAND_RATIO / 2))
            crop_bottom = min(img.shape[0], int(
                crop_bottom + big_box_height * BBOX_EXPAND_RATIO / 2))
            crop_left = max(
                0, int(crop_left - big_box_width * BBOX_EXPAND_RATIO / 2))
            crop_right = min(img.shape[1], int(
                crop_right + big_box_width * BBOX_EXPAND_RATIO / 2))

            # Crop the input image to the expanded big bounding box
            cropped_img = img[crop_top:crop_bottom, crop_left:crop_right]

            # At this time, random crop/distortions may be applied.
            # It was found however that there isn't a significant
            # performance improvement by applying this. So removed
            # the random crop/distortions to save on computation.

            # Resize cropped image to a constant square size
            final_img = cv2.resize(
                cropped_img, (FINAL_IMAGE_WIDTH, FINAL_IMAGE_HEIGHT))

            # Get length and digits label data
            label_digits_raw = bbox[i][0]['label'].astype(int)
            label_digits_raw[label_digits_raw == 10] = 0
            label_length = label_digits_raw.shape[0]

            # Populate labels differently when length is zero
            if label_length == 0:
                label_string = '_'
                label_digits = np.zeros(MAX_LEGAL_DIGITS)
                label_mask = np.zeros(MAX_LEGAL_DIGITS)
            else:
                label_string = ''.join(
                    label_digits_raw.astype(np.character).tolist())
                label_digits = np.zeros(MAX_LEGAL_DIGITS)
                label_mask = np.zeros(MAX_LEGAL_DIGITS)
                if label_length <= MAX_LEGAL_DIGITS:
                    label_digits[:label_length] = label_digits_raw
                    label_mask[:label_length] = 1

            # Fill in the return values
            images[count] = final_img
            labels[count] = (label_string, label_length,
                             label_digits, label_mask)

            # Increment counter in the current job
            count += 1

        # Perform a simple normalization
        images = (images / 255.0) - 0.5

        return images, labels

    def get_dataset_size(self):
        """Getter for dataset size."""
        return self._dataset_size

    def get_batch_size(self):
        """Getter for batch size."""
        return self._batch_size


def main():
    # Test if SVHNDigits works properly.
    # Invoking script from command-line triggers the test.
    svhn_digits = SVHNDigits('test', batch_size=2,
                             buffer_size=8, num_threads=2)

    for i in range(0, 32, 2):
        images, labels = svhn_digits.next_batch()
        plt.subplot(4, 8, i + 1)
        plt.imshow(images[0])
        plt.axis('off')
        plt.title(labels['string'][0])
        plt.subplot(4, 8, i + 2)
        plt.imshow(images[1])
        plt.axis('off')
        plt.title(labels['string'][1])

    plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))
