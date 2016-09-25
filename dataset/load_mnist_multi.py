"""From the MNIST dataset, generate multi-digit data.

Combine single digits from MNIST dataset by inducing various
orientations, scale and placements, into a robust synthetic 
multi-digit dataset generator. The dataset produces batches of
images and labels categorized into length, individual digits 
and masks. When the dataset is seen as supporting some maximum
number of digits, masks help to identify digits that are not
present in the label. That is, a 2 digit label corresponds
to a mask of [1, 1, 0, 0, 0] if 5 maximum digits are generated.

Additionally supports multi-threading for speediness.

NOTE: Unlike multi-digit scenarios such as in house numbers, this 
generates leading zeros. 
"""
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

from threading import Thread
from collections import deque
from Queue import Queue
from time import sleep

from load_mnist import load_mnist

## Constants
# Maximum number of digits to generate
MAX_GENERATE_DIGITS = 5
# Maximum number of digits allowed in the label.
# This allows length of sequence in the label
# to be enumerated finitely: (0, 1, ... N, >N).
MAX_LEGAL_DIGITS = MAX_GENERATE_DIGITS - 1


class MNISTMulti(object):
    """Encapsulates MNIST multi-digit generator."""

    def __init__(self, single_images, single_labels, random_state=101010, batch_size=128, dataset_size=50000, buffer_size=16, num_threads=8):
        """Constructor for MNISTMulti.
        
        Instantiate object with 
            Single digit MNIST dataset already loaded in memory.
            Random state (seed) to use for generating digits and distortions.
            Batch size indicating number of images and labels to be delivered per batch.
            Dataset size to limit the number of multi-digit 'combinations' produced.
            Buffer size indicating number of batches to queue up.
            Number of worker threads filling up the queue.
        """
        self._single_images = single_images
        self._single_labels = single_labels
        assert self._single_images.shape[
            0] == self._single_labels.shape[0], "Mismatch in dataset"
        self._single_data_size = single_labels.shape[0]
        self._init_random_state = random_state
        self._random = np.random.RandomState(self._init_random_state)
        self._batch_size = batch_size
        self._counter = 0
        self._dataset_size = dataset_size
        self._max_threads = num_threads
        self._deck_size = buffer_size
        self._deck = deque()
        self._task_queue = Queue()
        self.initialize_threads()

    def initialize_threads(self):
        # Start worker threads
        for i in range(self._max_threads):
            worker = Thread(target=self.fill_deck)
            worker.setDaemon(True)
            worker.start()

        # Queue up tasks for the worker threads for the first time
        for i in range(self._deck_size):
            self._task_queue.put(i)

        # Wait till workers finish the first 'set'
        self._task_queue.join()

    # Object to bundle generated batches of data
    class DeckRecord(object):
        pass

    # Worker thread
    def fill_deck(self):
        while True:
            # Blocking until task is available
            _ = self._task_queue.get()

            # Get next batch
            images, labels = self.next_batch_queue()

            # Bundle up and fill the 'deck' (deque)
            # with the latest batch
            entry = self.DeckRecord()
            entry.images = images
            entry.labels = labels
            self._deck.append(entry)

            # Notify completion of task
            self._task_queue.task_done()

    def next_batch(self):
        """Provide the next batch of data from the queue."""
        # Wait (block) if queue is empty
        while len(self._deck) == 0:
            print "load_mnist_multi num_threads {} or buffer_size {} too low".format(
                self._max_threads, self._deck_size)
            sleep(1)

        # Get oldest batch from deck
        item = self._deck.popleft()

        # Create task to process next batch in background
        self._task_queue.put(self._counter)

        # Return images and labels
        return item.images, item.labels

    # Generate a batch
    def next_batch_queue(self):
        # Initialize arrays to store images and labels in
        images = np.ndarray([self._batch_size, 64, 64])
        labels = np.ndarray([self._batch_size], dtype=[('string', 'S5'), ('length', 'i4'), (
            'digits', 'i4', MAX_LEGAL_DIGITS), ('mask', 'i4', MAX_LEGAL_DIGITS)])

        for i in xrange(self._batch_size):
            # First fill in single digits into "full" image
            full_image = np.zeros([128, 192])

            # Generate somewhat real-world-like length of digits (street numbers)
            # That is, the distribution of length of house numbers may not be uniform.
            # It is most likely to find 2 or 3 digit house numbers rather than
            # say 4 or 5.
            multi_digit_length = max(
                0, min(MAX_GENERATE_DIGITS, int(self._random.normal(2.5, 1.5))))

            # Populate labels differently when length is zero
            if multi_digit_length == 0:
                final_image = np.zeros([64, 64])
                label_string = '_'
                label_length = multi_digit_length
                label_digits = np.zeros(MAX_LEGAL_DIGITS)
                label_mask = np.zeros(MAX_LEGAL_DIGITS)
            else:
                # Pick random numbers from single digit dataset
                im_indices = self._random.randint(
                    0, self._single_data_size, multi_digit_length)

                # Distort all digits of each multi digit set randomly
                resize = self._random.random_integers(22, 32, 2)
                resize = tuple(resize)  # cv2.reize() requires tuple
                resized_ims = np.concatenate(
                    [cv2.resize(self._single_images[x], resize) for x in im_indices], axis=1)

                # Center the generated distorted images in "full" image
                # Note: row and col indices in cv2 are switched
                row_index = full_image.shape[0] // 2 - resize[1] // 2
                col_index = full_image.shape[
                    1] // 2 - resize[0] * multi_digit_length // 2
                full_image[row_index:(row_index + resized_ims.shape[0]),
                           col_index:(col_index + resized_ims.shape[1])] = resized_ims

                # Randomly crop "full" image
                crop_row = self._random.randint(row_index - 16, row_index + 4)
                crop_col = self._random.randint(col_index - 16, col_index + 4)
                cropped_image = full_image[
                    crop_row:crop_row + 32 + 8, crop_col:crop_col + (32 * len(im_indices)) + 8]

                # Resize to 64x64
                final_image = cv2.resize(cropped_image, (64, 64))

                # Generate label data
                label_string = ''.join(self._single_labels[
                                       im_indices].astype(np.character).tolist())
                label_length = multi_digit_length
                label_digits = np.zeros(MAX_LEGAL_DIGITS)
                label_mask = np.zeros(MAX_LEGAL_DIGITS)
                if label_length <= MAX_LEGAL_DIGITS:
                    label_digits[:label_length] = self._single_labels[
                        im_indices]
                    label_mask[:label_length] = 1

            # Fill in the return values
            images[i] = final_image
            labels[i] = (label_string, label_length, label_digits, label_mask)

        # Perform a very simple normalization
        images = (images - 128) / 255.0

        # Keep track number of generated images. If configured dataset size
        # is reached, start over by re-seeding random generator with inital
        # value
        self._counter += 1
        if (self._dataset_size // self._batch_size) <= self._counter:
            self._counter = 0
            self._random = np.random.RandomState(self._init_random_state)

        return images, labels

    def get_dataset_size(self):
        """Getter for dataset size."""
        return self._dataset_size

    def get_batch_size(self):
        """Getter for batch size."""
        return self._batch_size


def main():
    # Test if datasets are loaded properly.
    # Invoking script from command-line triggers the test.

    mnist = load_mnist(normalized=False)

    mnist_multi = MNISTMulti(
        mnist.train_images, mnist.train_labels, batch_size=2, dataset_size=16, num_threads=2, buffer_size=8)

    for i in range(0, 32, 2):
        images, labels = mnist_multi.next_batch()
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
