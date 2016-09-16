import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

from threading import Thread
from collections import deque
from Queue import Queue
from time import sleep

from load_mnist import load_mnist

class MNISTMulti(object):
    def __init__(self, single_images, single_labels, random_state=101010, batch_size=128, dataset_size=50000, buffer_size=16, num_threads=8):
        self._single_images = single_images
        self._single_labels = single_labels
        assert self._single_images.shape[0] == self._single_labels.shape[0], "Mismatch in dataset"
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
        for i in range(self._max_threads):
            worker = Thread(target=self.fill_deck)
            worker.setDaemon(True)
            worker.start()

        for i in range(self._deck_size):
            self._task_queue.put(i)
        
        self._task_queue.join()

    class DeckRecord(object):
        pass

    def fill_deck(self):
        while True:
            _ = self._task_queue.get()
            images, labels = self.next_batch_queue()
            entry = self.DeckRecord()
            entry.images = images
            entry.labels = labels
            self._deck.append(entry)
            self._task_queue.task_done()

    def next_batch(self):
        while len(self._deck) == 0:
            print "load_mnist_multi num_threads {} or buffer_size {} too low".format(
                self._max_threads, self._deck_size)
            sleep(1)
        item = self._deck.popleft()
        self._task_queue.put(self._counter)
        return item.images, item.labels
    
    def next_batch_queue(self):
        # Note there could be leading zeros generated, but that should be OK

        images = np.ndarray([self._batch_size, 64, 64])
        labels = np.ndarray([self._batch_size], dtype='S5')

        for i in xrange(self._batch_size):
            # First fill in single digits into "full" image
            full_image = np.zeros([128, 192])
            
            # Generate somewhat real-world-like length of digits (street numbers)
            multi_digit_length = max(0, min(5, int(self._random.normal(2.5, 1.5))))

            if multi_digit_length == 0:
                final_image = np.zeros([64, 64])
                final_label = '_'
            else:
                # Pick random numbers from single digit dataset
                im_indices = self._random.randint(0, self._single_data_size, multi_digit_length)

                # Distort all digits of each multi digit set randomly
                resize = self._random.random_integers(22, 32, 2)
                resize = tuple(resize) # cv2.reize() requires tuple
                resized_ims = np.concatenate([cv2.resize(self._single_images[x], resize) for x in im_indices], axis=1)

                # Center the generated distorted images in "full" image
                row_index = full_image.shape[0] // 2 - resize[1] // 2   # Note: row and col indices in cv2 are switched
                col_index = full_image.shape[1] // 2 - resize[0] * multi_digit_length // 2
                full_image[row_index:(row_index+resized_ims.shape[0]), col_index:(col_index+resized_ims.shape[1])] = resized_ims

                # Randomly crop "full" image
                crop_row = self._random.randint(row_index - 16, row_index + 4)
                crop_col = self._random.randint(col_index - 16, col_index + 4)
                cropped_image = full_image[crop_row:crop_row+32+8,crop_col:crop_col+(32*len(im_indices))+8]

                # Resize to 64x64
                final_image = cv2.resize(cropped_image, (64,64))

                # Generate label
                final_label = ''.join(self._single_labels[im_indices].astype(np.character).tolist())

            # Fill in the return values
            images[i] = final_image
            labels[i] = final_label
        
        # Perform a very simple normailzation;
        # Should be OK for MNIST dataset.
        images = (images - 128) / 255.0

        self._counter += 1
        if (self._dataset_size // self._batch_size) <= self._counter:
            self._counter = 0
            self._random = np.random.RandomState(self._init_random_state)

        return images, labels

def main():
    mnist = load_mnist(normalized=False)

    mnist_multi = MNISTMulti(
        mnist.train_images, mnist.train_labels, batch_size=2, dataset_size=16, num_threads=2, buffer_size=8)

    for i in range(0, 32, 2):
        images, labels = mnist_multi.next_batch()
        plt.subplot(4,8,i+1)
        plt.imshow(images[0])
        plt.axis('off')
        plt.title(labels[0])
        plt.subplot(4,8,i+2)
        plt.imshow(images[1])
        plt.axis('off')
        plt.title(labels[1])

    plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))