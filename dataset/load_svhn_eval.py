"""Produce batches of SVHN digits AND SVHN region test datasets.

Wrapping the SVHN digits and SVHN region dataset loaders, produce
consistent batches of data from both of the aforementioned loaders
and bunch them up into a single returned batch. That is the batch
produced pertains to same images from both loaders.

NOTE: This only wraps generating of combined dataset for test
data exclusively. It is also important to limit the number of
threads and buffer for the individual dataset loaders to 1 so
that batches do not get mixed up because of multi-threaded
processing. 
"""
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from load_svhn_digits import SVHNDigits
from load_svhn_region import SVHNRegion

class SVHNEval(object):
    """Encapsulates SVHN combined eval dataset loader."""
    def __init__(self, batch_size=32):
        """Constructor for SVHNEval.
        
        Instantiate object with 
            Batch size indicating number of images, labels and bounding boxes to be delivered per batch.
        """
        self._region_obj = SVHNRegion('test', batch_size=batch_size, buffer_size=1, num_threads=1)
        self._digits_obj = SVHNDigits('test', batch_size=batch_size, buffer_size=1, num_threads=1)

    def next_batch(self):
        """Get the combined next batch of data for eval."""
        region_images, region_bboxes = self._region_obj.next_batch()
        digits_images, digits_labels = self._digits_obj.next_batch()

        return region_images, region_bboxes, digits_labels
    
    def get_dataset_size(self):
        """Getter for dataset size."""
        return self._region_obj.get_dataset_size()

def main():
    # Test if SVHNEval works properly.
    # Invoking script from command-line triggers the test.
    svhn_eval = SVHNEval(batch_size=2)

    for i in range(0, 32, 2):
        images, bboxes, labels = svhn_eval.next_batch()
        ax = plt.subplot(4,8,i+1)
        plt.imshow(images[0] + 0.5)
        plt.axis('off')
        plt.title(labels['string'][0])
        bbox_left, bbox_top, bbox_width, bbox_height = bboxes[0, 1], bboxes[0, 0], bboxes[0, 3], bboxes[0, 2]
        ax.add_patch(patches.Rectangle((bbox_left, bbox_top), bbox_width, bbox_height, fill=False))
        ax = plt.subplot(4,8,i+2)
        plt.imshow(images[1] + 0.5)
        plt.axis('off')
        plt.title(labels['string'][1])
        bbox_left, bbox_top, bbox_width, bbox_height = bboxes[1, 1], bboxes[1, 0], bboxes[1, 3], bboxes[1, 2]
        ax.add_patch(patches.Rectangle((bbox_left, bbox_top), bbox_width, bbox_height, fill=False))

    plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))