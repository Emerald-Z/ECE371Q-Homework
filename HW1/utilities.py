import cv2
from matplotlib import pyplot as plt
import numpy as np

def display(rows, cols, images, labels, gray_scale=False, large_size=False):
    assert(len(images) == rows * cols == len(labels))
    if large_size:
        figure = plt.figure(figsize=(15, 15))
    else:
        figure = plt.figure(figsize=(10, 7))

    if gray_scale:
        plt.gray()
    for i in range(rows):
        for j in range(cols):
            if (cols <= rows): 
                idx = ((i * rows) + j)
            else:
                idx = ((i * cols) + j)
            figure.add_subplot(rows, cols, idx + 1)
            plt.imshow(images[idx]) # opencv represents in BGR vs RGB expected
            # plt.axis('off')
            plt.title(labels[idx])

    plt.show()
