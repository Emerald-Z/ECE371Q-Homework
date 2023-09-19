from matplotlib import pyplot as plt
import numpy as np
from numpy import lib
import cv2
from utilities import display

stars = cv2.imread("stars.jpg",cv2.IMREAD_GRAYSCALE)
fingerprint = cv2.imread("fingerprint.jpg",cv2.IMREAD_GRAYSCALE)

# a
def threshold(image, threshold):
    _, thresh = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY)
    thresh = np.zeros(image.shape[0], image.shape[1])
    # for i in range(image.shape[0]): # not using opencv
    #     for j in range(image.shape[1]):
    #         if image[i,j] >= threshold:
    #             thresh[i,j] = 0
    #         else: 
    #             thresh[i,j] = 1
    return thresh


'''
    Thresholding is more effective at values that are near the low-middle range, for both the comets and the 
    fingerprints are able to be effectively displayed in binary version. That being said, information does seem
    to be lost when thresholding. So I decided to use a lower threshold value of 80 for the stars(more white
    is preserved) and a higher threshold value of 120 for the fingerprint - so that noise would be factored out
    but not too much of the actual information(white rays for stars, black lines for prints) would be. 
'''
stars_thresh = threshold(stars, 80)
fingerprint_thresh = threshold(fingerprint, 120)

display(1, 2, [stars_thresh, fingerprint_thresh], ['stars', 'fingerprint'], True)

# b
def custom_dilate(image, neighbor_size):
    padding = int(neighbor_size/2)
    # pad matrix with size of neighbor
    padded_image = np.pad(image, pad_width=padding, mode='edge')
    padded_image = np.reshape(padded_image, (padded_image.shape[0], padded_image.shape[1], 1))
    # generate sliding window view
    windows = lib.stride_tricks.sliding_window_view(padded_image, (neighbor_size, neighbor_size, 1))
    windows = np.reshape(windows, (image.shape[0], image.shape[1], -1))
    # or each pixel
    or_windows = np.bitwise_or(np.reshape(image, (image.shape[0], image.shape[1], 1)), windows)
    return np.reshape(or_windows[:,:,1], (image.shape[0], image.shape[1]))
    # reshape and return

def custom_erode(image, neighbor_size):
    padding = int(neighbor_size/2)
    # pad matrix with size of neighbor
    padded_image = np.pad(image, pad_width=padding, mode='edge')
    padded_image = np.reshape(padded_image, (padded_image.shape[0], padded_image.shape[1], 1))
    # generate sliding window view
    windows = lib.stride_tricks.sliding_window_view(padded_image, (neighbor_size, neighbor_size, 1))
    windows = np.reshape(windows, (image.shape[0], image.shape[1], -1))
    # or each pixel
    or_windows = np.bitwise_and(np.reshape(image, (image.shape[0], image.shape[1], 1)), windows)
    return np.reshape(or_windows[:,:,1], (image.shape[0], image.shape[1]))
    # reshape and return

images = [custom_dilate(stars_thresh, 3), custom_dilate(fingerprint_thresh, 3), custom_erode(stars_thresh, 3), custom_erode(fingerprint_thresh, 3)]
labels = ['stars dilate', 'fingerprint dilate', 'stars erode', 'fingerprint erode']
display(2, 2, images, labels, True)

# c
def open(image, window):
    return custom_dilate(custom_erode(image, window), window)

def close(image, window):
    return custom_erode(custom_dilate(image, window), window)

def open_close(image, window):
    return open(close(image, window), window)

def close_open(image, window):
    return close(open(image, window), window)

def median(image, window):
    # or majority filter
    padding = int(window/2)
    # pad matrix with size of neighbot
    padded_image = np.pad(image, pad_width=padding, mode='edge')
    padded_image = np.reshape(padded_image, (padded_image.shape[0], padded_image.shape[1], 1))
    # generate sliding window view
    windows = lib.stride_tricks.sliding_window_view(padded_image, (window, window, 1))
    windows = np.reshape(windows, (image.shape[0], image.shape[1], -1))
    # or each pixel
    or_windows = np.median(windows, axis=2) # can i use this
    return np.reshape(or_windows, (image.shape[0], image.shape[1]))

images = [open(fingerprint_thresh, 3), close(fingerprint_thresh, 3), open_close(fingerprint_thresh, 3), close_open(fingerprint_thresh, 3), median(fingerprint_thresh, 3)]
labels = ['open', 'close', 'open-close', 'close-open', 'median']
display(1, 5, images, labels, True, True)

# d
boundary_image = open_close(fingerprint_thresh, 5)
boundary_image = custom_dilate(boundary_image, 5) 
boundary_image = close(boundary_image, 5) 

images = [fingerprint_thresh, boundary_image]
labels = ['original binary fingerprint', 'boundary image']
display(1, 2, images, labels)

# e
def boundary_length(image):
    print(np.count_nonzero(image))
    return np.count_nonzero(image)

boundary_length(boundary_image)

plt.show()
