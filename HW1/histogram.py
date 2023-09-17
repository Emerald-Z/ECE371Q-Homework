from matplotlib import pyplot as plt
import numpy as np
import cv2
from utilities import display

image = cv2.imread("q2.jpg", cv2.IMREAD_GRAYSCALE)
h, w = image.shape

# a
counts = np.zeros(256)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        counts[image[i][j]] += 1
# counts, bins = np.histogram(image.flatten(), bins=256) # with numpy built in

plt.title("original image histogram q2")
plt.hist(image.flatten(), bins=256)

'''
    The image is predominantly underexposed, for it is predominantly dark
    the histogram is right-skewed, meaning there are greater amounts of 
    smaller values
'''
A = np.max(image.flatten()) # minimum
B = np.min(image.flatten()) # maximum 
K = 256
# b
def histogram_equalization(image):
    normalized = 1.0 / (h * w)
    mapping = np.zeros((256, ), dtype=np.float16)

    for i in range(256):
        for j in range(i+1):
            mapping[i] += counts[j] * normalized # actual count
        mapping[i] = round(mapping[i] * 255)

    for i in range(h):
        for j in range(w):
            tmp = image[i, j]
            image[i, j] = mapping[tmp]
    return image

# c
def contrast_stretching(image):
    return (K - 1) * ((image - A) / (A + B)) 

# d
def gamma_correction(image, gamma):
    inv_gamma = 1 / gamma
    return ((np.array(image) / 255) ** inv_gamma) * 255

# e
images =[cv2.cvtColor(image, cv2.COLOR_BGR2RGB), histogram_equalization(np.array(image)), contrast_stretching(np.array(image)), gamma_correction(image, 3.5)]
labels = ['Original', 'Histogram equalization', 'color stretching', 'Gamma level between 2-4, this is 3.5']
display(2, 2, images, labels, True)

'''
    Out of the 3 enhancements, they all showed visible improvements. I belive that the gamma correction 
    technique performed the best. Full Contrast Stretching became overexposed, and histogram equalization
    (second best) was still slightly underexposed. Additionally, the gamma level is customizeable, and this
    allowed for the best, most visually appealing color enhancement.
'''

plt.show()