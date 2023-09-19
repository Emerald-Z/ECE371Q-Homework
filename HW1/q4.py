from matplotlib import pyplot as plt
import numpy as np
import cv2
from utilities import display

image = cv2.imread("q4.jpg",cv2.IMREAD_GRAYSCALE)
# a
# convert [0, 255] to [0, 7]
three_bit = np.array(image) * (7/256)
q3_a = plt.figure(figsize=(10, 7))

# display(1, 2, [image, np.array(three_bit, 'uint8')], ['original', '3-bit quantized'], True)

# b
bit_planes = np.unpackbits(np.reshape(image, (image.shape[0], image.shape[1], -1)), axis=2)
images = [bit_planes[:,:, i] for i in reversed(range(8))]

labels = ['{i}th plane'.format(i=i) for i in range(8)]
# display(2, 4, images, labels, True)


# c
q3_c = plt.figure(figsize=(10, 7))
rows = 2
columns = 4

# editing fourth plane to have a black square in the middle
plt.gray()

def generate_image(plane):
    copy_img = bit_planes.copy()
    copy_img[int(image.shape[0] / 4) : int(image.shape[0] * 3 / 4), int(image.shape[1] / 4) : int(image.shape[1] * 3 / 4), plane] = 0
    return copy_img

images = [np.packbits(generate_image(i), axis=2) for i in reversed(range(8))]
labels = ["plane {i}".format(i = i) for i in range(8)]
display(2, 4, images, labels, True)

'''
    - some of the images are distorted, actually the latest frames(most significant) are distorted
    - plane 5 has some kind of grayscale distortion, while 6 and 7 are heavily distorted(the face in plane 7 is unscrutable)
    - the distortion is mainly on the very detailed parts of the image, where fine detail(more information) is of
    more importance. The hidden info(square) is more apparent on the later frames because the fine dark detail is
    mostly contained in the most significant planes
'''
