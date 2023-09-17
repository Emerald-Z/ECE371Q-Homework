from matplotlib import pyplot as plt
import numpy as np
import cv2
from utilities import display

image = cv2.imread("q4.jpg",cv2.IMREAD_GRAYSCALE)
# a
# convert [0, 255] to [0, 7]
three_bit = np.array(image) * (7/256)
q3_a = plt.figure(figsize=(10, 7))

display(1, 2, [image, np.array(three_bit, 'uint8')], ['original', '3-bit quantized'], True)

# b
bit_planes = np.unpackbits(np.array(np.reshape(image, (image.shape[0], image.shape[1], -1)), 'uint8'), axis=2)
images =[bit_planes[:,:, i] for i in reversed(range(8))]
labels = ['{i}th plane'.format(i=i) for i in range(8)]
display(2, 4, images, labels, True)


# c
q3_c = plt.figure(figsize=(10, 7))
rows = 2
columns = 4

q3_c.add_subplot(rows, columns, 1)
plt.imshow(bit_planes[:,:, 7]) # opencv represents in BGR vs RGB expected
plt.axis('off')
plt.title('0th plane') # make sure endianness is right???

q3_c.add_subplot(rows, columns, 2)
plt.imshow(np.packbits(bit_planes[:,:, 6:7], axis=-1))
plt.axis('off')
plt.title('Planes: 0, 1')

q3_c.add_subplot(rows, columns, 3)
plt.imshow(np.packbits(bit_planes[:,:, 5:7], axis=-1))
plt.axis('off')
plt.title('Planes: 0, 1, 2')

q3_c.add_subplot(rows, columns, 4)
plt.imshow(np.packbits(bit_planes[:,:, 4:7], axis=-1))
plt.axis('off')
plt.title('Planes: 0, 1, 2, 3')

q3_c.add_subplot(rows, columns, 5)
plt.imshow(np.packbits(bit_planes[:,:, 3:7], axis=-1))
plt.axis('off')
plt.title('Planes: 0, 1, 2, 3, 4')

q3_c.add_subplot(rows, columns, 6)
plt.imshow(np.packbits(bit_planes[:,:, 2:7], axis=-1))
plt.axis('off')
plt.title('Planes: 0, 1, 2, 3, 4, 5')

q3_c.add_subplot(rows, columns, 7)
plt.imshow(np.packbits(bit_planes[:,:, 1:7], axis=-1))
plt.axis('off')
plt.title('Planes: 0, 1, 2, 3, 4, 5, 6')

q3_c.add_subplot(rows, columns, 8)
plt.imshow(np.packbits(bit_planes[:,:, 0:7], axis=-1))
plt.axis('off')
plt.title('All planes')

'''
    - some of the images are distorted, actually 1 - 7 are distorted
    - only the last one is full(all planes)
    - the distortion is mainly on the very detailed parts of the image, where fine detail(more information) is of
    more importance. The background smoothes out the fastest, and the face smoothes out the slowest
'''
plt.show()