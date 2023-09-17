import cv2
from matplotlib import pyplot as plt
import numpy as np
from utilities import display

# a
image = cv2.imread('balloons.jpg')
cv2.imshow('og balloons', image)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image

# b
b, g, r = cv2.split(image)

display(2, 2, [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in [b, g, r, image]], ['Blue', 'Green', 'Red', 'Original'])

# c
figure_c = plt.figure(figsize=(10, 7))
rows = 1
columns = 3

y = 0.299 * np.array(r) + 0.587 * np.array(g) + 0.114 * np.array(b)
u = -0.147 * np.array(r) - 0.289 * np.array(g) + 0.436 * np.array(b)
v = 0.615 * np.array(r) - 0.515 * np.array(g) - 0.1 * np.array(b)

display(1, 3, [cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB) for img in [y, u, v]], ['Y', 'U', 'V'])

# d
def grayscale_equal(r, g, b):
    return (np.array(r) + np.array(g) + np.array(b)) / 3

def grayscale_cv_weights(r, g, b):
    return (0.299 * np.array(r) + 0.587 * np.array(g) + 0.114 * np.array(b)) / 3

def luminosity_method_weights(r, g, b):
    return (0.2126 * np.array(r) + 0.7152 * np.array(g) + 0.0722 * np.array(b)) / 3

def desaturation(r, g, b):
    return np.maximum(np.array(r), np.array(g), np.array(b)) + np.minimum(np.array(r), np.array(g), np.array(b))


figure_d = plt.figure(figsize=(10, 7))
rows = 2
columns = 2

labels = ['Equal', 'openCV weights', 'luminosity(Y)', 'desaturation']
display(2, 2, [cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB) for img in [grayscale_equal(r, g, b), grayscale_cv_weights(r, g, b), luminosity_method_weights(r, g, b), desaturation(r, g, b)]], labels)

# e
half_h = int(image.shape[0] / 2)
half_w = int(image.shape[1] / 2)
copy_image = image.copy()
top_left_quadrant = copy_image[0:half_h, 0:half_w]

def rotate(grid):
    output = np.zeros(grid.shape)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            output[i, grid.shape[1]-1-j] = grid[grid.shape[0]-i-1, j]
    return output 

copy_image[0:half_h, 0:half_w] = rotate(top_left_quadrant)

figure_e = plt.figure(figsize=(10, 7))

display(1, 1, [cv2.cvtColor(np.uint8(copy_image), cv2.COLOR_BGR2RGB)], ['top left quadrant rotation'])

# f

figure_f = plt.figure(figsize=(10, 7))
display(1, 3, [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in [image, image[::-1], image[:, ::-1]]], ['Original', 'Vertical', 'Horizontal'])

# g
def color_filter(color, image, r, g, b):
    output = np.zeros(image.shape)
    if (color == 'blue'):
        output[:, :, 0] = b
    elif (color == 'red'):
        output[:, :, 2] = r
    else:
        # assume green
        output[:, :, 1] = g
    return output
    
figure_g = plt.figure(figsize=(10, 7))

images = [cv2.cvtColor(np.uint8(color_filter(color, image, r, g, b)), cv2.COLOR_BGR2RGB) for color in ['red', 'green', 'blue']]

display(1, 3, images, ['red', 'green', 'blue'])

# h

'''
    Color enhancement transforms are used to improve the visual appearance of an image by adjusting color contrast, 
    saturation, and other color-related properties. For example, contrast enhancement can be used to increase the visual 
    separation between colors in an image, while saturation enhancement can be used to make colors appear more vivid and intense.

    LAB color space expresses color variations across three channels. One channel for brightness and two channels for color:
        L-channel: representing lightness in the image
        a-channel: representing change in color between red and green
        b-channel: representing change in color between yellow and blue
    In the following I perform adaptive histogram equalization on the L-channel and convert the resulting image back to BGR color space. 
    This enhances the brightness while also limiting contrast sensitivity. I have done the following using OpenCV 3.0.0 and python:
'''
figure_h = plt.figure(figsize=(10, 7))
rows = 1
columns = 1

lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l_channel, a, b = cv2.split(lab)

# Applying CLAHE to L-channel
# feel free to try different values for the limit and grid size:
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
cl = clahe.apply(l_channel)

# merge the CLAHE enhanced L-channel with the a and b channel
limg = cv2.merge((cl,a,b))

# Converting image from LAB Color model to BGR color spcae
enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

figure_h.add_subplot(rows, columns, 1)
plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)) # opencv represents in BGR vs RGB expected
plt.axis('off')
plt.title('color enhancement')

plt.show()