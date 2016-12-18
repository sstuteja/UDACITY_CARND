import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import helpers

# reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

# printing out some stats and plotting
print('Image Type: ', type(image))
print('Image Dimensions: ', image.shape)
plt.imshow(image)
pass