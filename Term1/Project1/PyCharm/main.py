import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import helpers

# reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
plt.imshow(image)
pass

# Obtain grayscale version of this image
gray_image = helpers.grayscale(image)
plt.imshow(gray_image, cmap='gray')
pass

# Perform histogram equalization
histeq_image = helpers.histeq(gray_image)
plt.imshow(histeq_image, cmap='gray')
pass

# Perform thresholding
binary_image = helpers.threshold_image_gray(histeq_image, threshold=230)
plt.imshow(binary_image, cmap='gray')
pass

# Perform region masking
ysize = image.shape[0]
xsize = image.shape[1]
vertices = np.array([[0, ysize], [xsize, ysize], [xsize*0.5, 0.55*ysize]], np.int32)
binary_image_masked = helpers.region_of_interest(binary_image, vertices)
plt.imshow(binary_image_masked, cmap='gray')
pass