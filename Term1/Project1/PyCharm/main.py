import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import helpers

# reading in an image
image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
plt.imshow(image)
pass

# Obtain grayscale version of this image
gray_image = helpers.grayscale(image)
plt.imshow(gray_image, cmap='gray')
pass

# Apply Gaussian blurring
blur_image = helpers.gaussian_blur(gray_image, kernel_size=3)

# Perform histogram equalization
histeq_image = helpers.histeq(blur_image)
plt.imshow(histeq_image, cmap='gray')
pass

# Perform thresholding
binary_image = helpers.threshold_image_gray(histeq_image, threshold=220)

# Perform Canny edge detection
edge_image = helpers.canny(binary_image, low_threshold=100, high_threshold=200)

# Perform region masking
ysize = image.shape[0]
xsize = image.shape[1]
vertices = np.array([[0, ysize], [xsize, ysize], [xsize*0.55, 0.6*ysize], [xsize*0.45, 0.6*ysize]], np.int32)
binary_image_masked = helpers.region_of_interest(edge_image, vertices)
plt.imshow(binary_image_masked, cmap='gray')
pass

# Apply Hough Transform
line_image = np.copy(image)*0 # creating a blank to draw lines on
line_image = helpers.hough_lines(binary_image_masked, rho=1, theta=np.pi/180, threshold=1, min_line_len=3, max_line_gap=10)
plt.imshow(line_image, cmap='gray')
pass

final_image = helpers.weighted_img(line_image, image)
plt.imshow(final_image)
pass