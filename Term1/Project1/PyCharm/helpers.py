import math
import cv2
import numpy as np


def lane_detect(image):
    """Perform lane detection"""

    # Apply Gaussian blurring
    blur_image = gaussian_blur(image, kernel_size=3)

    # Perform thresholding
    binary_image = threshold_image_gray(blur_image, threshold=220)

    # Perform Canny edge detection
    edge_image = canny(binary_image, low_threshold=150, high_threshold=200)

    # Perform region masking
    ysize = image.shape[0]
    xsize = image.shape[1]
    vertices = np.array([[0, ysize], [xsize, ysize], [xsize * 0.55, 0.6 * ysize], [xsize * 0.45, 0.6 * ysize]],
                        np.int32)
    binary_image_masked = region_of_interest(edge_image, vertices)

    # Apply Hough Transform
    line_image = np.copy(image) * 0  # creating a blank to draw lines on
    line_image = hough_lines(binary_image_masked, rho=1, theta=np.pi / 180, threshold=1, min_line_len=3,
                             max_line_gap=10)

    final_image = weighted_img(line_image, image)

    return final_image


def threshold_image_color(img, red_threshold, green_threshold, blue_threshold):
    """
    Performs image thresholding for color images.
    """
    image = np.copy(img)
    idx1 = img[:, :, 0] < red_threshold
    idx2 = img[:, :, 1] < green_threshold
    idx3 = img[:, :, 2] < blue_threshold
    image[idx1 | idx2 | idx3] = [0, 0, 0]
    image[~(idx1 | idx2 | idx3)] = [255, 255, 255]
    return image


def threshold_image_gray(img, threshold):
    """
    Performs image thresholding for grayscale images.
    """
    image = np.copy(img)
    idx = img < threshold
    image[idx] = 0
    image[~idx] = 255
    return image


def histeq(img):
    red = cv2.equalizeHist(img[:, :, 0])
    green = cv2.equalizeHist(img[:, :, 1])
    blue = cv2.equalizeHist(img[:, :, 2])
    return cv2.merge((red, green, blue))


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def grayscale_OpenCV(img):
    """Applies the Grayscale transform"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, [vertices], ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    list_x_left = []
    list_y_left = []
    list_x_right = []
    list_y_right = []
    Y2_LEFT = 0.65 * img.shape[0]
    Y2_RIGHT = 0.65 * img.shape[0]
    Y1_LEFT = img.shape[0]
    Y1_RIGHT = img.shape[0]
    Slope_Left_LLIM = -65 * math.pi / 180
    Slope_Left_ULIM = -20 * math.pi / 180
    Slope_Right_LLIM = 20 * math.pi / 180
    Slope_Right_ULIM = 65 * math.pi / 180
    Sum_SlopeLeft = 0.0
    Sum_SlopeRight = 0.0
    Count_SlopeLeft = 0.0
    Count_SlopeRight = 0.0
    for line in lines:
        for x1, y1, x2, y2 in line:
            thisSlopeAngle = math.atan2(y2 - y1, x2 - x1)
            if x1 <= 0.5*img.shape[1] and x2 <= 0.5*img.shape[1]:
                list_x_left.extend([x1, x2])
                list_y_left.extend([y1, y2])
                Sum_SlopeLeft += thisSlopeAngle
                Count_SlopeLeft += 1.0
            elif x1 >= 0.6*img.shape[1] and x2 >= 0.6*img.shape[1]:
                list_x_right.extend([x1, x2])
                list_y_right.extend([y1, y2])
                Sum_SlopeRight += thisSlopeAngle
                Count_SlopeRight += 1.0

            # cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    if len(list_x_left) >= 2 and len(list_y_left) >= 2:
        min_list_x_left = min(list_x_left)
        max_list_x_left = max(list_x_left)

        list_x_left_array = np.array(list_x_left)
        list_y_left_array = np.array(list_y_left)
        sorted_indices = np.argsort(list_x_left_array)
        list_x_left_array = list_x_left_array[sorted_indices]
        list_y_left_array = list_y_left_array[sorted_indices]
        list_x_left = list_x_left_array.tolist()
        list_y_left = list_y_left_array.tolist()

        COEFFS_LEFT = np.polyfit(list_x_left_array, list_y_left_array, 1)
        if math.tan(Slope_Left_ULIM) >= COEFFS_LEFT[0] >= math.tan(Slope_Left_LLIM):
            X1_LEFT = (Y1_LEFT - COEFFS_LEFT[1])/COEFFS_LEFT[0]
            X2_LEFT = (Y2_LEFT - COEFFS_LEFT[1])/COEFFS_LEFT[0]
            cv2.line(img, (int(X1_LEFT), int(Y1_LEFT)), (int(X2_LEFT), int(Y2_LEFT)), color, thickness)

    if len(list_x_right) >= 2 and len(list_y_right) >= 2:
        min_list_x_right = min(list_x_right)
        max_list_x_right = max(list_x_right)

        list_x_right_array = np.array(list_x_right)
        list_y_right_array = np.array(list_y_right)
        sorted_indices = np.argsort(list_x_right_array)
        list_x_right_array = list_x_right_array[sorted_indices]
        list_y_right_array = list_y_right_array[sorted_indices]
        list_x_right = list_x_right_array.tolist()
        list_y_right = list_y_right_array.tolist()

        COEFFS_RIGHT = np.polyfit(list_x_right_array, list_y_right_array, 1)

        if math.tan(Slope_Right_ULIM) >= COEFFS_RIGHT[0] >= math.tan(Slope_Right_LLIM):
            X1_RIGHT = (Y1_RIGHT - COEFFS_RIGHT[1]) / COEFFS_RIGHT[0]
            X2_RIGHT = (Y2_RIGHT - COEFFS_RIGHT[1]) / COEFFS_RIGHT[0]
            cv2.line(img, (int(X1_RIGHT), int(Y1_RIGHT)), (int(X2_RIGHT), int(Y2_RIGHT)), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, thickness=10)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1.0, lambd=0.0):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * alpha + img * beta + lambd
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, lambd)