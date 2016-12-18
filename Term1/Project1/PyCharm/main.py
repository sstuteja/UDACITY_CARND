import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import helpers
import os
import cv2

# Step 1: Performing lane detection on all test images
imglist = ['test_images/solidWhiteCurve.jpg', \
           'test_images/solidWhiteRight.jpg', \
           'test_images/solidYellowCurve.jpg', \
           'test_images/solidYellowCurve2.jpg', \
           'test_images/solidYellowLeft.jpg', \
           'test_images/whiteCarLaneSwitch.jpg']
for img in imglist:
    image = mpimg.imread(img)
    final_image = helpers.lane_detect(image)

    (filename, file_extension) = os.path.splitext(img)
    mpimg.imsave(filename + "_PROCESSED" + file_extension, final_image)

# Step 2: Performing lane detection on all test videos
vidlist = ['test_videos/challenge.mp4', \
           'test_videos/solidWhiteRight.mp4', \
           'test_videos/solidYellowLeft.mp4']
for vid in vidlist:
    (filename, file_extension) = os.path.splitext(vid)

    cap = cv2.VideoCapture(vid)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(filename + "_PROCESSED" + file_extension, fourcc, 20.0, (640, 480))

    while (cap.isOpened()):
        (ret, frame) = cap.read()
        final_image = helpers.lane_detect(frame)
        out.write(final_image)
        cv2.imshow('frame', final_image)

    cap.release()
    out.release()