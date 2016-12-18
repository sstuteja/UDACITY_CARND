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

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

# Step 2: Performing lane detection on all test videos
vidlist = ['test_videos/solidWhiteRight.mp4', \
           'test_videos/solidYellowLeft.mp4', \
           'test_videos/challenge.mp4']
for vid in vidlist:
    (filename, file_extension) = os.path.splitext(vid)

    cap = cv2.VideoCapture(vid)
    (ret, frame) = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(filename + "_PROCESSED" + '.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    cap.release()
    cap = cv2.VideoCapture(vid)

    while cap.isOpened():
        (ret, frame) = cap.read()
        if ret:
            final_image = helpers.lane_detect(frame, UseOpenCV=True)
            out.write(final_image)
            cv2.imshow('final_image', final_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
