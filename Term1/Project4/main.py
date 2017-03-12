import cv2
import numpy as np
import os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import math
import imageio

#MAKE SURE ALL INPUT IMAGES ARE RGB IMAGES, NOT BGR IMAGES

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255), colorspace='HLS', channel=0):
    # Apply the following steps to img
    # 1) Convert to specified color scheme
    if colorspace == 'GRAY':
        out = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif colorspace == 'HLS':
        out = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        out = out[:, :, channel]
    elif colorspace == 'HSV':
        out = cv2.cvtColor(img, cv2.CCOLOR_RGB2HSV)
        out = out[:, :, channel]
    elif colorspace == 'RGB':
        out = out[:, :, channel]

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel_out = cv2.Sobel(out, cv2.CV_64F, 1, 0)
    else:
        sobel_out = cv2.Sobel(out, cv2.CV_64F, 0, 1)

    # 3) Take the absolute value of the derivative or gradient
    abs_sobel_out = np.absolute(sobel_out)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel_out / np.max(abs_sobel_out))

    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= min(thresh)) & (scaled_sobel <= max(thresh))] = 1

    # 6) Return this mask as your binary_output image
    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255), colorspace='HLS', channel=0):
    # Calculate gradient magnitude
    # Apply the following steps to img
    # 1) Convert to specified color scheme
    if colorspace == 'GRAY':
        out = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif colorspace == 'HLS':
        out = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        out = out[:, :, channel]
    elif colorspace == 'HSV':
        out = cv2.cvtColor(img, cv2.CCOLOR_RGB2HSV)
        out = out[:, :, channel]
    elif colorspace == 'RGB':
        out = out[:, :, channel]

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(out, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(out, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude
    sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * sobelxy / np.max(sobelxy))

    # 5) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(sobelxy)
    mag_binary[(scaled_sobel >= min(mag_thresh)) & (scaled_sobel <= max(mag_thresh))] = 1

    # 6) Return this mask as your binary_output image
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2), colorspace='HLS', channel=0):
    # Apply the following steps to img
    # 1) Convert to specified color scheme
    if colorspace == 'GRAY':
        out = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif colorspace == 'HLS':
        out = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        out = out[:, :, channel]
    elif colorspace == 'HSV':
        out = cv2.cvtColor(img, cv2.CCOLOR_RGB2HSV)
        out = out[:, :, channel]
    elif colorspace == 'RGB':
        out = out[:, :, channel]

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(out, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(out, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the gradient direction angle
    grad_angle = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * grad_angle / np.max(grad_angle))

    # 5) Create a binary mask where mag thresholds are met
    dir_binary = np.zeros_like(grad_angle)
    dir_binary[(scaled_sobel >= min(thresh)) & (scaled_sobel <= max(thresh))] = 1

    # 6) Return this mask as your binary_output image
    return dir_binary

def camera_calibration(imagepathlist=[], nx=9, ny=6, shape=(720, 1280)):        
    #3D points in real world space
    objpoints = []
    
    #2D points in image plane
    imgpoints = []
    
    #Prepare object points: (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    
    #Read all images
    for thisImagePath in imagepathlist:
        gray = cv2.imread(thisImagePath, cv2.IMREAD_GRAYSCALE)
        
        #Some images are 1281X721, most are 1280X720. All images should be the
        #same size, hence this line
        gray = gray[0:shape[0], 0:shape[1]]

        #Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            gray = cv2.drawChessboardCorners(gray, (nx, ny), corners, ret)
            #plt.figure()
            #plt.imshow(gray, cmap='gray')
    
    return cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

def warp(img, src=np.float32([[253,686], [1041,677], [584,457], [696,457]]), dst=np.float32([[253, 686], [1041, 686], [253,0], [1041,0]])):
    img_size = (img.shape[1], img.shape[0])
    
    #Compute the perspective transform M
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, Minv

def sliding_window_init(binary_warped, ShouldVisualize=False, nwindows=15, margin=100, minpix=50, ym_per_pix = 30/720, xm_per_pix = 3.7/700):
    #Assuming you have created a warped binary image called
    #binary_warped
    histogram = np.sum(binary_warped[math.floor(binary_warped.shape[0]/2):, :], axis=0)
    
    #Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    
    #Find the peak of the left and right halves of the histogram
    #These will be the starting point for the left and right
    #lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    #Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    
    #Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    #Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    #Create empty lists to receive left and right
    left_lane_inds = []
    right_lane_inds = []
    
    #Step through the windows one by one
    for window in range(nwindows):
        #Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        if ShouldVisualize:
            #Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (40, 255, 40), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (40, 255, 40), 2)
            
        #Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        #Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        #If you found > minpix pixels, recenter next window
        #on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
    #Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    #Extract left and right pixel positions
    leftx_px = nonzerox[left_lane_inds]
    lefty_px = nonzeroy[left_lane_inds]
    rightx_px = nonzerox[right_lane_inds]
    righty_px = nonzeroy[right_lane_inds]
    
    #Calculate radius of curvature
    y_eval_px = binary_warped.shape[0]
    y_eval_meters = y_eval_px * ym_per_pix
    
    #Fit a second order polynomial to each
    left_fit_px = np.polyfit(lefty_px, leftx_px, 2)
    right_fit_px = np.polyfit(righty_px, rightx_px, 2)
    left_fit_meters = np.polyfit(lefty_px*ym_per_pix, leftx_px*xm_per_pix, 2)
    right_fit_meters = np.polyfit(righty_px*ym_per_pix, rightx_px*xm_per_pix, 2)
    
    ploty_px = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    ploty_meters = ploty_px * ym_per_pix
    left_fitx_px = left_fit_px[0]*ploty_px**2 + left_fit_px[1]*ploty_px + left_fit_px[2]
    right_fitx_px = right_fit_px[0]*ploty_px**2 + right_fit_px[1]*ploty_px + right_fit_px[2]
    left_fitx_meters = left_fit_meters[0]*ploty_meters**2 + left_fit_meters[1]*ploty_meters + left_fit_meters[2]
    right_fitx_meters = right_fit_meters[0]*ploty_meters**2 + right_fit_meters[1]*ploty_meters + right_fit_meters[2]
    
    left_curverad_px = ((1 + (2*left_fit_px[0]*y_eval_px + left_fit_px[1])**2)**1.5) / np.absolute(2*left_fit_px[0])
    right_curverad_px = ((1 + (2*right_fit_px[0]*y_eval_px + right_fit_px[1])**2)**1.5) / np.absolute(2*right_fit_px[0])
    left_curverad_meters = ((1 + (2*left_fit_meters[0]*y_eval_meters + left_fit_meters[1])**2)**1.5) / np.absolute(2*left_fit_meters[0])
    right_curverad_meters = ((1 + (2*right_fit_meters[0]*y_eval_meters + right_fit_meters[1])**2)**1.5) / np.absolute(2*right_fit_meters[0])
    
    lane_center_px = 0.5*(left_fitx_px[-1] + right_fitx_px[-1])
    center_offset_px = midpoint - lane_center_px
    center_offset_meters = center_offset_px * xm_per_pix
    
    #Obtaining reference lane as the more reliable lane for radius of curvature measurement
    if len(leftx_px) > len(rightx_px):
        ReferenceLane = 'LEFT'
    else:
        ReferenceLane = 'RIGHT'
    
    return {'out_img': out_img, \
            'leftx_px': leftx_px, 'lefty_px': lefty_px, 'rightx_px': rightx_px, 'righty_px': righty_px, \
            'ym_per_pix': ym_per_pix, 'xm_per_pix': xm_per_pix, \
            'left_fit_px': left_fit_px, 'right_fit_px': right_fit_px, \
            'left_fit_meters': left_fit_meters, 'right_fit_meters': right_fit_meters, \
            'left_fitx_px': left_fitx_px, 'right_fitx_px': right_fitx_px, \
            'left_fitx_meters': left_fitx_meters, 'right_fitx_meters': right_fitx_meters, \
            'ploty_px': ploty_px, 'ploty_meters': ploty_meters, \
            'left_curverad_px': left_curverad_px, 'right_curverad_px': right_curverad_px, \
            'left_curverad_meters': left_curverad_meters, 'right_curverad_meters': right_curverad_meters, \
            'center_offset_px': center_offset_px, 'center_offset_meters': center_offset_meters, \
            'ReferenceLane': ReferenceLane}

def sliding_window_next(binary_warped, left_fit_px, right_fit_px, margin=100, ym_per_pix=30/720, xm_per_pix=3.7/700):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    midpoint_px = np.int(binary_warped.shape[1]/2)
    
    left_lane_inds = ((nonzerox > (left_fit_px[0]*(nonzeroy**2) + left_fit_px[1]*nonzeroy + left_fit_px[2] - margin)) & (nonzerox < (left_fit_px[0]*(nonzeroy**2) + left_fit_px[1]*nonzeroy + left_fit_px[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit_px[0]*(nonzeroy**2) + right_fit_px[1]*nonzeroy + right_fit_px[2] - margin)) & (nonzerox < (right_fit_px[0]*(nonzeroy**2) + right_fit_px[1]*nonzeroy + right_fit_px[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx_px = nonzerox[left_lane_inds]
    lefty_px = nonzeroy[left_lane_inds] 
    rightx_px = nonzerox[right_lane_inds]
    righty_px = nonzeroy[right_lane_inds]
    
    y_eval_px = binary_warped.shape[0]
    y_eval_meters = y_eval_px * ym_per_pix
    
    # Fit a second order polynomial to each
    left_fit_new_px = np.polyfit(lefty_px, leftx_px, 2)
    right_fit_new_px = np.polyfit(righty_px, rightx_px, 2)
    left_fit_new_meters = np.polyfit(lefty_px * ym_per_pix, leftx_px * xm_per_pix, 2)
    right_fit_new_meters = np.polyfit(righty_px * ym_per_pix, rightx_px * xm_per_pix, 2)
    
    ploty_px = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    ploty_meters = ploty_px * ym_per_pix
    left_fitx_new_px = left_fit_new_px[0]*ploty_px**2 + left_fit_new_px[1]*ploty_px + left_fit_new_px[2]
    right_fitx_new_px = right_fit_new_px[0]*ploty_px**2 + right_fit_new_px[1]*ploty_px + right_fit_new_px[2]
    left_fitx_new_meters = left_fit_new_meters[0]*ploty_meters**2 + left_fit_new_meters[1]*ploty_meters + left_fit_new_meters[2]
    right_fitx_new_meters = right_fit_new_meters[0]*ploty_meters**2 + right_fit_new_meters[1]*ploty_meters + right_fit_new_meters[2]
    
    left_curverad_px = ((1 + (2*left_fit_new_px[0]*y_eval_px + left_fit_new_px[1])**2)**1.5) / np.absolute(2*left_fit_new_px[0])
    right_curverad_px = ((1 + (2*right_fit_new_px[0]*y_eval_px + right_fit_new_px[1])**2)**1.5) / np.absolute(2*right_fit_new_px[0])
    left_curverad_meters = ((1 + (2*left_fit_new_meters[0]*y_eval_meters + left_fit_new_meters[1])**2)**1.5) / np.absolute(2*left_fit_new_meters[0])
    right_curverad_meters = ((1 + (2*right_fit_new_meters[0]*y_eval_meters + right_fit_new_meters[1])**2)**1.5) / np.absolute(2*right_fit_new_meters[0])
    
    lane_center_px = 0.5*(left_fitx_new_px[-1] + right_fitx_new_px[-1])
    center_offset_px = midpoint_px - lane_center_px
    center_offset_meters = center_offset_px * xm_per_pix
    
    #Obtaining reference lane as the more reliable lane for radius of curvature measurement
    if len(leftx_px) > len(rightx_px):
        ReferenceLane = 'LEFT'
    else:
        ReferenceLane = 'RIGHT'
    
    return {'leftx_px': leftx_px, 'lefty_px': lefty_px, 'rightx_px': rightx_px, 'righty_px': righty_px, \
            'ym_per_pix': ym_per_pix, 'xm_per_pix': xm_per_pix, \
            'left_fit_px': left_fit_new_px, 'right_fit_px': right_fit_new_px, \
            'left_fit_meters': left_fit_new_meters, 'right_fit_meters': right_fit_new_meters, \
            'left_fitx_px': left_fitx_new_px, 'right_fitx_px': right_fitx_new_px, \
            'left_fitx_meters': left_fitx_new_meters, 'right_fitx_meters': right_fitx_new_meters, \
            'ploty_px': ploty_px, 'ploty_meters': ploty_meters, \
            'left_curverad_px': left_curverad_px, 'right_curverad_px': right_curverad_px, \
            'left_curverad_meters': left_curverad_meters, 'right_curverad_meters': right_curverad_meters, \
            'center_offset_px': center_offset_px, 'center_offset_meters': center_offset_meters, \
            'ReferenceLane': ReferenceLane}

def fill_lane_area(undist, warped_image, Minv, left_fitx_px, right_fitx_px, ploty_px, radius_of_curvature_meters, center_offset_meters, ReferenceLane):
    zero_warp = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((zero_warp, zero_warp, zero_warp))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx_px, ploty_px]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx_px, ploty_px])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    txt = 'Radius of curvature (m):' + str(round(radius_of_curvature_meters))
    result = cv2.putText(result, txt, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    txt = 'Offset from center (m):' + str(round(center_offset_meters, 2))
    result = cv2.putText(result, txt, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    txt = 'Reference Lane: ' + ReferenceLane
    result = cv2.putText(result, txt, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    return result

if __name__ == '__main__':
    #Step 1: Camera Calibration
    t = time.time()
    print('CALIBRATING CAMERA ... ', )
    ImagePaths = ['camera_cal/calibration1.jpg', \
                  'camera_cal/calibration2.jpg', \
                  'camera_cal/calibration3.jpg', \
                  'camera_cal/calibration4.jpg', \
                  'camera_cal/calibration5.jpg', \
                  'camera_cal/calibration6.jpg', \
                  'camera_cal/calibration7.jpg', \
                  'camera_cal/calibration8.jpg', \
                  'camera_cal/calibration9.jpg', \
                  'camera_cal/calibration10.jpg', \
                  'camera_cal/calibration11.jpg', \
                  'camera_cal/calibration12.jpg', \
                  'camera_cal/calibration13.jpg', \
                  'camera_cal/calibration14.jpg', \
                  'camera_cal/calibration15.jpg', \
                  'camera_cal/calibration16.jpg', \
                  'camera_cal/calibration17.jpg', \
                  'camera_cal/calibration18.jpg', \
                  'camera_cal/calibration19.jpg', \
                  'camera_cal/calibration20.jpg']
    ret, mtx, dist, rvecs, tvecs = camera_calibration(imagepathlist=ImagePaths)
    print('DONE, TOOK ', time.time() - t, ' SECONDS')
    
    #Demonstrate undistortion on sample calibration image
    img = mpimg.imread(ImagePaths[4])
    plt.figure(); plt.imshow(mpimg.imread(ImagePaths[0])); plt.title('Original Image')
    plt.figure(); plt.imshow(cv2.undistort(img, mtx, dist, None, mtx)); plt.title('Undistorted Image')    
    
    #Step 2: Set up image list
    ImageList = ['test_images/straight_lines1.jpg', \
                 'test_images/straight_lines2.jpg', \
                 'test_images/test1.jpg', \
                 'test_images/test2.jpg', \
                 'test_images/test3.jpg', \
                 'test_images/test4.jpg', \
                 'test_images/test5.jpg', \
                 'test_images/test6.jpg']
    for thisImage in ImageList:
        t = time.time()
        print('PROCESSING IMAGE ', thisImage, ' ... ', )
        origimg = mpimg.imread(thisImage)
        #plt.figure(); plt.imshow(origimg)
        img = cv2.undistort(origimg, mtx, dist, None, mtx)
        ImagePath, Ext = os.path.splitext(thisImage)
        plt.imsave(ImagePath + '_1_undistorted.png', img)
        
        #Step 3: Apply thresholding
        # Choose a Sobel kernel size
        ksize = 3 # Choose a larger odd number to smooth gradient measurements
        
        # Apply each of the thresholding functions
        gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(10, 100), colorspace='HLS', channel=2)
        grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(10, 100), colorspace='HLS', channel=2)
        mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(10, 100), colorspace='HLS', channel=2)
        dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.5, 1.6), colorspace='HLS', channel=2)
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        plt.imsave(ImagePath + '_2_binary.png', combined, cmap='gray')
    
        warped_img, Minv = warp(combined)
        plt.imsave(ImagePath + '_3_warped.png', warped_img, cmap='gray')
        
        x = sliding_window_init(warped_img, ShouldVisualize=True)
        plt.imsave(ImagePath + '_4_SlidingWindowLanes.png', x['out_img'], cmap='gray')
        
        if x['ReferenceLane'] == 'LEFT':
            radius_of_curvature_meters = x['left_curverad_meters']
        else:
            radius_of_curvature_meters = x['right_curverad_meters']
            
        out_img = fill_lane_area(undist=img, warped_image=warped_img, Minv=Minv, \
                                 left_fitx_px=x['left_fitx_px'], \
                                 right_fitx_px=x['right_fitx_px'], \
                                 ploty_px=x['ploty_px'], \
                                 radius_of_curvature_meters=radius_of_curvature_meters, \
                                 center_offset_meters=x['center_offset_meters'], \
                                 ReferenceLane=x['ReferenceLane'])
        plt.imsave(ImagePath + '_5_PROCESSED.png', out_img)
        print('DONE, TOOK ', time.time() - t, ' SECONDS')
        
    #Step 3: Process videos
    VideoList = ['project_video.mp4']
    for thisVideo in VideoList:
        t = time.time()
        print('PROCESSING VIDEO ', thisVideo, ' ... ', )
        (filename, file_extension) = os.path.splitext(thisVideo)
        
        vidreader = imageio.get_reader(thisVideo, 'ffmpeg')
        fps = vidreader.get_meta_data()['fps']
        vidwriter = imageio.get_writer(filename + '_PROCESSED' + file_extension, fps=fps)
        for i, frame in enumerate(vidreader):
#                
            ##################
            # PIPELINE
            ##################
            origimg = frame
            img = cv2.undistort(origimg, mtx, dist, None, mtx)
            
            #Step 3: Apply thresholding
            # Choose a Sobel kernel size
            ksize = 3 # Choose a larger odd number to smooth gradient measurements
            
            # Apply each of the thresholding functions
            gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(10, 100), colorspace='HLS', channel=2)
            grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(10, 100), colorspace='HLS', channel=2)
            mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(10, 100), colorspace='HLS', channel=2)
            dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.5, 1.6), colorspace='HLS', channel=2)
            combined = np.zeros_like(dir_binary)
            combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        
            warped_img, Minv = warp(combined)
            
            if i == 0:
                x = sliding_window_init(warped_img)
            else:
                x = sliding_window_next(warped_img, x['left_fit_px'].copy(), x['right_fit_px'].copy())
                
            if x['ReferenceLane'] == 'LEFT':
                radius_of_curvature_meters = x['left_curverad_meters']
            else:
                radius_of_curvature_meters = x['right_curverad_meters']
            
            final_image = fill_lane_area(undist=img, warped_image=warped_img, Minv=Minv, \
                                     left_fitx_px=x['left_fitx_px'], \
                                     right_fitx_px=x['right_fitx_px'], \
                                     ploty_px=x['ploty_px'], \
                                     radius_of_curvature_meters=radius_of_curvature_meters, \
                                     center_offset_meters=x['center_offset_meters'], \
                                     ReferenceLane=x['ReferenceLane'])
            
            ########################
            
            vidwriter.append_data(final_image)
        
        vidwriter.close()
        print('DONE, TOOK ', time.time()-t, ' SECONDS')
    
    
    