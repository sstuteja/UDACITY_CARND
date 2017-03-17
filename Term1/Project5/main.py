import glob
import time
import os
import os.path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lesson_functions as lib
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.ndimage.measurements
import pickle
import imageio
import random

#Initialize Settings
config = {}
config['color_space'] = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
config['orient'] = 9 # HOG orientations
config['pix_per_cell'] = 8 # HOG pixels per cell
config['cell_per_block'] = 2 # HOG cells per block
config['spatial_size'] = (32, 32) # Spatial binning dimensions
config['hist_bins'] = 32 # Number of histogram bins
config['spatial_feat'] = True #Enable/disable spatial features
config['hist_feat'] = True #Enable/disable histogram features
config['hog_feat'] = True #Enable/disable HOG features
config['hog_channel'] = 'ALL' #Can be 0, 1, 2, or 'ALL'
config['ystart'] = 400 #Start of cropped image (Y)
config['ystop'] = 656 #Stop position of cropped image (Y)
config['scale'] = 1.5 #Scale factor for subsampling window search

VehicleImagePaths = [os.path.join('vehicles', 'vehicles', 'GTI_Far', '*.png'), \
                         os.path.join('vehicles', 'vehicles', 'GTI_Left', '*.png'), \
                         os.path.join('vehicles', 'vehicles', 'GTI_MiddleClose', '*.png'), \
                         os.path.join('vehicles', 'vehicles', 'GTI_Right', '*.png'), \
                         os.path.join('vehicles', 'vehicles', 'KITTI_extracted', '*.png')]
NonVehicleImagePaths = [os.path.join('non-vehicles', 'non-vehicles', 'Extras', '*.png'), \
                        os.path.join('non-vehicles', 'non-vehicles', 'GTI', '*.png')]

VehicleImageList = []
for thisVehicleImagePath in VehicleImagePaths:
    VehicleImageList.extend(glob.glob(thisVehicleImagePath))
    
NonVehicleImageList = []
for thisNonVehicleImagePath in NonVehicleImagePaths:
    NonVehicleImageList.extend(glob.glob(thisNonVehicleImagePath))
    
#Show random car image and demonstrate HOG
idx_car = random.randint(0, len(VehicleImageList)-1)
idx_noncar = random.randint(0, len(NonVehicleImageList)-1)
carimg = cv2.imread(VehicleImageList[idx_car])
carimg = cv2.cvtColor(carimg, cv2.COLOR_BGR2RGB)
carimg_YCrCb = cv2.cvtColor(carimg, cv2.COLOR_RGB2YCrCb)
feat, carimg_Y_HOG = lib.get_hog_features(carimg_YCrCb[:, :, 0], orient=config['orient'], pix_per_cell=config['pix_per_cell'], cell_per_block=config['cell_per_block'], vis=True)
noncarimg = cv2.imread(NonVehicleImageList[idx_noncar])
noncarimg = cv2.cvtColor(noncarimg, cv2.COLOR_BGR2RGB)
noncarimg_YCrCb = cv2.cvtColor(noncarimg, cv2.COLOR_RGB2YCrCb)
feat, noncarimg_Y_HOG = lib.get_hog_features(noncarimg_YCrCb[:, :, 0], orient=config['orient'], pix_per_cell=config['pix_per_cell'], cell_per_block=config['cell_per_block'], vis=True)
plt.figure()
plt.subplot(4, 4, 1); plt.imshow(carimg_YCrCb[:, :, 0], cmap='gray'); plt.title('Car CH-1', fontsize=10);
plt.subplot(4, 4, 2); plt.imshow(carimg_Y_HOG, cmap='gray'); plt.title('Car CH-1 HOG', fontsize=10);
plt.subplot(4, 4, 3); plt.imshow(noncarimg_YCrCb[:, :, 0], cmap='gray'); plt.title('Non Car CH-1', fontsize=10);
plt.subplot(4, 4, 4); plt.imshow(noncarimg_Y_HOG, cmap='gray'); plt.title('Non Car CH-1 HOG', fontsize=10);
plt.subplot(4, 4, 5); plt.imshow(carimg_YCrCb[:, :, 0], cmap='gray'); plt.title('Car CH-1', fontsize=10);
plt.subplot(4, 4, 6); plt.imshow(carimg_YCrCb[:, :, 0], cmap='gray'); plt.title('Car CH-1 Features', fontsize=10);
plt.subplot(4, 4, 7); plt.imshow(noncarimg_YCrCb[:, :, 0], cmap='gray'); plt.title('Non Car CH-1', fontsize=10);
plt.subplot(4, 4, 8); plt.imshow(noncarimg_YCrCb[:, :, 0], cmap='gray'); plt.title('Non Car CH-1 Features', fontsize=10);
plt.subplot(4, 4, 9); plt.imshow(carimg_YCrCb[:, :, 1], cmap='gray'); plt.title('Car CH-2', fontsize=10);
plt.subplot(4, 4, 10); plt.imshow(carimg_YCrCb[:, :, 1], cmap='gray'); plt.title('Car CH-2 Features', fontsize=10);
plt.subplot(4, 4, 11); plt.imshow(noncarimg_YCrCb[:, :, 1], cmap='gray'); plt.title('Non Car CH-2', fontsize=10);
plt.subplot(4, 4, 12); plt.imshow(noncarimg_YCrCb[:, :, 1], cmap='gray'); plt.title('Non Car CH-2 Features', fontsize=10);
plt.subplot(4, 4, 13); plt.imshow(carimg_YCrCb[:, :, 2], cmap='gray'); plt.title('Car CH-3', fontsize=10);
plt.subplot(4, 4, 14); plt.imshow(carimg_YCrCb[:, :, 2], cmap='gray'); plt.title('Car CH-3 Features', fontsize=10);
plt.subplot(4, 4, 15); plt.imshow(noncarimg_YCrCb[:, :, 2], cmap='gray'); plt.title('Non Car CH-3', fontsize=10);
plt.subplot(4, 4, 16); plt.imshow(noncarimg_YCrCb[:, :, 2], cmap='gray'); plt.title('Non Car CH-3 Features', fontsize=10);

if os.path.exists('config.p'):
    config = pickle.load(open('config.p', 'rb'))
else:
    #Step 1: Extract features and labels, train the classifier
    t = time.time()
    print('EXTRACTING CAR FEATURES...')
    car_features = lib.extract_features(VehicleImageList, \
                                        color_space = config['color_space'], \
                                        spatial_size = config['spatial_size'], \
                                        hist_bins = config['hist_bins'], \
                                        orient = config['orient'], \
                                        pix_per_cell = config['pix_per_cell'], \
                                        cell_per_block = config['cell_per_block'], \
                                        spatial_feat = config['spatial_feat'], \
                                        hist_feat = config['hist_feat'], \
                                        hog_feat = config['hog_feat'], \
                                        hog_channel = config['hog_channel'])
    print('DONE, TOOK', time.time()-t, 'SECONDS')

    print('EXTRACTING NON CAR FEATURES...')
    t = time.time()
    notcar_features = lib.extract_features(NonVehicleImageList, \
                                           color_space = config['color_space'], \
                                           spatial_size = config['spatial_size'], \
                                           hist_bins = config['hist_bins'], \
                                           orient = config['orient'], \
                                           pix_per_cell = config['pix_per_cell'], \
                                           cell_per_block = config['cell_per_block'], \
                                           spatial_feat = config['spatial_feat'], \
                                           hist_feat = config['hist_feat'], \
                                           hog_feat = config['hog_feat'], \
                                           hog_channel = config['hog_channel'])
    print('DONE, TOOK', time.time()-t, 'SECONDS')

    print('SCALING ALL FEATURE VECTORS...')
    t = time.time()
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    config['X_scaler'] = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = config['X_scaler'].transform(X)
    print('FEATURES:', scaled_X.shape)
    print('DONE, TOOK', time.time()-t, 'SECONDS')

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.35, random_state=rand_state)

    #svc = LinearSVC(max_iter=10000000)
    config['svc'] = SVC(cache_size=600, verbose=True, kernel='rbf', C=1.0)
    t = time.time()
    print('NOW TRAINING SUPPORT VECTOR CLASSIFIER...')
    config['svc'].fit(X_train, y_train)
    print('DONE, TOOK', time.time()-t, 'SECONDS')

    t = time.time()
    print('CHECKING CLASSIFIER ACCURACY...')
    print('Test Accuracy of SVC = ', config['svc'].score(X_test, y_test))
    print('DONE, TOOK', time.time()-t, 'SECONDS')

    t = time.time()
    print('SAVING CONFIGURATION TO config.p ...')
    pickle.dump(config, open('config.p', 'wb'))
    print('DONE, TOOK', time.time()-t, 'SECONDS')

###############################################################################

# Step 2: Run the find_cars function on all test images
test_image_list = [os.path.join('test_images', 'test1.jpg'), \
                   os.path.join('test_images', 'test2.jpg'), \
                   os.path.join('test_images', 'test3.jpg'), \
                   os.path.join('test_images', 'test4.jpg'), \
                   os.path.join('test_images', 'test5.jpg'), \
                   os.path.join('test_images', 'test6.jpg')]

for thisImage in test_image_list:
    t = time.time()
    print('EVALUATING IMAGE', thisImage, '...')
    
    root, ext = os.path.splitext(thisImage)
    img = cv2.imread(thisImage)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox_list = lib.find_cars(img, \
                          ystart = config['ystart'], \
                          ystop = config['ystop'], \
                          scale = config['scale'], \
                          svc = config['svc'], \
                          X_scaler = config['X_scaler'], \
                          orient = config['orient'], \
                          pix_per_cell = config['pix_per_cell'], \
                          cell_per_block = config['cell_per_block'], \
                          spatial_size = config['spatial_size'], \
                          hist_bins = config['hist_bins'], \
                          spatial_feat = config['spatial_feat'], \
                          hist_feat = config['hist_feat'], \
                          hog_feat = config['hog_feat'], \
                          hog_channel = config['hog_channel'], \
                          color_space = config['color_space'])
    
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    features, img_hog_ch1 = lib.get_hog_features(img_YCrCb[:, :, 0], \
                                                 orient = config['orient'], \
                                                 pix_per_cell = config['pix_per_cell'], \
                                                 cell_per_block = config['cell_per_block'], \
                                                 vis = True)
    features, img_hog_ch2 = lib.get_hog_features(img_YCrCb[:, :, 1], \
                                                 orient = config['orient'], \
                                                 pix_per_cell = config['pix_per_cell'], \
                                                 cell_per_block = config['cell_per_block'], \
                                                 vis = True)
    features, img_hog_ch3 = lib.get_hog_features(img_YCrCb[:, :, 2], \
                                                 orient = config['orient'], \
                                                 pix_per_cell = config['pix_per_cell'], \
                                                 cell_per_block = config['cell_per_block'], \
                                                 vis = True)
    
    mpimg.imsave(root + '_1_FEATURE_Y.png', img_YCrCb[:, :, 0], cmap = 'gray')
    mpimg.imsave(root + '_2_FEATURE_Cr.png', img_YCrCb[:, :, 1], cmap = 'gray')
    mpimg.imsave(root + '_3_FEATURE_Cb.png', img_YCrCb[:, :, 2], cmap = 'gray')
    
    mpimg.imsave(root + '_4_HOG_CHANNEL_Y.png', img_hog_ch1, cmap = 'gray')
    mpimg.imsave(root + '_5_HOG_CHANNEL_Cr.png', img_hog_ch2, cmap = 'gray')
    mpimg.imsave(root + '_6_HOG_CHANNEL_Cb.png', img_hog_ch3, cmap = 'gray')
    
    img_hogsubsampling = lib.draw_boxes(img, bbox_list)
    mpimg.imsave(root + '_7_HOG_SUBSAMPLING.png', img_hogsubsampling)
    
    heatmap = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
    heatmap = lib.add_heat(heatmap, bbox_list)
    heatmap = lib.apply_heatmap_threshold(heatmap, threshold=0)
    mpimg.imsave(root + '_8_HEATMAP.png', heatmap, cmap='gray')
    
    heatmap[heatmap != 0] = 255
    mpimg.imsave(root + '_9_HEATMAP_BINARY.png', heatmap, cmap='gray')
    
    labels = scipy.ndimage.measurements.label(heatmap)
    img_final = lib.draw_labeled_bboxes(np.copy(img), labels)
    mpimg.imsave(root + '_10_PROCESSED.png', img_final)    
    
    print('DONE, TOOK', time.time()-t, 'SECONDS')
    
###############################################################################

# Step 3: Process the project video

VideoList = ['test_video.mp4', 'project_video.mp4']

for thisVideo in VideoList:
    t = time.time()
    print('PROCESSING VIDEO', thisVideo, '...')
    filename, file_extension = os.path.splitext(thisVideo)
    
    vidreader = imageio.get_reader(thisVideo, 'ffmpeg')
    fps = vidreader.get_meta_data()['fps']
    vidwriter = imageio.get_writer(filename + '_PROCESSED' + file_extension, fps=fps)
    ctr = 1
    for i, frame in enumerate(vidreader):
        if i % 15 == 0:
            bbox_list = lib.find_cars(frame, \
                                      ystart = config['ystart'], \
                                      ystop = config['ystop'], \
                                      scale = config['scale'], \
                                      svc = config['svc'], \
                                      X_scaler = config['X_scaler'], \
                                      orient = config['orient'], \
                                      pix_per_cell = config['pix_per_cell'], \
                                      cell_per_block = config['cell_per_block'], \
                                      spatial_size = config['spatial_size'], \
                                      hist_bins = config['hist_bins'], \
                                      spatial_feat = config['spatial_feat'], \
                                      hist_feat = config['hist_feat'], \
                                      hog_feat = config['hog_feat'], \
                                      hog_channel = config['hog_channel'], \
                                      color_space = config['color_space'])
            
            heatmap = np.zeros((frame.shape[0], frame.shape[1])).astype(np.uint8)
            heatmap = lib.add_heat(heatmap, bbox_list)
            heatmap = lib.apply_heatmap_threshold(heatmap, threshold=0)
            
            heatmap[heatmap != 0] = 255
            labels = scipy.ndimage.measurements.label(heatmap)
            img_final = lib.draw_labeled_bboxes(np.copy(frame), labels)
        
        else:
            img_final = lib.draw_labeled_bboxes(np.copy(frame), labels)
        
        vidwriter.append_data(img_final)
        
    vidwriter.close()
    print('DONE, TOOK', time.time()-t, 'SECONDS')