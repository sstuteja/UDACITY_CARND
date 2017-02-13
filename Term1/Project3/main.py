import numpy as np
import sklearn.utils
import sklearn.model_selection
import sklearn.preprocessing
import cv2
import os
import csv

from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#Discard first line as it represents the header
samples.pop(0)

train_samples, validation_samples = sklearn.model_selection.train_test_split(samples, test_size = 0.2)

#Define generator function
def generator(samples, batch_size = 32):
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)

                #Adding code to resize the image to make training faster
                center_image = cv2.resize(center_image, None, fx=0.5, fy=0.5)

                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                #Augmenting data set
                images.append(cv2.flip(center_image, flipCode=1))
                angles.append(-center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#Compile and train the model
model = Sequential()

model.add(Cropping2D(cropping=((25, 10), (0, 0)), input_shape=(80, 160, 3)))

model.add(Lambda(lambda x:x/127.5 - 1.0))

model.add(Convolution2D(24, 5, 5, init='he_normal', border_mode='valid'))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Convolution2D(36, 5, 5, init='he_normal', border_mode='valid'))
model.add(Activation('relu'))

model.add(Convolution2D(48, 5, 5, init='he_normal', border_mode='valid'))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3, init='he_normal', border_mode='valid'))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3, init='he_normal', border_mode='valid'))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=2)
pass
