import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        #print(line[3])

        
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        #print(current_path)
        image = cv2.imread(current_path)
        #print(image.shape)
        images.append(image)
        #print(line[3])
        corr = 0.0
        if (i == 1):
            corr = 0.2
        if (i == 2):
            corr = -0.2
        measurement = float(line[i+3]) + corr
        measurements.append(measurement)
    
augmented_images, augmented_measurements= [], []
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement)
    augmented_measurements.append(measurement*-1.0)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
cv2.imwrite('1.jpg',images[1])

model = Sequential()
model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2)))
#model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(48, kernel_size=(5, 5), strides=(2,2)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(3, 3)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(3, 3)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
#model.add(Activation('relu'))
model.add(Dense(50))
#model.add(Activation('relu'))
model.add(Dense(10))
#model.add(Activation('relu'))
model.add(Dense(1))
#model.add(Activation('softmax'))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True,epochs=3)

model.save('model.h5')

          
