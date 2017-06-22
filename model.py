import csv
import cv2
import pickle
import numpy as np
import time


# parse csv file to read in images and steering angles
# csv format: center,left,right,steering,throttle,brake,speed

print('Loading data...')
start = time.time()
images = []
measurements = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		img_filename = 'data/' + line[0] # taking only center images for now
		image = cv2.imread(img_filename)
		images.append(image)
		measurements.append(float(line[3]))

X_train = np.array(images)
y_train = np.array(measurements)
print(y_train.shape, ' images loaded in ', time.time() - start, 's')

# pickle.dump({"X_train": images, "y_train": measurements}, open('data.p', 'wb'))

# load data
# start = time.time()
# data = pickle.load(open('data.p', 'rb'))

# X_train = np.array(data["X_train"])
# y_train = np.array(data["y_train"])
# print(y_train.shape, ' images loaded in ', time.time() - start, 's')


from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D

# build and train a regression model (LeNet)
model = Sequential()
# normalize and mean center images
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')											
model.fit(X_train, y_train, nb_epoch=6, validation_split=0.2, shuffle=True)
model.save('model.h5')




# if __name__ == '__main__':
# 	main()
