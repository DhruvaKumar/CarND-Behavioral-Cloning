import csv
import cv2
import pickle
import numpy as np
import time
import math
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn
import pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')


print('Loading data...')

# parse csv file to read in images and steering angles
# csv format: center,left,right,steering,throttle,brake,speed
samples = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(('data/'+line[0], float(line[3]))) # center image
		samples.append(('data/'+line[1].strip(), float(line[3])+0.25)) # left image
		samples.append(('data/'+line[2].strip(), float(line[3])-0.25)) # right image
		break

print(samples)

def preprocess_img(img):

	# change to YUV space as suggested in the Nvidia paper
	img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

	return img


# generate training and validation data
# train_samples, validation_samples = train_test_split(samples, test_size=0)
train_samples = samples

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1:
		samples = shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				image = cv2.imread(batch_sample[0])
				image = preprocess_img(image)

				images.append(image)
				angles.append(batch_sample[1])
		
			X = np.array(images)
			y = np.array(angles)	
			yield shuffle(X, y)


batch_size = 32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(train_samples, batch_size=batch_size)


print('Training...')
print('training data=',len(train_samples))
print('batch_size=',batch_size)

from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten, Lambda, Activation, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Cropping2D, ELU
from keras.regularizers import l2, activity_l2

start = time.time()

# build and train a regression model (LeNet)
model = Sequential()

# crop off top and bottom portion of image which do not contain the road
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
# normalize and mean center images
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(32, 5, 5))
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 5, 5))
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(128, 3, 3))
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(ELU())
model.add(Dense(64))
model.add(ELU())
model.add(Dense(16))
model.add(ELU())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')											
model_history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), 
	validation_data=validation_generator, nb_val_samples=len(train_samples), nb_epoch=5, verbose=1)
model.save('model_test.h5')

print('training completed in ', time.time() - start, 's')
print('training loss',model_history.history['loss'])
print('validation loss',model_history.history['val_loss'])

# # save model history to file
# pickle.dump(model_history.history, open("model_loss.p", "wb"))

# # plot the training and validation loss for each epoch
# plt.plot(model_history.history['loss'])
# plt.plot(model_history.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.savefig('loss_mse.png')
# plt.show()


# if __name__ == '__main__':
# 	main()
