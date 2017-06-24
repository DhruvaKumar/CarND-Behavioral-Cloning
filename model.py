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


# parse csv file to read in images and steering angles
# csv format: center,left,right,steering,throttle,brake,speed

print('Loading data...')

# read in image paths from csv file
samples = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

# generate training and validation data
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1:
		samples = shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				# TODO: load in other images too
				# TODO: change from BGR to RGB
				center_img_filename = 'data/' + batch_sample[0]
				center_image = cv2.imread(center_img_filename)
				images.append(center_image)
				angles.append(float(batch_sample[3]))
		
			X = np.array(images)
			y = np.array(angles)	
			yield shuffle(X, y)

batch_size = 32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


print('Training...')
print('training data=',len(train_samples))
print('validation data=',len(validation_samples))
print('batch_size=',batch_size)

from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten, Lambda, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D

start = time.time()

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
model_history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), 
	validation_data=validation_generator, nb_val_samples=len(validation_samples) nb_epoch=6, verbose=1)
model.save('model.h5')

print('training completed in ', time.time() - start, 's')


# save model history to file
pickle.dump(model_history.history, open("model_loss.p", "wb"))

# plot the training and validation loss for each epoch
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss_mse.png')
plt.show()



# if __name__ == '__main__':
# 	main()