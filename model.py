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

from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten, Lambda, Activation, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Cropping2D, ELU


# TODO: look into why steering is oscillating (prediction takes time or we predict angle > what we expect)
# TODO: reduce angle offset?
# TODO: reduce size of image so that prediction is faster?
# TODO: reduce complexity of model
# TODO: time how much prediction is taking for each image

# TODO: add vertical shifts
# TODO: add brightness/contrast changes



def load_data():
	print('Loading data...')

	# parse csv file to read in images and steering angles
	# csv format: center,left,right,steering,throttle,brake,speed
	data_lines = []
	with open('data/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			data_lines.append(line)

	return np.array(data_lines)


def preproc_data(data_lines):
	'''
	apply histogram equalization.
	'''

	print('histogram equalization: size before', data_lines.shape)

	data_lines = shuffle(data_lines)
	num_bins = 25
	angles = [float(data_line[3]) for data_line in data_lines]
	avg_samples_per_bin = len(angles) // num_bins
	hist, bin_edges = np.histogram(angles, bins=num_bins)

	# cap at 700
	cap = int(avg_samples_per_bin * 1.8)
	data_lines_capped = np.array([], dtype=data_lines.dtype).reshape(0, data_lines.shape[1])
	for idx, h in enumerate(list(hist)):
		# mask of angles that fall in this bin
		mask = (angles >= bin_edges[idx]) & (angles < bin_edges[idx+1])

		# cap data_lines to 700 per bin
		data_lines_capped = np.concatenate([data_lines_capped, data_lines[mask][:cap]])

	print('histogram equalization: size after', data_lines_capped.shape)

	return shuffle(data_lines_capped)



def preprocess_img(img):
	# crop off top and bottom portion of image which do not contain the road
	img = img[50:140, :, :]

	# resize
	img = cv2.resize(img, (64,64))

	# change to YUV space as suggested in the Nvidia paper
	return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)


def augment_img(data_line):

	# choose camera image
	camera = np.random.randint(3)
	angle_offset = 0
	# left image
	if (camera == 1):
		angle_offset = 0.25
	# right image
	elif (camera == 2):
		angle_offset = -0.25

	image = cv2.imread('data/'+data_line[camera].strip())
	angle = float(data_line[3]) + angle_offset

	# flip images and invert steering angle
	flip = np.random.random()
	if (flip > 0.5):
		image = cv2.flip(image, 1) 
		angle *= -1

	# TODO: brightness and contrast
	# TODO: vertical shift

	return image, angle


def data_generator(data_lines, batch_size=32):
	# num_samples = len(data_lines)
	idx = 0
	idx_global = 0
	angles_all = []
	while 1:	
		images = []
		angles = []
		for i in range(batch_size):
			random_index = np.random.randint(0,len(data_lines))

			image, angle = augment_img(data_lines[random_index])

			# # choose abs(angles) < 0.08 with 35% probability
			# prob = np.random.random()
			# while(abs(angle) < 0.06 and prob < 0.30):
			# 	image, angle = augment_img(data_lines[random_index])

			image = preprocess_img(image)

			images.append(image)
			angles.append(angle)

		X = np.array(images)
		y = np.array(angles)

		# record all angles
		idx = idx + 1
		angles_all.extend(angles)
		# print('idx=',idx,' angles=',len(angles_all))
		if (idx == (20000 // batch_size)-2):
			print('# of angles=', len(angles_all))
			hist, bin_edges = np.histogram(angles_all, bins=20)
			print('hist', hist)
			print('bin', bin_edges)

			plt.hist(angles_all, bins=20)
			plt.title('steering angles batch')
			s = 'hist_batch'+str(idx_global)+'.png'
			plt.savefig(s)
			plt.show()
			idx = 0
			idx_global = idx_global + 1
		yield shuffle(X, y)


# visualize training data for debugging
def visualize_train_data(train_samples):
	print('saving train data...')

	gen = data_generator(train_samples, batch_size=32)
	X_batch,y_batch = next(gen)
	for i in range(20):
		# print(i)
		X = cv2.cvtColor(X_batch[i], cv2.COLOR_YUV2BGR)
		cv2.imwrite("train/train" + str(i) + "_" + str(y_batch[i]) + ".jpg", X)
		# angles_all = np.concatenate([angles_all, y])




def train(train_generator, validation_generator):
	print('Training...')
	start = time.time()

	# build model
	model = Sequential()
	
	# normalize and mean center images
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64,64,3)))
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
	model.add(Dense(100))
	model.add(ELU())
	model.add(Dense(50))
	model.add(ELU())
	model.add(Dense(1))

	# train model
	model.compile(loss='mse', optimizer='adam')											
	model_history = model.fit_generator(train_generator, samples_per_epoch=20000, 
		validation_data=validation_generator, nb_val_samples=5000, nb_epoch=3, verbose=1)
	model.save('model.h5')

	print('training completed in ', time.time() - start, 's')
	print('training loss',model_history.history['loss'])
	print('validation loss',model_history.history['val_loss'])

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



def main():
	# load data
	data_lines = load_data()

	# preprocess data: histogram equalization (remove data biased towards any particular steering angle)
	data_lines = preproc_data(data_lines)

	# generate training and validation data
	train_samples, validation_samples = train_test_split(data_lines, test_size=0.2)
	# debug: visualize train data
	visualize_train_data(train_samples)

	batch_size = 32
	train_generator = data_generator(train_samples, batch_size=batch_size)
	validation_generator = data_generator(validation_samples, batch_size=batch_size)

	# train model
	train(train_generator, validation_generator)



if __name__ == '__main__':
	main()

