import cv2
import csv
from keras.models import load_model
from sklearn.utils import shuffle


# predict some images

samples = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(('data/'+line[0], float(line[3]))) # center image
		# samples.append(('data/'+line[1].strip(), float(line[3])+0.25)) # left image
		# samples.append(('data/'+line[2].strip(), float(line[3])-0.25)) # right image

samples = shuffle(samples)

# test accuracy
model = load_model('model.h5')

print('testing model...')

for i in range(30):
	sample = samples[i]
	img = cv2.imread(sample[0])
	img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	angle_pred = model.predict(img[None, :, :, :])
	print(sample[0])
	print('angle=',sample[1], ' angle_pred=', angle_pred)
	cv2.imwrite("predict" + str(i) + "_" + str(sample[1]) + "_" + str(angle_pred) + ".jpg", img)