import cv2
import csv
from keras.models import load_model


# predict some images

samples = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(('data/'+line[0], float(line[3]))) # center image
		samples.append(('data/'+line[1].strip(), float(line[3])+0.25)) # left image
		samples.append(('data/'+line[2].strip(), float(line[3])-0.25)) # right image
		break

# test accuracy
model = load_model('model_test.h5')

print('testing model...')

for sample in samples:
	img = cv2.imread(sample[0])
	img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	angle_pred = model.predict(img[None, :, :, :])
	print(sample[0])
	print('angle=',sample[1], ' angle_pred=', angle_pred)