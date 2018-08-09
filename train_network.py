import nvidia_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as Shuffle
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
import PIL

csv_file = 'data/driving_log.csv'
BATCH_SIZE = 32
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
image_size = (160,320)


def _load_img(path):
	return img_to_array(load_img(relative_path(path), target_size=image_size))

def random_flip(frames, angle):
	if np.random.random() < 0.5:
		frames = np.array(frames)[:, :, ::-1]
		angle = angle*(-1)
	return frames, angle

def relative_path(path):
	return os.path.join(*path.split('/')[-3:])

def generator(samples, batch_size, train=False):
	while True:
		if train: samples = Shuffle(samples)

		batch_frames=[]
		batch_angles=[]

		for sample in samples:
			center, left, right, angle, _, _, _ = sample

			center_frame = _load_img(center)
			left_frame = _load_img(left)
			right_frame = _load_img(right)
			angle = float(angle)

			frames = [center_frame, left_frame, right_frame]
			correction = 0.25 if angle!=0 else 0.05
			angles = np.array([angle, angle+correction, angle-correction])

			frames, angles = random_flip(frames, angles)

			batch_frames.extend(frames)
			batch_angles.extend(angles)

			if len(batch_frames) == batch_size*3:
				X = np.array(batch_frames)
				Y = np.array(batch_angles)
				batch_frames = []
				batch_angles = []
				yield X, Y

with open(csv_file, 'r') as csvreader:
	samples = [x.strip().split(',') for x in csvreader.readlines()]

train_samples, test_samples = train_test_split(samples, test_size=0.05, random_state=42)
print('Train Samples: ', len(train_samples))
print('Validation Samples: ', len(test_samples))

train_generator = generator(train_samples, batch_size=BATCH_SIZE, train=True)
test_generator  = generator(test_samples, batch_size=BATCH_SIZE, train=False)

model = nvidia_model.build_model()

model.compile(optimizer='adam', loss="mse", metrics=['acc'])

if not os.path.exists('checkpoints'):
	os.makedirs('checkpoints')

model_checkpoint = ModelCheckpoint(filepath="./checkpoints/weights_{epoch:02d}-{val_loss:.4f}.h5", 
									monitor='val_loss', verbose=1, save_best_only=True, mode='min')

print(model.summary())

model.fit_generator(train_generator, 
                    validation_data=test_generator,
                    steps_per_epoch=np.ceil(len(train_samples)/BATCH_SIZE),
                    validation_steps=np.ceil(len(test_samples)/BATCH_SIZE),
                    epochs=100,
                    callbacks=[model_checkpoint])