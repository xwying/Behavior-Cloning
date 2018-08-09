from keras import layers
from keras.models import Sequential, Model
import tensorflow as tf
import keras.backend as K

def build_model():
	x_in = layers.Input((160,320,3))
	x = layers.Cropping2D(cropping=((75,25), (0,0)))(x_in)
	# x = layers.Lambda(lambda x_img: tf.image.resize_images(x_img, (66,200)), name='resize')(x)
	x = layers.Lambda(lambda x_img:x_img/127.5 - 1.0, name='normalize')(x)
	x = layers.Conv2D(24, (5,5), strides=(2,2), activation='relu')(x)
	x = layers.Conv2D(36, (5,5), strides=(2,2), activation='relu')(x)
	x = layers.Conv2D(48, (5,5), strides=(2,2), activation='relu')(x)
	x = layers.Conv2D(64, (3,3), strides=(1,1), activation='relu')(x)
	# x = layers.Conv2D(64, (3,3), strides=(1,1), activation='relu')(x)
	x = layers.Flatten()(x)
	x = layers.Dropout(0.4)(x)
	x = layers.Dense(100, activation='relu')(x)
	x = layers.Dropout(0.4)(x)
	x = layers.Dense(50, activation='relu')(x)
	x = layers.Dropout(0.4)(x)
	x = layers.Dense(10, activation='relu')(x)
	x = layers.Dense(1)(x)

	return Model(inputs=x_in, outputs=x)

if __name__ == '__main__':
	model = build_model()
	print(model.summary())