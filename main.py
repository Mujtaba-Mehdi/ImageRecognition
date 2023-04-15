import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10 #loads the dataset

(images_train, labels_train), (images_test, labels_test) = cifar10.load_data()

print('images_train shape:', images_train.shape)
print('labels_train shape:', labels_train.shape)
print('images_test shape:', images_test.shape)
print('labels_test shape:', labels_test.shape)

i = 0 #index for image selection and image class identifier

classify = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] #array to categorize all the images in their own categories accordingly to the CIFAR-10 dataset

print('Image Class:', classify[labels_train[i][0]]) #this will print the image class category to the user

img = plt.imshow(images_train[i])
plt.show() #this will show the image as a test to show if initial images will load

#This will normalize the images to have scaled down values between 0 - 1 rather than 0 - 255
images_train = images_train.astype('float32')
images_test = images_test.astype('float32')
images_train = images_train / 255
images_test = images_test / 255

labels_train_converted = to_categorical(labels_train)
labels_test_converted = to_categorical(labels_test)

training_models = Sequential(
			[Conv2D(filters=32, kernel_size=(5, 5), activation = 'relu', padding='same', input_shape = (32, 32, 3)),
			 MaxPooling2D(pool_size = (2,2)),
			 Conv2D(filters=64, kernel_size=(5, 5), activation = 'relu', padding='same'),
			 MaxPooling2D(pool_size = (2,2)),
			 Conv2D(filters=128, kernel_size=(5, 5), activation = 'relu', padding='same'),
			 MaxPooling2D(pool_size = (2,2)),
			 Flatten(),
			 Dense(512, activation = 'relu'),
			 Dropout(0.5),
			 Dense(256, activation = 'relu'),
			 Dropout(0.5),
			 Dense(128, activation = 'relu'),
			 Dropout(0.5),
			 Dense(64, activation = 'relu'),
			 Dropout(0.5),
			 Dense(10, activation = 'softmax')
			]
)

training_models.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

fit_model = training_models.fit(
    images_train,
    labels_train_converted,
    batch_size=500,
    epochs=100,
    validation_split=0.2,
)

training_models = Sequential(
			[Conv2D(filters=32, kernel_size=(3, 3), activation = 'relu', padding='same', input_shape = (32, 32, 3)),
			 MaxPooling2D(pool_size = (2,2)),
			 Conv2D(filters=64, kernel_size=(3, 3), activation = 'relu', padding='same'),
			 Conv2D(filters=64, kernel_size=(3, 3), activation = 'relu', padding='same'),
			 MaxPooling2D(pool_size = (2,2)),
			 Flatten(),
			 Dense(128, activation = 'relu'),
			 Dropout(0.5),
			 Dense(10, activation = 'softmax')
			]
)

history = training_models.fit(
    images_train,
    labels_train_converted,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
)

results = training_models.evaluate(images_test, labels_test_converted, batch_size=32)
print("Accuracy: %.1f%%" % (results[1]*100))