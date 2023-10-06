import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.layers import BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# Download EMNIST data
from emnist import extract_training_samples, extract_test_samples
train_images, train_labels = extract_training_samples('balanced')
test_images, test_labels = extract_test_samples('balanced')
print(train_images.shape)
print(test_images.shape)

# Preprocessing
train_images, test_images = train_images / 255.0, test_images / 255.0 # Normalize values to be between 0 and 1
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Class names (47 total)
class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

# View a test image and its class
index = 3
plt.imshow(train_images[index] , cmap=plt.cm.binary)
print(train_labels[index])
plt.xlabel(class_name[train_labels[index]])
plt.show()

#CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), padding='same', activation='tanh', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))
model.add(layers.Conv2D(48, (5, 5), padding='same', activation='tanh'))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))
model.add(layers.Conv2D(64, (5, 5), padding='same', activation='tanh'))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))
model.add(layers.Flatten()),
model.add(layers.Dense(256, activation='tanh')),
model.add(layers.Dense(96, activation='tanh')),
model.add(BatchNormalization())
model.add(layers.Dense(47, activation='softmax'))
model.summary()

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# About 0.91 accuracy after 10 epochs
history = model.fit(train_images,
                    train_labels,
                    epochs=10,
                    batch_size=128,
                    verbose=1,
                    validation_split=0.2,
                    validation_data=(test_images, test_labels))

# Loading images and processing into format
def load(file):
 image = load_img(file, grayscale=True, target_size=(28, 28))
 image = np.invert(image)
 image = img_to_array(image)
 image = image.reshape(-1, 28, 28, 1)
 image = image.astype('float32') / 255.0
 return image

# Predict an image file
image = load('s.png')
plt.imshow(image.reshape(28, 28, 1), cmap = plt.cm.binary)
plt.show()
prediction = model.predict(image)
digit = np.argmax(prediction) # Choose class with highest probability
print(digit)
print(class_name[digit])