import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential

import keras


#Getting Train Image Count
X_train=pathlib.Path()/'Data'/'train'
image_count= len(list(X_train.glob('*/*.jpg')))
print("Train Images:",image_count)

#Getting Validation Image Count
X_valid=pathlib.Path()/'Data'/'valid'
image_count = len(list(X_valid.glob('*/*.jpg')))
print("Validation Images:", image_count)

#Getting Test Image Count
X_test=pathlib.Path('Data/test')
image_count = len(list(X_test.glob('*/*.jpg')))
print("Test Images:", image_count)

#Creating TFs from relative locations
train_ds = keras.preprocessing.image_dataset_from_directory(
    directory='Data/train/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(500, 500))

test_ds = keras.preprocessing.image_dataset_from_directory(
    directory='Data/test/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(500, 500))

valid_ds = keras.preprocessing.image_dataset_from_directory(
    directory='Data/valid/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(500, 500))

#Confirming Class Names
class_names = train_ds.class_names
print(class_names)

#Starting the model
model = Sequential()

#Rescale Layer
model.add(layers.Rescaling(1./255, input_shape=(500, 500, 3)))

#Conv2D & Pooling Layer 1
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

#Conv2D & Pooling Layer 2
model.add(layers.Conv2D(64, (3, 3), activation='leaky_relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

#Conv2D & Pooling Layer 3
model.add(layers.Conv2D(96, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

#Flatten Layer
model.add(layers.Flatten())

#Dense Layer 1
model.add(layers.Dense(32, activation='relu'))


#Final Dense Layer
model.add(layers.Dense(3, activation='softmax'))


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(train_ds, epochs=1, batch_size=32,validation_data=valid_ds)

test_loss, test_acc = model.evaluate(test_ds)
print(f'Test accuracy: {test_acc:.4f}')

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()