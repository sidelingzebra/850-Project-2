import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from keras.layers import LeakyReLU

import keras

import tensorboard

import datetime
import keras_tuner


batch_size=32
img_height = 500
img_width = 500
epochs=12
num_classes = 3


#Creating TFs from relative locations
train_ds = keras.preprocessing.image_dataset_from_directory(
    directory='Data/train/',
    labels='inferred',
    label_mode='categorical',
    seed=123,
    batch_size=batch_size,
    image_size=(img_height, img_width))

valid_ds = keras.preprocessing.image_dataset_from_directory(
    directory='Data/valid/',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_height, img_width))

test_ds = keras.preprocessing.image_dataset_from_directory(
    directory='Data/test/',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_height, img_width))


class_names = train_ds.class_names
print(class_names)


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


#Normalize
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(train_ds))
valid_ds = valid_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(valid_ds))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(test_ds))


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)



#Early Exit
# callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)



model = tf.keras.Sequential([
  tf.keras.layers.Resizing(250,250),
  

  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(), 
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(),

  




  
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Dense(num_classes,activation='softmax')
])



# initial_learning_rate = 0.1
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=100000,
#     decay_rate=0.96,
#     staircase=True)

model.compile(
  optimizer='Adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'])

# keras.optimizers.SGD(learning_rate=lr_schedule)

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

history=model.fit(
  train_ds,
  validation_data=valid_ds,
  epochs=epochs)
# callbacks=[callback]

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

model.save("Model_9.keras")






