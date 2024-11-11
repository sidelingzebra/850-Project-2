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



batch_size=32
img_height = 500
img_width = 500
epochs=10
num_classes = 3

test_ds = keras.preprocessing.image_dataset_from_directory(
    directory='Data/test/',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_height, img_width))

class_names = test_ds.class_names
print(class_names)

model = keras.models.load_model("Model_6.keras")








img1=keras.utils.load_img(path='Data/test/crack/test_crack.jpg',
                           target_size=(500,500,3))
img1_a=tf.keras.utils.img_to_array(img1)
img1_a=img1_a/255
img1_a=tf.expand_dims(img1_a, 0)

predictions1 = model.predict(img1_a)
print(predictions1)
score1 = tf.nn.softmax(predictions1[0])

print("\nImage 1:")
print("  Actual Class: Crack")
print("  Predicted Class: {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score1)], 100 * np.max(score1)))
print("    Crack: {:.2f}% ".format(predictions1[0,0]*100))
print("    Missing Head: {:.2f}% ".format(predictions1[0,1]*100))
print("    Paint Off: {:.2f}% \n".format(predictions1[0,2]*100))

img2=keras.utils.load_img(path='Data/test/missing-head/test_missinghead.jpg',
                           target_size=(500,500,3))
img2_a=tf.keras.utils.img_to_array(img2)
img2_a=img2_a/255
img2_a=tf.expand_dims(img2_a, 0)

predictions2 = model.predict(img2_a)
print(predictions2)
score2 = tf.nn.softmax(predictions2[0])
print("\nImage 2:")
print("  Actual Class: Missing Head")
print("  Predicted Class: {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score2)], 100 * np.max(score2)))
print("    Crack: {:.2f}% ".format(predictions2[0,0]*100))
print("    Missing Head: {:.2f}% ".format(predictions2[0,1]*100))
print("    Paint Off: {:.2f}% \n".format(predictions2[0,2]*100))

img3=keras.utils.load_img(path='Data/test/paint-off/test_paintoff.jpg',
                           target_size=(500,500,3))
img3_a=tf.keras.utils.img_to_array(img3)
img3_a=img3_a/255
img3_a=tf.expand_dims(img3_a, 0)

predictions3 = model.predict(img3_a)
print(predictions3)
score3 = tf.nn.softmax(predictions3[0])
print("\nImage 3:")
print("  Actual Class: Paint Off")
print("  Predicted Class: {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score3)], 100 * np.max(score3)))
print("    Crack: {:.2f}% ".format(predictions3[0,0]*100))
print("    Missing Head: {:.2f}% ".format(predictions3[0,1]*100))
print("    Paint Off: {:.2f}% \n".format(predictions3[0,2]*100))

fig=plt.figure(figsize=(3, 3))
ax=fig.add_subplot()
ax.imshow(img1)
ax.axis('off')
plt.figtext(1,1,"Actual Class: Crack")
plt.figtext(1,0.9,"Predicted Class: {}".format(class_names[np.argmax(score1)])) 
plt.figtext(1,0.5,"Crack: {:.2f}% ".format(predictions1[0,0]*100))
plt.figtext(1,0.4,"Missing Head: {:.2f}% ".format(predictions1[0,1]*100))
plt.figtext(1,0.3,"Paint Off: {:.2f}% ".format(predictions1[0,2]*100))
plt.show()


fig=plt.figure(figsize=(3, 3))
ax=fig.add_subplot()
ax.imshow(img2)
ax.axis('off')
plt.figtext(1,1,"Actual Class: Missing Head")
plt.figtext(1,0.9,"Predicted Class: {}".format(class_names[np.argmax(score2)])) 
plt.figtext(1,0.5,"Crack: {:.2f}% ".format(predictions2[0,0]*100))
plt.figtext(1,0.4,"Missing Head: {:.2f}% ".format(predictions2[0,1]*100))
plt.figtext(1,0.3,"Paint Off: {:.2f}% ".format(predictions2[0,2]*100))
plt.show()

fig=plt.figure(figsize=(3, 3))
ax=fig.add_subplot()
ax.imshow(img3)
ax.axis('off')
plt.figtext(1,1,"Actual Class: Paint Off")
plt.figtext(1,0.9,"Predicted Class: {}".format(class_names[np.argmax(score3)])) 
plt.figtext(1,0.5,"Crack: {:.2f}% ".format(predictions3[0,0]*100))
plt.figtext(1,0.4,"Missing Head: {:.2f}% ".format(predictions3[0,1]*100))
plt.figtext(1,0.3,"Paint Off: {:.2f}% ".format(predictions3[0,2]*100))
plt.show()
