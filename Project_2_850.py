import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib

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