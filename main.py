# import libraries
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# create list to store data
label = []
data = []

# get image data
for img in os.listdir('cell_images/Parasitized'):
    try:
        image = cv2.imread(os.path.join('cell_images/Parasitized', img), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (100, 100))
        data.append(image)
        label.append(np.array(1))
    except Exception as e:
        pass

for img in os.listdir('cell_images/Uninfected'):
    try:
        image = cv2.imread(os.path.join('cell_images/Uninfected', img), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (100, 100))
        data.append(image)
        label.append(np.array(2))
    except Exception as e:
        pass

# split data into test, train, and validation
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, train_size=0.7)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, train_size=0.5)

# convert to one dimensional array
X_train = np.concatenate(X_train).ravel().tolist()

# create model
model = Sequential()

model.add(Dense(100, input_shape=(10000,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size='None', epochs=30, verbose=2)
