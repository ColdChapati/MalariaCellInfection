# import libraries
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# create list to store data
label = []
data = []

# get image data
for img in os.listdir('cell_images/Parasitized'):
    try:
        image = cv2.imread(os.path.join('cell_images/Parasitized', img))
        image = cv2.resize(image, (50, 50))
        data.append(image)
        label.append(1)
    except Exception as e:
        pass

for img in os.listdir('cell_images/Uninfected'):
    try:
        image = cv2.imread(os.path.join('cell_images/Uninfected', img))
        image = cv2.resize(image, (50, 50))
        data.append(image)
        label.append(-1)
    except Exception as e:
        pass

# convert and reshape data
data = np.array(data).reshape(-1,50,50,3)
label = np.array(label)

# split data
X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=0.9)

# create model
model = Sequential()

# first layer
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(50,50,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

# second layer
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.15))

# third layer
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))

# output layer
model.add(Dense(1, activation='sigmoid'))

# print model summary
model.summary()

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
results = model.fit(X_train, y_train, batch_size=25, epochs=10, verbose=2, validation_split=0.1)

# save model
model.save('malaria_weights.h5')

# plot accuracy vs epochs
plt.plot(results.history['val_accuracy'], label='val_accuracy', color='lightpink')
plt.plot(results.history['accuracy'], label='accuracy', color='c')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()

# print loss vs epochs
plt.plot(results.history['val_loss'], label='val_loss', color='lightpink')
plt.plot(results.history['loss'], label='loss', color='c')
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.show()
