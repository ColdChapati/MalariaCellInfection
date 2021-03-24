# import libraries
from keras.models import load_model
import numpy as np
from keras.models import Sequential
import matplotlib.pyplot as plt
import os
import cv2

# get image ready
image = os.path.join('cell_images', 'test_image4.png')
image = cv2.imread(image)
image = cv2.resize(image, (25, 25))
image = np.array(image).reshape(-1,25,25,3)

# load model
model = load_model('malaria_weights.h5')

# predict test image
prediction = model.predict(image)

# print result
print(prediction)
