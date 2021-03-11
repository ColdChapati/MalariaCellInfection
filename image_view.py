import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

for img in os.listdir('cell_images/Uninfected'):
    image = cv2.imread(os.path.join('cell_images/Uninfected', img))
    image = cv2.resize(image, (25, 25))
    plt.imshow(image)
    plt.show()
