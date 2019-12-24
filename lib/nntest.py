import tensorflow as tf
import numpy as np
import random
import cv2
import math
import time
names = ['straight', 'straight and left', 'straight and right', 'left', 'right', 'left and right', 'cross','stop']

import view
model = tf.keras.models.load_model('final.h5')

turn = False
t = []

while True:

    gray = view.pre_process('L')
    dst = view.bin(gray)

    t.append(dst)
    t = np.array(t)
    ans = np.argmax(model.predict(t))
    t = []
    print(names[ans])
    cv2.imshow("???", cv2.resize(dst,(640,320)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break