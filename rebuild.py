import tensorflow as tf
import numpy as np
import random
import cv2
import math
import time
names = ['straight', 'straight and left', 'straight and right', 'left', 'right', 'left and right', 'cross','stop']

import view
import RC
model = tf.keras.models.load_model('final.h5')

turn = False
t = []

while True:
    #图像预处理
    gray = view.pre_process('F')
    error, ang = view.detect(gray)
    dst = view.bin(gray)
    cv2.imshow("???", dst)
    #RC.run(RC.get_xbox_data(), False)

    #只有当未转向时才直行
    '''
    if turn == False:
        RC.run('R000F030R0000', False)'''

    t.append(dst)
    t = np.array(t)
    ans = np.argmax(model.predict(t))
    t=[]

    #检测到非直线且未转向时减速判断
    temp = []
    if names[ans] != 'straight' and turn == False:
        data = view.pid_run(error, ang)
        RC.run(data, False)
        while len(temp) <= 2:
            t.append(dst)
            t = np.array(t)
            temp.append(names[np.argmax(model.predict(t))])
            t=[]
        print("max is:" + view.get_max(temp))
    
    if names[ans] == 'straight':
        data = view.pid_run(error, ang)
        RC.run(data, False)
        turn = False
        
    temp = view.get_max(temp)

    #未转向时才转向
    if turn == False:
        if temp == 'right':
            RC.run('R000F000R1700', False)
            turn = True
        elif temp == 'straight':
            pass
        else:
            RC.run('R000F000L1700', False)
            turn = True

    tf.keras.backend.clear_session()
    name, but = RC.get_buttom()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        RC.run('R000F000R0000',False)
        break
    if but[8] == 1:
        RC.run('R000F000R0000',False)
        break

time.sleep(0.5)

while True:
    RC.run(RC.get_xbox_data(), False)
    r, but = RC.get_buttom()

    if but[8] == 1:
        RC.run('R000F000R0000',False)
        break
    

