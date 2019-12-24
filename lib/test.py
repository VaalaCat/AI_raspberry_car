import cv2
import numpy as np
import pygame
import os
import time
import serial
'''
from simple_pid import PID

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=0)
ser.write(str.encode('R000F000R0000'))
'''
center = 320
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 320)

towards = 0
'''
pid = PID(0.05, 0.07, 0.1, setpoint=1)
pidw = PID(0.05, 0.05, 0.1, setpoint=1)
'''
LineThreshold = 60


def s_line(temp):
    if temp < 50:
        return True
    return False


def pos(x, y, c):
    if x[0] == '-':
        x = x.replace('-', '')
        x = c[0] + x
    else:
        x = c[2] + x
    if (len(x) != y):
        delta = y - len(x)
        s = ''
        for i in range(0, delta):
            s += '0'
        x = x[0] + s + x[1:]
    return x


while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    Blackline = cv2.inRange(gray, 0, 60)
    kernel = np.ones((3, 3), np.uint8)

    Blackline = cv2.erode(Blackline, kernel, iterations=5)
    Blackline = cv2.dilate(Blackline, kernel, iterations=9)

    contours_blk, hierarchy_blk = cv2.findContours(Blackline.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    blackbox=(0,0),(0,0),0
    if len(contours_blk) > 0:
        blackbox = cv2.minAreaRect(contours_blk[0])

    (x_min, y_min), (w_min, h_min), ang = blackbox

    if ang < -45:
        ang = 90 + ang
        if w_min < h_min and ang > 0:
            ang = (90-ang)*-1
        if w_min > h_min and ang < 0:
            ang = 90 + ang

    setpoint = 320
    error = int(x_min - setpoint)
    ang = int(ang)
    box = cv2.boxPoints(blackbox)
    box = np.int0(box)

    cv2.drawContours(frame, [box], 0, (0, 0, 255), 3)
    cv2.putText(frame, str(ang), (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, str(error), (10, 320),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.line(frame, (int(x_min), 200), (int(x_min), 250), (255, 0, 0), 3)
    cv2.imshow("???",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''
    error = -error
    error = int(pid(error))
    if error > 100:
        error = 100
    if error < -100:
        error = -100

    ang = pid(ang)
    if ang > 100:
        ang = 100
    if ang < -100:
        ang = -100
    ang = str(int(ang * 10))
    cur = str(int((error / 320) * 255))

    if towards == 0:
        # 直线行驶数据
        data = (pos(cur, 4, 'L,R') + 'F030' + pos(ang,5,'L,R'))
    else:
        # 转向时数据
        data = 'R000F030' + road[now] + '2000'
        now += 1

    time.sleep(0.05)
    print(data)
    ser.write(bytes(data, encoding='ascii'))'''
    
