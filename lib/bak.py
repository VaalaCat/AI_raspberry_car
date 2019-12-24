import cv2
import numpy as np
import pygame
import os
import time
import serial
from simple_pid import PID

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=0)
ser.write(str.encode('R000F000R0000'))

center = 320
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

towards = 0

pid = PID(0.15, 0.1, 0.1, setpoint=1)
pidw = PID(0.1, 0.1, 0.1, setpoint=1)

LineThreshold = 30

roadfile = open('road', 'r')
road = roadfile.read()

#判断是否为直线
def s_line(temp):
    if temp < 100:
        return True
    return False

#格式化传输数据
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

now = 0

while (True):

    #图像预处理
    ret, frame = cap.read()
    frame[:, 639] = frame[:, 637]
    frame[:, 638] = frame[:, 637]
    frame[:, 0] = frame[:, 2]
    frame[:, 1] = frame[:, 2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #变量初始化
    cur = 340
    x = []
    y = []
    delta = []

    #隔20行扫描
    while cur >= 20:
        row = gray[cur]
        diff = np.diff(row.tolist())

        #找到每行的突变点
        edge = np.where(np.logical_or(
            diff == np.max(diff), diff == np.min(diff)))
        #edge = np.where(np.logical_or(diff >= LineThreshold, diff <= -LineThreshold))

        #获得黑线相关数据并保存
        if len(edge) > 0 and len(edge[0]) > 1:
            cv2.circle(frame, (edge[0][0], cur), 2, (255, 0, 0), -1)
            cv2.circle(frame, (edge[0][1], cur), 2, (255, 0, 0), -1)

            middle = int((edge[0][0] + edge[0][1]) / 2)
            cv2.circle(frame, (middle, cur), 2, (0, 0, 255), -1)

            delta.append(abs(edge[0][0] - edge[0][1]))
            x.append(middle)
            y.append(cur)

        cur -= 20
    
    #获取转向方向
    flag = 0
    for i in range(0,len(delta)):
        if s_line(delta[i]) != True:
            if x[i] == 320:
                continue
            flag = abs(x[i] - 320) / (x[i] - 320)
            break
    
    #更新方向
    #只有当转向完成后才将方向置为0
    if flag == 0 and towards != 0:
        direction = 0
        towards = 0
        now += 1

    #只有在直线时才更新转向方向
    if flag != 0 and towards == 0:
        towards = flag
        direction = 0
    
    #画出线段(调试用)       
    #cv2.line(frame, (x[0], y[0]), (x[17], y[17]), (0, 255, 0))
    #for i in range(1, 18):
    #    cv2.line(frame, (x[i], y[i]), (x[i - 1], y[i - 1]), (0, 255, 0))

    direction = 320 - x[7]
    direction = pid(direction)

    #处理PID上限防止积分溢出
    if direction > 100:
        direction = 100
    if direction < -100:
        direction = -100

    cur = str(int((direction / 320) * 255))

    if towards == 0:
        #直线行驶数据
        data = (pos(cur, 4, 'L,R') + 'F050' + 'R0000')
    else:
        #转向时数据
        if road[now] == 'F':
            data = 'R000F030' + road[now] + '0000'
        else:
            data = 'R000F030' + road[now] + '2000'

    time.sleep(0.05)
    print(data)
    ser.write(bytes(data, encoding='ascii'))
    if road[now] == 'N':
        ser.write(bytes('R000F000R0000', encoding='ascii'))
    cv2.imshow("??", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        ser.write(bytes('R000F000R0000', encoding='ascii'))
        break
