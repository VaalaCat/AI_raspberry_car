import cv2
import numpy as np
from simple_pid import PID

pid = PID(0.05, 0.03, 0.01, setpoint=1)
pidw = PID(0.05, 0.05, 0.1, setpoint=1)

center = 320

LineThreshold = 30
LineWidth = 50

f_speed = 40

capf = cv2.VideoCapture(0)
capf.set(3, 640)
capf.set(4, 320)#480)
capl = cv2.VideoCapture(1)
capl.set(3, 640)
capl.set(4, 320)#480)
capr = cv2.VideoCapture(2)
capr.set(3, 640)
capr.set(4, 480)
capb = cv2.VideoCapture(3)
capb.set(3, 640)
capb.set(4, 480)

names = ['straight', 'straight and left', 'straight and right', 'left', 'right', 'left and right', 'cross', 'stop']


#判断是否为直线
def s_line(width):
    if width < LineWidth:
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

def get_max(obj):
    frq = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in obj:
        frq[names.index(i)] += 1
    frq = np.array(frq)
    return names[np.argmax(frq)]

#识别传入图像内是否有线并返回线两行像素线中点
def detect(gray):
    Blackline = cv2.inRange(gray, 0, 60)
    kernel = np.ones((3, 3), np.uint8)

    Blackline = cv2.erode(Blackline, kernel, iterations=5)
    Blackline = cv2.dilate(Blackline, kernel, iterations=9)

    img_blk, contours_blk, hierarchy_blk = cv2.findContours(Blackline.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    blackbox=(0,0),(0,0),0
    (x_min, y_min), (w_min, h_min), ang = blackbox

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

    return (error, ang)


#按传入摄像头标签获取并预处理图像
def pre_process(CapCur):
    if CapCur == 'F':
        ret, frame = capf.read()
    elif CapCur == 'L':
        ret, frame = capl.read()
    elif CapCur == 'B':
        ret, frame = capb.read()
    else:
        ret, frame = capr.read()

    #图像边缘处理
    frame[:, 639] = frame[:, 637]
    frame[:, 638] = frame[:, 637]
    frame[:, 0] = frame[:, 2]
    frame[:, 1] = frame[:, 2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def bin(image):
    dst = cv2.resize(image, (28, 28))
    retval, dst = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU)
    return dst

def pid_run(error, ang):
    error = -error
    error = int(pid(error))
    if error > 100:
        error = 100
    if error < -100:
        error = -100

    ang = pidw(ang)
    if ang > 100:
        ang = 100
    if ang < -100:
        ang = -100

    ang = str(int(ang * 10))
    cur = str(int((error / 320) * 255))
    data = (pos(cur, 4, 'L,R') + pos(str(f_speed),4,'B,F') + pos(ang,5,'L,R'))
    return data
