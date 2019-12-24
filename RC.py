import pygame
import os
import time
import serial

but = {0: 'bta', 1: 'btb', 2: 'btx', 3: 'bty', 4: 'lt', 5: 'rt', 8: 'xbox', 9: 'la', 10: 'ra',
        'bta': 0, 'btb': 1, 'btx': 2, 'bty': 3, 'lt': 4, 'rt': 5, 'xbox': 8, 'la': 9, 'ra': 10}

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=0)
ser.write(str.encode('R000F000R0000'))
os.putenv('SDL_VIDEODRIVER', 'fbcon')
pygame.display.init()
pygame.joystick.init()
pygame.joystick.Joystick(0).init()


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


def get_buttom():
    ans = []
    pygame.event.pump()
    for i in range(0, 11):
        ans.append(pygame.joystick.Joystick(0).get_button(i))
    return (but, ans)


def get_xbox_data():
    pygame.event.pump()
    lx = str(int(pygame.joystick.Joystick(0).get_axis(0) * 255+0.5))
    ly = str(int(pygame.joystick.Joystick(0).get_axis(1) * 255+0.5))
    rx = str(int(pygame.joystick.Joystick(0).get_axis(3) * 9999 + 0.5))
    data = (pos(lx, 4, 'L,R') + pos(ly, 4, 'F,B') + pos(rx, 5, 'L,R'))
    return data

# 传输数据到arduino


def run(speed,prt):
    time.sleep(0.05)
    if prt != False:
        print(speed)
    ser.write(bytes(speed, encoding='ascii'))
