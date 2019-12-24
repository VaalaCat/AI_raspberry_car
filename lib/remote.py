import pygame
import os
import time
import serial
import RC

ser = serial.Serial('/dev/ttyACM0',9600,timeout=0)
ser.write(str.encode('R000F000R0000'))

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
	
os.putenv('SDL_VIDEODRIVER', 'fbcon')
pygame.display.init()
pygame.joystick.init()
pygame.joystick.Joystick(0).init()
	# Prints the joystick's name
JoyName = pygame.joystick.Joystick(0).get_name()
print("Name of the joystick:")
print(JoyName)
# Gets the number of axes
JoyAx = pygame.joystick.Joystick(0).get_numaxes()
print("Number of axis:")
print(JoyAx)
JoyBt = pygame.joystick.Joystick(0).get_numbuttons()
print("Number of buttons:")
print(JoyBt)
JoyNb = pygame.joystick.Joystick(0).get_numhats()
print("Number of balls:")
print(JoyNb)
# Prints the values for axis0
bta = 0
btb = 1
btx = 2
bty = 3
lt = 4
rt = 5
xbox = 8
la = 9
ra = 10
'''
while 1:
	pygame.event.pump()
	for i in range(0, 11):
		if (pygame.joystick.Joystick(0).get_button(i) == True):
			print(i)
			time.sleep(0.2) 
	#print(pygame.joystick.Joystick(0).get_hat(0))
'''


while True:
	pygame.event.pump()
	lx = str(int(pygame.joystick.Joystick(0).get_axis(0) * 255+0.5))
	ly = str(int(pygame.joystick.Joystick(0).get_axis(1) * 255+0.5))
	rx = str(int(pygame.joystick.Joystick(0).get_axis(3) * 9999+0.5))
	#ry = pygame.joystick.Joystick(0).get_axis(4)
	#lz = pygame.joystick.Joystick(0).get_axis(2)
	#rz = pygame.joystick.Joystick(0).get_axis(5)
	#print(lx, ly, rx, ry)
	#print(lz, rz)
	data=(pos(lx, 4, 'L,R')+ pos(ly, 4, 'F,B')+ pos(rx, 5, 'L,R'))
	time.sleep(0.05)
	print(pos(lx, 4, 'L,R')+ pos(ly, 4, 'F,B')+ pos(rx, 5, 'L,R'))
	ser.write(bytes(data, encoding='ascii'))
