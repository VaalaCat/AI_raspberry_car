import serial
import time
ser = serial.Serial('/dev/ttyACM0',115200,timeout=0.5)

time.sleep(3)
data = bytearray(b'R000F000R0000')
ser.write(data)


while 1:
    for i in range(100, 255):
        i=str(i)
        temp = 'R' + '000' + 'F' + i + 'R0000'
        data = bytes(temp, encoding='ascii')
        ser.write(data)
        time.sleep(0.1)
        print(ser.readline())