import simple_pid
import pigpio
import time

pi = pigpio.pi()

short = 0.0005
long = 0.00168


pi.hardware_PWM(18, 0, 0)
time.sleep (0.1)
pi.hardware_PWM(18, 38000, 300000)
time.sleep (0.009)
pi.hardware_PWM(18, 0, 0)
time.sleep (0.0045)

pi.hardware_PWM(18, 38000, 300000)
time.sleep (short)
pi.hardware_PWM(18, 0, 0)
time.sleep (short) 

pi.hardware_PWM(18, 38000, 300000)
time.sleep (short)
pi.hardware_PWM(18, 0, 0)
time.sleep (short) 

pi.hardware_PWM(18, 38000, 300000)
time.sleep (short)
pi.hardware_PWM(18, 0, 0)
time.sleep (long) 
