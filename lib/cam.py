import view
import RC as rc
from simple_pid import PID
import cv2

towards = 'F'
rcswtich = 0

pids = PID(0.3, 0.1, 0.12, setpoint=1)
pidw = PID(0, 0, 0, setpoint=1)

if __name__ == "__main__":
    while True:
        frame1 = view.pre_process('L')
        frame2 = view.pre_process('R')
        frame3 = view.pre_process('F')
        frame4 = view.pre_process('B')
        cv2.imshow("???", frame1)
        cv2.imshow("ppp", frame2)
        cv2.imshow("jjj", frame3)
        cv2.imshow("iii", frame4)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break