import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

import view
import RC

data = []
cnt=0

if __name__ == '__main__':
    
    while True:
        RC.run(RC.get_xbox_data(),False)
        name, but = RC.get_buttom()
        gray = view.pre_process('F')
        dst = cv2.resize(gray, (28, 28))
        retval, dst = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU)
        cv2.imshow("now",dst)
        if but.count(1) != 0:
            data.append([but.index(1), dst])
            cnt += 1
            print(cnt)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if but[8] == 1:
            np.save('test',data)
            break