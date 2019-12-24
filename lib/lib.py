'''import numpy as np
import cv2

center = 320
cap = cv2.VideoCapture(1)
cap.set(3,640) #设置分辨率
cap.set(4, 480)


while (True):
    ret, frame = cap.read()'''
# 尝试使用矩阵求导巡线
# 失败
'''
    frame_rgb = frame# Drawing color points requires RGB image
    # ret, thresh = cv2.threshold(frame, 105, 255, cv2.THRESH_BINARY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    signed_thresh = thresh[start_height]#.astype(np.int16) # select only one row
    diff = np.diff(signed_thresh)  #The derivative of the start_height line

    points = [] #maximums and minimums of derivative
    points.append(np.where(diff == np.max(diff)))
    points.append(np.where(diff == np.min(diff)))

    cv2.line(frame_rgb,(0,start_height),(640,start_height),(0,255,0),1) # draw horizontal line where scanning 

    if len(points) > 0: # if finds something like a black line
        middle = (points[0]) + int(points[1])) / 2
        cv2.circle(frame_rgb, (points[0], start_height), 2, (255,0,0), -1)
        cv2.circle(frame_rgb, (points[1], start_height), 2, (255,0,0), -1)
        cv2.circle(frame_rgb, (int(middle), start_height), 2, (0,0,255), -1)
    else:
        start_height -= 5
        start_height = start_height % 480
        no_points_count += 1
    frames.append(frame_rgb)
    frames.append(thresh)	
    if no_points_count > 50:
        print("Line lost")
        break
    cv2.imshow("???",frame)'''
'''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    cv2.imshow("gray", gray)
    # 大津法二值化
    retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
#    cv2.imshow("dst", dst)
    # 膨胀，白区域变大
    dst = cv2.dilate(dst, None, iterations=2)
#    cv2.imshow("dst2", dst)
    # # 腐蚀，白区域变小 #
    dst = cv2.erode(dst, None, iterations=6)
    #cv2.imshow("dst3", dst)
    #canny边缘查找
    color = dst[400]
    canny = cv2.Canny(gray, 30, 150)
    canny = np.uint8(np.absolute(canny))
    #display two images in a figure
    cv2.imshow("Edge detection by Canny", np.hstack([gray,canny]))
    final = np.hstack([gray,canny])'''
'''try:
        # 找到黑色像素点个数
        line_count = np.sum(color == 0)
        # 找到黑色像素点索引
        line_index = np.where(color == 0)
        # 防止white_count=0的报错
        if line_count == 0:
            line_count = 1
        # 找到黑色像素的中心点位置
        # 边缘检测，计算黑色边缘的位置和/2，即是线的中央位置。
        center = (line_index[0][line_count - 1] + line_index[0][0]) / 2
        # 计算出center与标准中心点的偏移量
        direction = center - 320
        print(direction)
        
    except:
        continue'''
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
  # 上面全是抄的，都是废物，我要自己写了
'''gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cur = 400
    x = []
    y = []
    while cur > 60:
        row = gray[cur].astype(np.int16)
        diff = np.diff(row)
        edge = np.where(final == 255)
        #edge = np.where(np.logical_or(diff > 100, diff <-100))
        if len(edge) > 0 and len(edge[0]) > 1:
            cv2.circle(frame, (edge[0][0], cur), 2, (255, 0, 0), -1)
            cv2.circle(frame, (edge[0][1], cur), 2, (255, 0, 0), -1)
            middle = int((edge[0][0] + edge[0][1]) / 2)
            cv2.circle(frame, (middle, cur), 2, (0, 0, 255), -1)
            x.append(middle)
            y.append(cur)
        cur -= 20
    try:
        X = np.array(x)
        Y = np.array(y)
        direction = np.polyfit(X, Y, 1)
        print(direction)
        cv2.line(frame, (int(direction[1]), 0), (int(-direction[0] * 400 + direction[1]), 400), (0, 255, 0))
    except:
        pass
    cv2.imshow("??",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''
'''
# -*- coding: utf-8 -*-

# 通过OpenCV实现车道线检测

# Key Point:
# 1.打开视频文件
# 2.循环遍历每一帧
# 3.canny边缘检测，检测line
# 4.去除多余图像直线
# 5.霍夫变换
# 6.叠加变换与原始图像
# 7.车道检测


# Tools
# Canny检测


def do_canny(frame):
    # 将每一帧转化为灰度图像，去除多余信息
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 高斯滤波器，去除噪声，平滑图像
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    # 边缘检测
    # minVal = 50
    # maxVal = 150
    canny = cv.Canny(blur, 50, 150)

    return canny

# 图像分割，去除多余线条信息


def do_segment(frame):

        # 获取图像高度(注意CV的坐标系,正方形左上为0点，→和↓分别为x,y正方向)
    height = frame.shape[0]

    # 创建一个三角形的区域,指定三点
    polygons = np.array([
        [(0, height),
         (800, height),
         (380, 290)]
    ])

    # 创建一个mask,形状与frame相同，全为0值
    mask = np.zeros_like(frame)

    # 对该mask进行填充，做一个掩码
    # 三角形区域为1
    # 其余为0
    cv.fillPoly(mask, polygons, 255)

    # 将frame与mask做与，抠取需要区域
    segment = cv.bitwise_and(frame, mask)

    return segment

# 车道左右边界标定


def calculate_lines(frame, lines):
    # 建立两个空列表，用于存储左右车道边界坐标
    left = []
    right = []

    # 循环遍历lines
    for line in lines:
        # 将线段信息从二维转化能到一维
        x1, y1, x2, y2 = line.reshape(4)

        # 将一个线性多项式拟合到x和y坐标上，并返回一个描述斜率和y轴截距的系数向量
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]  # 斜率
        y_intercept = parameters[1]  # 截距

        # 通过斜率大小，可以判断是左边界还是右边界
        # 很明显左边界slope<0(注意cv坐标系不同的)
        # 右边界slope>0
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))

    # 将所有左边界和右边界做平均，得到一条直线的斜率和截距
    left_avg = np.average(left, axis=0)
    right_avg = np.average(right, axis=0)
    # 将这个截距和斜率值转换为x1,y1,x2,y2
    left_line = calculate_coordinate(frame, parameters=left_avg)
    right_line = calculate_coordinate(frame, parameters=right_avg)

    return np.array([left_line, right_line])

# 将截距与斜率转换为cv空间坐标


def calculate_coordinate(frame, parameters):
    # 获取斜率与截距
    slope, y_intercept = parameters

    # 设置初始y坐标为自顶向下(框架底部)的高度
    # 将最终的y坐标设置为框架底部上方150
    y1 = frame.shape[0]
    y2 = int(y1-150)
    # 根据y1=kx1+b,y2=kx2+b求取x1,x2
    x1 = int((y1-y_intercept)/slope)
    x2 = int((y2-y_intercept)/slope)
    return np.array([x1, y1, x2, y2])

# 可视化车道线


def visualize_lines(frame, lines):
    lines_visualize = np.zeros_like(frame)
    # 检测lines是否为空
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # 画线
            cv.line(lines_visualize, (x1, y1), (x2, y2), (0, 0, 255), 5)
    return lines_visualize


if __name__ == "__main__":

    # 视频读取
    cap = cv.VideoCapture(0)
    ret, frame = cap.read()
    frame = frame.astype(np.int64)
    # 当视频还是打开的时候，循环遍历每一帧
    while (1):
        # 边缘检测
        canny = do_canny(frame)
        # cv.imshow("canny", canny)
        # 图像分割，去除多余直线,只保留需要的直线
        # 原理见博文
        segment = do_segment(canny)
        # cv.imshow("segment", segment)

        # 原始空间中，利用Canny梯度，找到很多练成线的点
        # 利用霍夫变换，将这些点变换到霍夫空间中，转换为直线
        hough = cv.HoughLinesP(segment, 2, np.pi/180, 100, minLineLength=100, maxLineGap=50)
        # cv.imshow("hough", hough)

        # 将从hough检测到的多条线平均成一条线表示车道的左边界，
        # 一条线表示车道的右边界
        lines = calculate_lines(frame, hough)

        # 可视化
        lines_visualize = visualize_lines(frame, lines)  # 显示
        # cv.imshow("lines",lines_visualize)

        # 叠加检测的车道线与原始图像,配置两张图片的权重值
        # alpha=0.6, beta=1, gamma=1
        output = cv.addWeighted(frame, 0.6, lines_visualize, 1, 0.1)
        #cv.imshow("output", output)

        # q键退出
        if cv.waitKey(10) & 0xff == ord('q'):
            break

        # 释放，关闭
    cap.release()
    cv.destroyAllWindows()
'''

#再写一个


#失败
'''
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

blur_ksize = 5  # Gaussian blur kernel size
canny_lthreshold = 50  # Canny edge detection low threshold
canny_hthreshold = 150  # Canny edge detection high threshold

# Hough transform parameters
rho = 1
theta = np.pi / 180
threshold = 15
min_line_length = 40
max_line_gap = 20


def roi_mask(img, vertices):
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        mask_color = (255,) * channel_count
    else:
        mask_color = 255

    cv2.fillPoly(mask, vertices, mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_roi(img, vertices):
    cv2.polylines(img, vertices, True, [255, 0, 0], thickness=2)


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    draw_lanes(line_img, lines)
    return line_img


def draw_lanes(img, lines, color=[255, 0, 0], thickness=8):
    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            if k < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

    if (len(left_lines) <= 0 or len(right_lines) <= 0):
        return img

    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)
    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]
    right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
    right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]

    left_vtx = calc_lane_vertices(left_points, 325, img.shape[0])
    right_vtx = calc_lane_vertices(right_points, 325, img.shape[0])

    cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)
    cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)


def clean_lines(lines, threshold):
    slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0:
        mean = np.mean(slope)
        diff = [abs(s - mean) for s in slope]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slope.pop(idx)
            lines.pop(idx)
        else:
            break


def calc_lane_vertices(point_list, ymin, ymax):
    x = [p[0] for p in point_list]
    y = [p[1] for p in point_list]
    fit = np.polyfit(y, x, 1)
    fit_fn = np.poly1d(fit)

    xmin = int(fit_fn(ymin))
    xmax = int(fit_fn(ymax))

    return [(xmin, ymin), (xmax, ymax)]


def process_an_image(img):
    roi_vtx = np.array([[(0, img.shape[0]), (460, 325), (520, 325), (img.shape[1], img.shape[0])]])

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)
    edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)
    roi_edges = roi_mask(edges, roi_vtx)
    line_img = hough_lines(roi_edges, rho, theta, threshold, min_line_length, max_line_gap)
    res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)

    # plt.figure()
    # plt.imshow(gray, cmap='gray')
    # plt.savefig('../resources/gray.png', bbox_inches='tight')
    # plt.figure()
    # plt.imshow(blur_gray, cmap='gray')
    # plt.savefig('../resources/blur_gray.png', bbox_inches='tight')
    # plt.figure()
    # plt.imshow(edges, cmap='gray')
    # plt.savefig('../resources/edges.png', bbox_inches='tight')
    # plt.figure()
    # plt.imshow(roi_edges, cmap='gray')
    # plt.savefig('../resources/roi_edges.png', bbox_inches='tight')
    # plt.figure()
    # plt.imshow(line_img, cmap='gray')
    # plt.savefig('../resources/line_img.png', bbox_inches='tight')
    # plt.figure()
    # plt.imshow(res_img)
    # plt.savefig('../resources/res_img.png', bbox_inches='tight')
    # plt.show()


    return res_img


# img = mplimg.imread("../resources/lane.jpg")
# process_an_image(img)

output = '../resources/video_1_sol.mp4'
clip = VideoFileClip("../resources/video_1.mp4")
out_clip = clip.fl_image(process_an_image)
out_clip.write_videofile(output, audio=False)
'''
