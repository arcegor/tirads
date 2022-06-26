import cv2 as cv
import os

import numpy as np

k = 0

for filename in os.listdir("images1/source/images"):
    newContours = []
    newHierarchy = []
    img = cv.imread("images1/source/images/" + filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow("src", img)

    # src = cv.GaussianBlur(img, (3, 3), 0)
    dst = cv.fastNlMeansDenoising(gray, None, 20, 7, 21)
    # blur = cv.bilateralFilter(img, 9, 75, 75)
    # gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # edges = cv.Canny(image=dst, threshold1=100, threshold2=200)  # Canny Edge Detection

    # cv.imshow('Canny Edge Detection', edges)

    # grad_x = cv.Sobel(dst, cv.CV_64F, 1, 0, ksize=5)
    # #
    # grad_y = cv.Sobel(dst, cv.CV_64F, 0, 1, ksize=5)
    #
    # abs_grad_x = cv.convertScaleAbs(grad_x)
    # abs_grad_y = cv.convertScaleAbs(grad_y)
    #
    # grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    ret, th1 = cv.threshold(dst, 40, 255, cv.THRESH_BINARY)
    # th2 = cv.adaptiveThreshold(dst, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    # th3 = cv.adaptiveThreshold(dst, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    # cv.imshow("denoise", dst)
    contours, hierarchy = cv.findContours(image=th1, mode=cv.RETR_CCOMP, method=cv.CHAIN_APPROX_NONE)

    # res = np.concatenate((img, th1), axis=1)
    # th1 = cv.cvtColor(th1, cv.COLOR_GRAY2RGB)
    # cv.imwrite(str(k) + "_" + filename, res)
    # for i in range(len(contours)):
    #     print("Контур ", contours[i])
    #     print("Иерархия ", hierarchy[0][i])
    #     if hierarchy[0][i][3] == -1: continue
    #     newContours.append(contours[i])
    #     newHierarchy.append(hierarchy[0][i])
    # newHierarchy = np.expand_dims(newHierarchy, axis=0)
    # newContours = np.array(newContours)
    # print(np.shape(newContours))
    # print(np.shape(newHierarchy))
    backtorgb = cv.cvtColor(th1, cv.COLOR_GRAY2RGB)
    cv.drawContours(backtorgb, contours, 50, (255, 0, 0), 2, cv.LINE_AA, hierarchy, 2)
    # print(np.shape(contours))
    # print(np.shape(hierarchy))
    cv.imshow("threshold", backtorgb)
    # cv.imshow("sobel", grad)
    # cv.imshow("atgau", th3)
    cv.waitKey(0)
