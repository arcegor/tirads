import os
import re

import cv2 as cv

import tifffile as tiff


def parseDirs(src, dst):
    dirs = os.listdir(src)
    for dir in dirs:
        imgs = os.listdir(src + '/' + dir)
        for j in imgs:
            res = re.findall('labels', j)
            l = len(res)

            temp = convert(src + '/' + dir + '/' + j)
            if l == 0:
                for index, item in enumerate(temp):
                    path = dst + '/images/' + dir + '_' + str(index) + '.jpg'
                    cv.imwrite(path, item)
            else:
                for index, item in enumerate(temp):
                    item = cv.convertScaleAbs(item, alpha=255.0)
                    path = dst + '/labels/' + dir + '_' + str(index) + '_' + 'label' + '.jpg'
                    cv.imwrite(path, item)




def convert(img):
    tmp = []
    temp = tiff.imread(img)
    for img in temp:
        shape = img.shape
        if shape == (735, 975):
            img = img[100:700, 0:900]
        if shape == (735, 975, 3):
            img = img[100:700, 0:900]
        img = cv.resize(img, (738, 738))
        tmp.append(img)
    return tmp


if __name__ == '__main__':
    parseDirs("C:/Users/yvego/Downloads/SEGMENTED1", "resized")

