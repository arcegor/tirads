import os
import re

import cv2 as cv
import pandas as pd
import numpy as np


def drawBoxes(csv_path, dst):
    src = pd.read_csv(csv_path)
    df = src.values.tolist()
    for item in df:
        image = cv.imread(item[0])
        label = cv.imread(item[1])
        x1, x2, y1, y2 = item[2], item[3], item[4], item[5]

        cv.rectangle(image, (y1, x1), (y2, x2), (255, 0, 0), 2)

        cv.rectangle(label, (y1, x1), (y2, x2), (255, 0, 0), 2)

        out = cv.hconcat([image, label])
        name = re.split('/', item[0])[-1]
        cv.imwrite(dst + '/a_' + name, out)

        # cv.imwrite(dst + '/l_' + item[1], out_label)


drawBoxes('boxes.csv', 'images1/inputAnalitics')
