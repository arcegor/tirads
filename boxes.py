import os
import argparse
import cv2 as cv
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Great Description To Be Here')

parser.add_argument("-i",
                    "--image_path",
                    type=str)
parser.add_argument("-l",
                    "--labels_path",
                    type=str)
parser.add_argument("-c",
                    "--csv_path",
                    type=str)
parser.add_argument("-s",
                    "--shift",
                    type=int,
                    default=1)


def create_boxes(path_images, path_labels):
    # images = os.listdir(path_images)
    labels = os.listdir(path_labels)
    res = []
    k = 1

    for label in labels:
        image = label
        if '_label' in image:
            image = label.replace('_label', '')
        if '.label' in image:
            image = label.replace('.label', '')

        img = cv.imread(path_labels + "/" + label, cv.IMREAD_GRAYSCALE)

        rows, cols = img.shape

        x_max, x_min, y_max, y_min, flag = None, None, None, None, True
        for i in reversed(range(rows)):
            if not flag:
                break
            for j in range(cols):
                if img[i, j] > 200 and flag:
                    x_max = i + 1 + k
                    flag = False
        flag = True
        for i in range(rows):
            if not flag:
                break
            for j in range(cols):
                if img[i, j] > 200 and flag:
                    x_min = i - k
                    flag = False
        flag = True
        for i in reversed(range(cols)):
            if not flag:
                break
            for j in range(rows):
                if img[j, i] > 200 and flag:
                    y_max = i + 1 + k
                    flag = False
        flag = True
        for i in range(cols):
            if not flag:
                break
            for j in range(rows):
                if img[j, i] > 200 and flag:
                    y_min = i - k
                    flag = False
        if x_max:
            res.append([path_images + "/" + image, path_labels + "/" + label, x_min, x_max, y_min, y_max, cols, rows])
    return res


def boxes_to_csv():
    arr = create_boxes(args.image_part, args.label_path)
    column_name = ['image_path', 'image_label_path', 'up', 'down', 'left', 'right', 'width', 'height']
    df = pd.DataFrame(arr, columns=column_name)
    df.to_csv(args.csv_path, mode='a', header=not os.path.exists(args.csv_path), index=False)


def main():
    boxes_to_csv()


if __name__ == '__main__':
    args = parser.parse_args()
    main()
