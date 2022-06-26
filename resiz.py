import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps
import shutil
import csv
import re
import numpy as np
import pandas as pd

import cv2
from matplotlib import pyplot as plt


def resiz(pathin, pathout):
    files = os.listdir(pathin)
    files2 = os.listdir(pathout)
    for i in files:
        if i in files2: continue
        img = Image.open(pathin + '/' + i)
        resizedImage = img.resize((640, 640))
        resizedImage.save(pathout + '/' + i)


def pars(pathin, pathout):
    fileses = []
    files = os.listdir(pathin)
    for i in files:
        if '.xml' in i:
            fileses.append(i)
            tree = ET.parse(pathin + '/' + i)
            root_node = tree.getroot()
            for tag in root_node.findall('folder'):
                tag.text = 'datane'
            for tag in root_node.findall('path'):
                text = tag.text
                s = text[:37] + 'datane' + text[43:]
                tag.text = s
            tree.write(pathout + '/' + i)


def pars2(pathin, pathout):
    fileses = []
    files = os.listdir(pathin)
    for i in files:
        if '.xml' in i:
            fileses.append(i)
            tree = ET.parse(pathin + '/' + i)
            root_node = tree.getroot()
            for tag in root_node.findall('filename'):
                name = tag.text
            for tag in root_node.findall('object/name'):
                text = tag.text
            if text == 't':
                shutil.copy("datane/" + name, pathout)
                shutil.copy("annotations1/" + i, pathout)


def goto(pathin, pathout):
    dir = 'datane'
    files = os.listdir(pathin)
    files1 = os.listdir(dir)
    for i in files1:
        j = i.replace('.xml', '.jpg')
        if j in files:
            shutil.copy(pathin + '/' + j, pathout)


def clea(im, anno):
    imgs = os.listdir(im)
    annos = os.listdir(anno)
    for i in imgs:
        j = i.replace('.jpg', '.xml')
        if j in annos:
            shutil.copy(anno + '/' + j, 'imgs')


def prepare(pathin):
    # creating an og_image object
    files = os.listdir(pathin)
    for i in files:
        if 'disp.jpeg' in i:
            image = Image.open(pathin + "/" + i)
            # applying grayscale method
            gray_image = ImageOps.grayscale(image)
            gray_image.save(pathin + "/" + "gray_" + i)


def dirs(pathin, csvf):
    files = os.listdir(pathin)
    arr = []
    arr1 = []
    csvf = open(csvf, "r")
    reader = csv.reader(csvf)
    arr2 = []
    for row in reader:
        arr2.append(row)
    k = 0
    for file in files:
        for row in arr2:
            if file == row[0]:
                arr.append([row[0], row[1], row[2]])
                arr1.append([row[0], row[1], row[2]])
                k += 1
    with open("parisonmap.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(arr1)


def check(pathin):
    f1 = open("bb/bb_dis.csv", "r")
    r1 = csv.reader(f1)
    arr1 = []
    for row in r1:
        arr1.append(row)
    f2 = open("bb/bb_num.csv", "r")
    r2 = csv.reader(f2)
    arr2 = []
    for row in r2:
        arr2.append(row)
    files = os.listdir(pathin)
    k = 0
    sum = 0
    min = 100000000
    max = 0
    for file in files:
        for row2 in arr2:
            if row2[0] == file:
                num = int(row2[3])
                for row1 in arr1:
                    if int(row1[0]) == num:
                        k += 1
                        if int(row1[1]) > max:
                            max = int(row1[1])
                        if int(row1[1]) < min:
                            min = int(row1[1])
                        sum += int(row1[1])
    avg = sum // k
    arr = [k, min, max, avg]
    return arr


def prepro(file):
    file = open(file, "r")
    reader = csv.reader(file)
    l = []
    for row in reader:
        l.append(int(row[3]))
    return l


def coord(path):
    tree = ET.parse(path)
    root_node = tree.getroot()
    for tag in root_node.findall('object/bndbox/xmin'):
        xmin = int(tag.text)
    for tag in root_node.findall('object/bndbox/xmax'):
        xmax = int(tag.text)
    for tag in root_node.findall('object/bndbox/ymin'):
        ymin = int(tag.text)
    for tag in root_node.findall('object/bndbox/ymax'):
        ymax = int(tag.text)
    if xmin is not None:
        arr = [xmin, xmax, ymin, ymax]
    else:
        return False
    return arr


def pars3(pathin, pathout, src):
    fileses = []
    files = os.listdir(pathin)
    files_src = os.listdir(src)
    for i in files:
        if '.xml' in i:
            j = re.sub(i, '.xml', '')
            flag = False
            for k in files_src:
                temp = i.replace('.labels', '')
                temp = temp.replace('.xml', '.jpg')
                if k == temp:
                    flag = True
                    shutil.copy(src + '/' + k, pathout)
                    break
            if flag == False: continue
            fileses.append(i)
            tree = ET.parse(pathin + '/' + i)
            root_node = tree.getroot()
            for tag in root_node.findall('filename'):
                text = tag.text
                s = text.replace('.labels', '')
                tag.text = s
            for tag in root_node.findall('path'):
                text = tag.text
                s = text.replace('.labels', '')
                tag.text = s
            for tag in root_node.findall('folder'):
                tag.text = 'test'
            tree.write(pathout + '/' + i.replace('.labels', ''))


# pars3('images/test1', 'images/test', 'src')


def make_test_labels(src, lbl, out):
    src = os.listdir(src)
    for img in src:
        if '.xml' in img:
            continue
        name = img[:-4] + "_label" + ".jpg"
        shutil.copy(lbl + "/" + name, out)


make_test_labels('annotations/test', 'images1/source/labels', 'annotations/labels')

def resiz1(pathin, pathout):
    df = pd.read_csv(pathin)
    arr = df.values.tolist()
    for i in arr:
        image = Image.open(i[0])
        label = Image.open(i[1])
        resizedImage = image.resize((768, 768))
        resizedLabel = label.resize((768, 768))
        resizedImage.save(pathout + '/' + i)
