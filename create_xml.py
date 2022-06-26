import re
import xml.etree.ElementTree as ET
import argparse
import pandas as pd
import shutil

parser = argparse.ArgumentParser(description='Great Description To Be Here')

parser.add_argument("-x",
                    "--xml_path",
                    type=str)
parser.add_argument("-c",
                    "--csv_path",
                    type=str)


def boxes_to_xml(arr, flag):
    root = ET.Element('annotations')

    folder = ET.SubElement(root, 'folder')
    folder.text = flag

    filename = ET.SubElement(root, 'filename')
    filename.text = re.split('/', arr[0])[-1]

    path = ET.SubElement(root, 'path')
    path.text = arr[0]

    source = ET.SubElement(root, 'source')

    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    size = ET.SubElement(root, 'size')

    width = ET.SubElement(size, 'width')
    width.text = str(arr[6])

    height = ET.SubElement(size, 'height')
    height.text = str(arr[7])

    depth = ET.SubElement(size, 'depth')
    depth.text = '3'

    segmented = ET.SubElement(root, 'segmented')
    segmented.text = '0'

    object = ET.SubElement(root, 'object')

    name = ET.SubElement(object, 'name')
    name.text = 'rw'

    pose = ET.SubElement(object, 'pose')
    pose.text = 'Unspecified'

    truncated = ET.SubElement(object, 'truncated')
    truncated.text = '0'

    difficult = ET.SubElement(object, 'difficult')
    difficult.text = '0'

    bndbox = ET.SubElement(object, 'bndbox')

    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = str(arr[4])

    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = str(arr[2])

    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = str(arr[5])

    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = str(arr[3])

    ET.indent(root)

    tree = ET.ElementTree(root)

    return tree


def create_dataset():
    df = pd.read_csv(args.csv_path)
    arr = df.values.tolist()
    test_patterns = ['93', '39', '75', '17']
    for index, item in enumerate(arr):
        if re.split('_', re.split('/', item[0])[-1])[0] in test_patterns:
            name = re.split('/', item[0])[-1].replace('.jpg', '.xml')
            tree = boxes_to_xml(item, 'test')
            tree.write(args.xml_path + '/' + 'test' + '/' + name)
            shutil.copy(item[0], args.xml_path + '/' + 'test')

        else:
            name = re.split('/', item[0])[-1].replace('.jpg', '.xml')
            tree = boxes_to_xml(item, 'train')
            tree.write(args.xml_path + '/' + 'train' + '/' + name)
            shutil.copy(item[0], args.xml_path + '/' + 'train')


def main():
    create_dataset()


if __name__ == '__main__':
    args = parser.parse_args()
    main()
