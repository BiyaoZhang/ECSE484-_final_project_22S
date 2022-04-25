import numpy as np
import xml.etree.ElementTree as ET
import glob
from sklearn.cluster import KMeans


def get_width_and_heigth_from_xml(image_xml_path, resize_width, resize_height):
    result = []

    # get root
    tree = open(image_xml_path, "rt", encoding="utf-8").read()
    root = ET.fromstring(tree)

    # get width and height of image
    size = root.find("size")
    original_width = int(size.find("width").text)
    original_height = int(size.find("height").text)

    # traversal of all object labels
    for obj in root.findall("object"):
        # get bndbox node
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # resize coordinate
        xmin = xmin / original_width * resize_width
        ymin = ymin / original_height * resize_height
        xmax = xmax / original_width * resize_width
        ymax = ymax / original_height * resize_height

        # width and height
        width = int(xmax - xmin)
        length = int(ymax - ymin)

        result.append([width, length])

    return result


def get_width_and_heigth_from_folder(xml_folder_path, resize_width, resize_height):
    result = []
    xml_paths = glob.glob(xml_folder_path + '/*.xml')
    for image_xml_path in xml_paths:
        wh_set = get_width_and_heigth_from_xml(image_xml_path, resize_width, resize_height)
        for wh in wh_set:
            result.append(wh)
    return result


def get_anchor_from_folder(data_folder_path, resize_width, resize_height, number_of_clusters=9):
    xml_folder_path = data_folder_path + '/train'

    wh_set = get_width_and_heigth_from_folder(xml_folder_path, resize_width, resize_height)
    clustering = KMeans(n_clusters=number_of_clusters)
    clustering.fit(wh_set)
    centers = clustering.cluster_centers_

    area = centers[:, 0] * centers[:, 1]

    area = area[..., np.newaxis]

    centers = np.concatenate((centers, area), axis=-1)

    centers = centers[centers[:, 2].argsort()]

    centers = centers.astype(float)

    anchors = np.reshape(centers[:, 0:2], (3, 3, 2))

    write_anchors_to_txt(anchors, data_folder_path + '/anchors.txt')

    return anchors


def write_anchors_to_txt(anchors, text_path):
    anchors = anchors.flatten()
    # save anchors to txt

    with open(text_path, "w+") as file:
        for value in anchors:
            file.write(str(value) + '\n')


def load_anchors_from_txt(text_path):
    # read anchors from txt
    anchors = []

    # text_path = data_folder_path + '/anchors.txt'

    with open(text_path, 'r') as file:
        for i, value in enumerate(file):
            anchors.append(value.strip('\n'))

    return np.array(anchors).astype(float).reshape((3, 3, 2))


