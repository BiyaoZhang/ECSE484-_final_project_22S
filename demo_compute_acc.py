import glob
import xml.etree.ElementTree as ET
import numpy as np
import time
from yolov3.utils import *
import numpy as np
from yolov3.dataloader import Dataloader
from yolov3.yolov3 import Yolov3

import matplotlib.pyplot as plt

def get_string_from_xml(image_xml_path):
    result = ''
    # get root
    tree = open(image_xml_path, "rt", encoding="utf-8").read()
    root = ET.fromstring(tree)
    # traversal of all object labels
    for obj in root.findall("object"):
        letter = obj.find("name").text
        result += letter
    return result

def get_recognition_acc_from_data_folder(data_folder_path, yolo_model, conf_threshold=0.6, iou_threshold=0.7):
    # get all image paths and xml paths
    xml_paths = glob.glob(data_folder_path + '/*.xml')
    img_paths= glob.glob(data_folder_path + '/*.jpg')

    np.sort(img_paths)
    np.sort(xml_paths)

    num_samples = len(img_paths)

    wrong_list = []

    correct_counter = 0
    recognition_counter = 0
    for i in range(num_samples):
        start_time = time.time()

        result = yolo_model.prediction_and_get_str(img_paths[i], conf_threshold, iou_threshold)
        answer = get_string_from_xml(xml_paths[i])
        if result == answer:
            correct_counter += 1
        else:
            wrong_list.append(img_paths[i])

        recognition_counter += 1

        end_time = time.time()

        time_cost = end_time - start_time

        print('Progress:{:d}/{:d} \t current acc:{:.3f} \t recognition time:{:.2f}s'
              .format(recognition_counter, num_samples, float(correct_counter/num_samples), time_cost))

    acc = correct_counter / num_samples
    print('Wrong list:')

    for path in wrong_list:
        print(path)

    return acc


if __name__ == '__main__':
    # folder path of weights
    weights_folder_path = 'temp'

    # creat Yolov3 instance
    model = Yolov3()

    # load weights
    model.load_weights(weights_folder_path)

    # folder path of test data
    data_folder_path = 'test_image/test'

    # calculate acc
    acc = get_recognition_acc_from_data_folder(data_folder_path, model, conf_threshold=0.3, iou_threshold=0.6)

    print('total acc:{:7.2f}'.format(acc))



