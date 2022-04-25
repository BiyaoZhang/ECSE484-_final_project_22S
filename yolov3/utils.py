import cv2
import tensorflow as tf
import xml.etree.ElementTree as ET
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def get_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


def get_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    # Calculate the iou value between the two bounding boxes
    iou = inter_area / union_area

    # Calculate the coordinates of the upper left corner and the lower right corner of the smallest closed convex surface
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

    # Calculate the area of the smallest closed convex surface C
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # Calculate the GIoU value according to the GioU formula
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def load_class(class_txt_path):
    # loads class from a text file to dict
    classes = {}
    with open(class_txt_path, 'r') as file:
        for ID, name in enumerate(file):
            classes[ID] = name.strip('\n')
    return classes


def ParseXML(img_folder, file, Dataset_clases):
    for xml_file in glob.glob(img_folder + '/*.xml'):
        tree = ET.parse(open(xml_file))
        root = tree.getroot()
        image_name = root.find('filename').text
        img_path = img_folder + '/' + image_name
        for i, obj in enumerate(root.iter('object')):
            cls = obj.find('name').text
            if cls not in Dataset_clases:
                Dataset_clases.append(cls)
            cls_id = Dataset_clases.index(cls)
            xmlbox = obj.find('bndbox')
            OBJECT = (str(int(float(xmlbox.find('xmin').text))) + ','
                      + str(int(float(xmlbox.find('ymin').text))) + ','
                      + str(int(float(xmlbox.find('xmax').text))) + ','
                      + str(int(float(xmlbox.find('ymax').text))) + ','
                      + str(cls_id))
            img_path += ' ' + OBJECT

        file.write(img_path + '\n')


def creat_annotation_files(data_folder_path):
    #  data_path + '/' + 'annotations_' + datatype + '.txt'
    Dataset_train = data_folder_path + '/' + 'annotations_' + 'train' + '.txt'

    Dataset_test = data_folder_path + '/' + 'annotations_' + 'test' + '.txt'

    Dataset_clases_path = data_folder_path + '/' + 'classes' + '.txt'

    Dataset_clases = []

    for i, folder in enumerate(['train', 'test']):
        with open([Dataset_train, Dataset_test][i], "w+") as file:
            img_path = data_folder_path + '/' + folder

            ParseXML(img_path, file, Dataset_clases)

    with open(Dataset_clases_path, "w+") as file:
        for name in Dataset_clases:
            file.write(str(name) + '\n')


def get_d_h_m_s(seconds):
    day = 24 * 60 * 60
    hour = 60 * 60
    minute = 60

    days = seconds / day

    remain_seconds = seconds - (int(days) * day)
    hours = remain_seconds / hour

    remain_seconds = remain_seconds - (int(hours) * hour)
    minutes = remain_seconds / minute

    seconds = remain_seconds - (int(minutes) * minute)

    # return = [d, h, m, s]
    return [int(days), int(hours), int(minutes), int(seconds)]


def nms_for_classes(bboxes, iou_threshold):
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])

            iou = get_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])

            mask = np.where(iou < iou_threshold)

            cls_bboxes = cls_bboxes[mask]

    return best_bboxes


def nms_for_all(bboxes, iou_threshold):
    best_bboxes = []
    cls_bboxes = np.array(bboxes)
    while len(cls_bboxes) > 0:
        max_ind = np.argmax(cls_bboxes[:, 4])
        best_bbox = cls_bboxes[max_ind]
        best_bboxes.append(best_bbox)
        cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])

        iou = get_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])

        mask = np.where(iou < iou_threshold)

        cls_bboxes = cls_bboxes[mask]

    return best_bboxes


def show_image_and_boxes(image, boxes):
    plt.imshow(np.array(image))
    for box in boxes:
        rect = Rectangle((int(box[0]), int(box[1])), (int(box[2]) - int(box[0])), (int(box[3]) - int(box[1])),
                         fill=False, color='red')
        ax = plt.gca()
        ax.axes.add_patch(rect)
    plt.show()


def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes
