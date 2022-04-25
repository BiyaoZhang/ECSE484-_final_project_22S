import numpy as np
import os
import cv2
from yolov3.utils import *
import tensorflow as tf


class Dataloader(object):
    def __init__(self, data_path, datatype, num_classes, batch_size, anchors,
                 model_input_size=416, model_output_strides=None,
                 ):
        # annotation path =  dataname/dataname_train.txt or dataname/dataname_test.txt
        self.data_annotations_path = data_path + '/' + 'annotations_' + datatype + '.txt'

        self.batch_size = batch_size

        self.model_input_size = model_input_size

        if model_output_strides is None:
            model_output_strides = [8, 16, 32]
        self.model_output_strides = np.array(model_output_strides, dtype=np.int32)

        self.anchors = (np.array(anchors).T / self.model_output_strides).T

        self.num_anchors_per_output = np.shape(self.anchors)[1]

        self.num_outputs = np.shape(self.model_output_strides)[0]

        self.num_classes = num_classes

        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_annotations(self):
        result = []

        with open(self.data_annotations_path, 'r') as f:
            txt = f.readlines()
            # print(txt)
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
            # print(annotations)

        # filter
        for annotation in annotations:
            elements = annotation.split()
            # elements[0] should be path of image
            image_path = elements[0]
            if os.path.exists(image_path):
                boxes = []
                # elements[1:] should be boxes
                is_good_annotation = True
                for element in elements[1:]:
                    # legal box should only contain numbers and 5 numbers
                    if not element.replace(',', '').isnumeric() or len(element.split(',')) != 5:
                        is_good_annotation = False
                        break
                    else:
                        boxes.append(list(map(int, element.split(','))))
                if is_good_annotation:
                    result.append([image_path, np.array(boxes)])

        np.random.shuffle(result)

        return np.array(result)

    def preprocess_annotation(self, annotation):
        image_path = annotation[0]
        boxes = annotation[1]
        image = cv2.imread(image_path)

        image, bboxes = image_preprocess(np.copy(image), [self.model_input_size, self.model_input_size], np.copy(boxes))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (self.model_input_size, self.model_input_size))
        # # normalize image
        # image = image / 225.0
        #
        # h, w, _ = image.shape
        # scale_x = self.model_input_size / w
        # scale_y = self.model_input_size / h
        #
        # # transfer (x_min, y_min, x_max, y_max) to fit size (416, 416)
        # new_boxes = np.copy(boxes)
        # new_boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
        # new_boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y

        return image, bboxes

    def boxes_to_label_and_real_boxes(self, boxes):
        model_out_size = self.model_input_size // self.model_output_strides

        label = [np.zeros((model_out_size[i], model_out_size[i], self.num_anchors_per_output,
                           5 + self.num_classes)) for i in range(int(self.num_outputs))]

        obj_boxes_xywh = [np.zeros((100, 4)) for _ in range(int(self.num_outputs))]

        box_counts = np.zeros((3,))

        for box in boxes:
            box_xyxy = box[: 4]
            box_class_index = int(box[4])

            class_onehot = np.zeros(self.num_classes, dtype=np.float)
            class_onehot[box_class_index] = 1.0
            # smooth one hot vector
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = class_onehot * (1 - deta) + deta * uniform_distribution

            # xywh for (416, 461) size
            box_xywh = np.concatenate([(box_xyxy[:2] + box_xyxy[2:]) * 0.5, box_xyxy[2:] - box_xyxy[:2]],
                                      axis=-1)

            box_xywh_scaled = 1.0 * box_xywh[np.newaxis, :] / self.model_output_strides[:, np.newaxis]

            is_box_find_place_to_label = False

            ious_list = []

            for i in range(3):
                anchors_xywh = np.zeros((self.num_anchors_per_output, 4))
                anchors_xywh[:, 0:2] = np.floor(box_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                ious = get_iou(box_xywh_scaled[np.newaxis, :], anchors_xywh)
                ious_list.append(ious)

                # 0.3 iou threshold for determining whether this box should be labeled at this anchor
                ious_mask = ious > 0.3  # ious_mask = [True/False * 3]

                if np.any(ious_mask):
                    # save in these anchors

                    index_x, index_y = np.floor(1.0 * box_xywh_scaled[0:2]).astype(np.int32)

                    label[i][index_y, index_x, ious_mask, :] = 0
                    # label[i][index_y, index_x, ious_mask, 0:4] = box_xywh_scaled
                    label[i][index_y, index_x, ious_mask, 0:4] = box_xywh
                    label[i][index_y, index_x, ious_mask, 4:5] = 1.0
                    label[i][index_y, index_x, ious_mask, 5:] = smooth_onehot

                    index_for_obj_boxes = int(box_counts[i] % 100)
                    obj_boxes_xywh[i][index_for_obj_boxes, :4] = box_xywh
                    box_counts[i] += 1

                    is_box_find_place_to_label = True

            if not is_box_find_place_to_label:
                ious_array = np.array(ious_list).reshape(-1)
                index_of_max_iou = np.argmax(ious_array, axis=-1)

                index_of_outputs = int(index_of_max_iou / self.num_anchors_per_output)
                index_of_anchors = int(index_of_max_iou % self.num_anchors_per_output)

                index_x, index_y = np.floor(box_xywh_scaled[index_of_outputs, 0:2]).astype(np.int32)

                label[index_of_outputs][index_y, index_x, index_of_anchors, :] = 0
                # label[index_of_outputs][index_y, index_x, index_of_anchors, 0:4] = box_xywh_scaled
                label[index_of_outputs][index_y, index_x, index_of_anchors, 0:4] = box_xywh
                label[index_of_outputs][index_y, index_x, index_of_anchors, 4:5] = 1.0
                label[index_of_outputs][index_y, index_x, index_of_anchors, 5:] = smooth_onehot

                index_for_obj_boxes = int(box_counts[index_of_outputs] % 100)
                obj_boxes_xywh[index_of_outputs][index_for_obj_boxes, :4] = box_xywh
                box_counts[index_of_outputs] += 1

        # label = [s, m, l]
        # obj_boxes_xywh = [s, m, l]

        return label, obj_boxes_xywh

    def decode_annotation(self, annotation):

        input_image_tensor, boxes = self.preprocess_annotation(annotation)
        # label = [s, m, l]
        # obj_boxes_xywh = [s, m, l]
        label_tensor, obj_boxes_xywh = self.boxes_to_label_and_real_boxes(boxes)

        return input_image_tensor, label_tensor, obj_boxes_xywh

    def __next__(self):
        with tf.device('/cpu:0'):

            batch_input = np.zeros((self.batch_size, self.model_input_size, self.model_input_size, 3), dtype=np.float32)

            model_out_size = self.model_input_size // self.model_output_strides

            batch_label_sboxes = np.zeros((self.batch_size, model_out_size[0], model_out_size[0],
                                           self.num_anchors_per_output, 5 + self.num_classes), dtype=np.float32)
            batch_label_mboxes = np.zeros((self.batch_size, model_out_size[1], model_out_size[1],
                                           self.num_anchors_per_output, 5 + self.num_classes), dtype=np.float32)
            batch_label_lboxes = np.zeros((self.batch_size, model_out_size[2], model_out_size[2],
                                           self.num_anchors_per_output, 5 + self.num_classes), dtype=np.float32)

            batch_obj_sboxes = np.zeros((self.batch_size, 100, 4), dtype=np.float32)
            batch_obj_mboxes = np.zeros((self.batch_size, 100, 4), dtype=np.float32)
            batch_obj_lboxes = np.zeros((self.batch_size, 100, 4), dtype=np.float32)

            num_samples = 0
            if self.batch_count < self.num_batchs:
                while num_samples < self.batch_size:
                    anno_index = self.batch_count * self.batch_size + num_samples
                    if anno_index >= self.num_samples:
                        anno_index -= self.num_samples

                    input_image, label, obj_boxes = self.decode_annotation(self.annotations[anno_index])

                    batch_input[num_samples, :, :, :] = input_image

                    batch_label_sboxes[num_samples, :, :, :, :] = label[0]
                    batch_label_mboxes[num_samples, :, :, :, :] = label[1]
                    batch_label_lboxes[num_samples, :, :, :, :] = label[2]

                    batch_obj_sboxes[num_samples, :, :] = obj_boxes[0]
                    batch_obj_mboxes[num_samples, :, :] = obj_boxes[1]
                    batch_obj_lboxes[num_samples, :, :] = obj_boxes[2]

                    num_samples += 1
                self.batch_count += 1

                batch_starget = batch_label_sboxes, batch_obj_sboxes
                batch_mtarget = batch_label_mboxes, batch_obj_mboxes
                batch_ltarget = batch_label_lboxes, batch_obj_lboxes

                return batch_input, (batch_starget, batch_mtarget, batch_ltarget)
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def __len__(self):
        return self.num_batchs

    def __iter__(self):
        return self
