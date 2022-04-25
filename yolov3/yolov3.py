import numpy as np

import tensorflow as tf
import shutil
from yolov3.utils import *
from yolov3.anchorTools import *
from yolov3.kerasTools import *
from yolov3.dataloader import Dataloader
import time
import cv2


class Yolov3(object):

    def __init__(self):
        self.conf_threshold = 0.3
        self.iou_threshold = 0.80

        self.data_path = None
        self.class_dict = None
        self.anchors = None

        self.tf_model = None
        self.has_loaded_weights = False

        self.init_lr = 1e-4

        self.end_lr = 1e-6

        self.global_steps = None
        self.warmup_steps = None
        self.total_steps = None

        self.train_writer = None
        self.test_writer = None

        self.model_output_strides = np.array([8, 16, 32], dtype=np.int32)

        self.optimizer = None

        self.model_input_size = 416

    def train_per_step(self, input_data, target):
        with tf.GradientTape() as tape:
            out_puts = self.tf_model(input_data, training=True)

            giou_loss = conf_loss = prob_loss = 0
            for i, out_put in enumerate(out_puts):
                pred = self.decode_output(out_put, len(self.class_dict), i)
                loss_list = self.compute_loss(pred, out_put, *target[i], i)
                giou_loss += loss_list[0]
                conf_loss += loss_list[1]
                prob_loss += loss_list[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, self.tf_model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.tf_model.trainable_variables))

            self.global_steps.assign_add(1)
            if self.global_steps < self.warmup_steps:  # and not TRAIN_TRANSFER:
                lr = self.global_steps / self.warmup_steps * self.init_lr
            else:
                lr = self.end_lr + 0.5 * (self.init_lr - self.end_lr) * (
                        1 + tf.cos(
                    (self.global_steps - self.warmup_steps) / (self.total_steps - self.warmup_steps) * np.pi))

            self.optimizer.lr.assign(lr.numpy())

            with self.train_writer.as_default():
                tf.summary.scalar("lr", self.optimizer.lr, step=self.global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=self.global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=self.global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=self.global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=self.global_steps)
            self.train_writer.flush()

        return self.global_steps.numpy(), self.optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    def validation_per_step(self, input_data, target):
        out_puts = self.tf_model(input_data, training=False)

        giou_loss = conf_loss = prob_loss = 0
        for i, out_put in enumerate(out_puts):
            pred = self.decode_output(out_put, len(self.class_dict), i)
            loss_list = self.compute_loss(pred, out_put, *target[i], i)
            giou_loss += loss_list[0]
            conf_loss += loss_list[1]
            prob_loss += loss_list[2]

        total_loss = giou_loss + conf_loss + prob_loss

        return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    def train_preprocess(self, data_path):
        # creat annotaion_train.txt, annotaion_test.txt, classes.txt
        self.data_path = data_path
        creat_annotation_files(self.data_path)

        # get class dict for txt
        self.class_dict = load_class(self.data_path + '/classes.txt')

        # get anchor from data folder
        anchors = get_anchor_from_folder(self.data_path, self.model_input_size, self.model_input_size)
        self.anchors = (np.array(anchors).T / self.model_output_strides).T

        # get model
        if self.tf_model is None:
            self.tf_model = create_yolov3(len(self.class_dict))


    def train(self, data_path, epochs, train_batch_size, test_batch_size):
        self.train_preprocess(data_path)

        self.tf_model.trainable = True

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if len(gpus) > 0:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError:
                pass

        log_path = self.data_path + '/log'
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
        self.train_writer = tf.summary.create_file_writer(log_path)
        self.test_writer = tf.summary.create_file_writer(log_path)

        train_set = Dataloader(data_path=data_path, datatype='train', num_classes=len(self.class_dict),
                               batch_size=train_batch_size, anchors=self.anchors)

        test_set = Dataloader(data_path=data_path, datatype='test', num_classes=len(self.class_dict),
                              batch_size=test_batch_size, anchors=self.anchors)

        steps_per_epoch = len(train_set)
        self.global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        self.warmup_steps = 2 * steps_per_epoch
        self.total_steps = epochs * steps_per_epoch

        self.optimizer = tf.keras.optimizers.Adam()

        total_start = time.time()
        best_val_loss = 1000
        for epoch in range(epochs):

            epoch_start = time.time()

            for input_data, target in train_set:
                batch_start = time.time()

                results = self.train_per_step(input_data, target)

                batch_end = time.time()

                batch_runtime = batch_end - batch_start
                cur_step = results[0] % steps_per_epoch
                print(
                    "epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, giou_loss:{:7.2f}, conf_loss:{:7.2f}, prob_loss:{:7.2f}, total_loss:{:7.2f}, batch_runtime:{:7.2f}s"
                        .format(epoch, cur_step, steps_per_epoch, results[1], results[2], results[3], results[4],
                                results[5], batch_runtime))

            if len(test_set) == 0:
                self.tf_model.save_weights(os.path.join(self.data_path, 'weights'))
                continue

            count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
            for input_data, target in test_set:
                results = self.validation_per_step(input_data, target)
                count += 1
                giou_val += results[0]
                conf_val += results[1]
                prob_val += results[2]
                total_val += results[3]

            # writing validate summary data
            with self.test_writer.as_default():
                tf.summary.scalar("validate_loss/total_val", total_val / count, step=epoch)
                tf.summary.scalar("validate_loss/giou_val", giou_val / count, step=epoch)
                tf.summary.scalar("validate_loss/conf_val", conf_val / count, step=epoch)
                tf.summary.scalar("validate_loss/prob_val", prob_val / count, step=epoch)
            self.test_writer.flush()

            print("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".
                  format(giou_val / count, conf_val / count, prob_val / count, total_val / count))

            if best_val_loss > (total_val / count):
                self.tf_model.save_weights(os.path.join(self.data_path, 'weights'))

            epoch_end = time.time()

            dhms_time = get_d_h_m_s(epoch_end - epoch_start)
            print('Epoch running_time: {:2.0f}d{:2.0f}h{:2.0f}m{:2.0f}s = {:2.0f}s\n\n'
                  .format(dhms_time[0], dhms_time[1], dhms_time[2], dhms_time[3], int(epoch_end - epoch_start)))

        total_end = time.time()
        dhms_time = get_d_h_m_s(total_end - total_start)
        print('Total running_time: {:2.0f}d{:2.0f}h{:2.0f}m{:2.0f}s = {:2.0f}s\n\n'
              .format(dhms_time[0], dhms_time[1], dhms_time[2], dhms_time[3], int(total_end - total_start)))

    def compute_loss(self, pred, conv, label, bboxes, i=0):
        NUM_CLASS = len(self.class_dict)
        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = self.model_output_strides[i] * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        giou = tf.expand_dims(get_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        iou = get_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        # Find the value of IoU with the real box The largest prediction box
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        # If the largest iou is less than the threshold, it is considered that the prediction box contains no objects, then the background box
        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < 0.5, tf.float32)

        conf_focal = tf.pow(respond_bbox - pred_conf, 2)

        # Calculate the loss of confidence
        # we hope that if the grid contains objects, then the network output prediction box has a confidence of 1 and 0 when there is no object.
        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return giou_loss, conf_loss, prob_loss

    # def decode_output(self, conv_output, NUM_CLASS, i=0):
    #     # where i = 0, 1 or 2 to correspond to the three grid scales
    #     conv_shape = tf.shape(conv_output)
    #     batch_size = conv_shape[0]
    #     output_size = conv_shape[1]
    #
    #     conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))
    #
    #     conv_raw_dxdy = conv_output[:, :, :, :, 0:2]  # offset of center position
    #     conv_raw_dwdh = conv_output[:, :, :, :, 2:4]  # Prediction box length and width offset
    #     conv_raw_conf = conv_output[:, :, :, :, 4:5]  # confidence of the prediction box
    #     conv_raw_prob = conv_output[:, :, :, :, 5:]  # category probability of the prediction box
    #
    #     # next need Draw the grid. Where output_size is equal to 13, 26 or 52
    #     y = tf.range(output_size, dtype=tf.int32)
    #     y = tf.expand_dims(y, -1)
    #     y = tf.tile(y, [1, output_size])
    #     x = tf.range(output_size, dtype=tf.int32)
    #     x = tf.expand_dims(x, 0)
    #     x = tf.tile(x, [output_size, 1])
    #
    #     xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    #     xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    #     xy_grid = tf.cast(xy_grid, tf.float32)
    #
    #     # Calculate the center position of the prediction box:
    #     pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * self.model_output_strides[i]
    #     # Calculate the length and width of the prediction box:
    #     # pred_wh = (tf.exp(conv_raw_dwdh) * self.anchors[i]) * self.model_input_size
    #     pred_wh = tf.exp(conv_raw_dwdh) * self.anchors[i]
    #
    #     pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    #     pred_conf = tf.sigmoid(conv_raw_conf)  # object box calculates the predicted confidence
    #     pred_prob = tf.sigmoid(conv_raw_prob)  # calculating the predicted probability category box object
    #
    #     # calculating the predicted probability category box object
    #     return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def decode_output(self, conv_output, NUM_CLASS, i=0):
        # where i = 0, 1 or 2 to correspond to the three grid scales
        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]  # offset of center position
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]  # Prediction box length and width offset
        conv_raw_conf = conv_output[:, :, :, :, 4:5]  # confidence of the prediction box
        conv_raw_prob = conv_output[:, :, :, :, 5:]  # category probability of the prediction box

        # next need Draw the grid. Where output_size is equal to 13, 26 or 52
        y = tf.range(output_size, dtype=tf.int32)
        y = tf.expand_dims(y, -1)
        y = tf.tile(y, [1, output_size])
        x = tf.range(output_size, dtype=tf.int32)
        x = tf.expand_dims(x, 0)
        x = tf.tile(x, [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        # Calculate the center position of the prediction box:
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * self.model_output_strides[i]
        # Calculate the length and width of the prediction box:
        pred_wh = (tf.exp(conv_raw_dwdh) * self.anchors[i]) * self.model_output_strides[i]

        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        pred_conf = tf.sigmoid(conv_raw_conf)  # object box calculates the predicted confidence
        pred_prob = tf.sigmoid(conv_raw_prob)  # calculating the predicted probability category box object

        # calculating the predicted probability category box object
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def load_weights(self, weights_folder):
        weights_path = os.path.join(weights_folder, 'weights')

        # load anchors
        anchors = load_anchors_from_txt(os.path.join(weights_folder, 'anchors.txt'))
        self.anchors = (np.array(anchors).T / self.model_output_strides).T

        # load classes
        self.class_dict = load_class(os.path.join(weights_folder, 'classes.txt'))

        if self.tf_model is None:
            self.tf_model = create_yolov3(len(self.class_dict))

        self.tf_model.load_weights(weights_path)

    def prediction_and_draw_boxes(self, image_path, conf_threshold=0.3, iou_threshold=0.45):
        if self.tf_model is None:
            raise RuntimeError('Please train a model or load weights')

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        boxes = self.detection(image_path)

        image = cv2.imread(image_path)

        show_image_and_boxes(image, boxes)

        str_result = ''
        for i, box in enumerate(boxes):
            str_result = str_result + self.class_dict[int(box[5])]

        return str_result

    def prediction_and_get_str(self, image_path, conf_threshold=0.3, iou_threshold=0.45):
        if self.tf_model is None:
            raise RuntimeError('Please train a model or load weights')

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        boxes = self.detection(image_path)

        str_result = ''
        for i, box in enumerate(boxes):
            str_result = str_result + self.class_dict[int(box[5])]

        # print(str_result)
        return str_result

    def detection(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # input_data = cv2.resize(np.copy(image), (self.model_input_size, self.model_input_size))
        # input_data = input_data / 225.0

        input_data = image_preprocess(np.copy(image), [self.model_input_size, self.model_input_size])
        input_data = tf.expand_dims(input_data, 0)

        pred_boxes = []
        # out_puts = self.tf_model(input_data[np.newaxis, :], training=False)
        # pred_boxes = self.tf_model.predict(input_data[np.newaxis, :])
        self.tf_model.trainable = False
        out_puts = self.tf_model.predict(input_data)

        for i, out_put in enumerate(out_puts):
            pred_box = self.decode_output(out_put, len(self.class_dict), i)
            pred_boxes.append(pred_box)

        pred_boxes = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_boxes]
        pred_boxes = tf.concat(pred_boxes, axis=0)

        # print(np.shape(pred_boxes))

        boxes = self.post_prediction(pred_boxes, image, self.conf_threshold)
        # print(boxes)
        boxes = nms_for_classes(boxes, self.iou_threshold)
        # print(boxes)
        boxes = nms_for_all(boxes, self.iou_threshold)

        # print(boxes)

        # sort boxes for left to right
        boxes = np.array(boxes)
        if boxes.ndim > 1:
            boxes = boxes[boxes[:, 0].argsort()]

        return boxes

    def post_prediction(self, pred_boxes, original_image, score_threshold=0.3):
        valid_scale = [0, np.inf]
        pred_bbox = np.array(pred_boxes)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                    pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
        # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        org_h, org_w = original_image.shape[:2]
        resize_ratio = min(self.model_input_size / org_w, self.model_input_size / org_h)

        dw = (self.model_input_size - resize_ratio * org_w) / 2
        dh = (self.model_input_size - resize_ratio * org_h) / 2

        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # 3. clip some boxes those are out of range
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # 4. discard some invalid boxes
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # 5. discard boxes with low scores
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > score_threshold
        mask = np.logical_and(scale_mask, score_mask)
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

        result = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

        return result
