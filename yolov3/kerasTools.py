import tensorflow as tf
import numpy as np


def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strides,
                                  padding=padding, use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)
    if bn:
        conv = tf.keras.layers.BatchNormalization()(conv)

    if activate:
        conv = tf.keras.layers.LeakyReLU(alpha=0.1)(conv)

    return conv


def residual_block(input_layer, input_channel, filter_num1, filter_num2):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1))
    conv = convolutional(conv, filters_shape=(3, 3, filter_num1, filter_num2))

    residual_output = short_cut + conv
    return residual_output


def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')


def darknet53(input_data):
    input_data = convolutional(input_data, (3, 3, 3, 32))
    input_data = convolutional(input_data, (3, 3, 32, 64), downsample=True)
    for i in range(1):
        input_data = residual_block(input_data, 64, 32, 64)
    input_data = convolutional(input_data, (3, 3, 64, 128), downsample=True)
    for i in range(2):
        input_data = residual_block(input_data, 128, 64, 128)
    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True)
    for i in range(8):
        input_data = residual_block(input_data, 256, 128, 256)
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True)
    for i in range(8):
        input_data = residual_block(input_data, 512, 256, 512)
    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True)
    for i in range(4):
        input_data = residual_block(input_data, 1024, 512, 1024)
    return route_1, route_2, input_data


def YOlov3(input_layer, NUM_CLASS):
    # After the input layer enters the Darknet-53 network, we get three branches
    route_1, route_2, conv = darknet53(input_layer)
    # See the orange module (DBL) in the figure above, a total of 5 Subconvolution operation
    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))
    conv_lobj_branch = convolutional(conv, (3, 3, 512, 1024))

    # conv_lbox is used to predict large-sized objects , Shape = [None, 13, 13, 255]
    conv_lbox = convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = convolutional(conv, (1, 1, 512, 256))
    # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
    # upsampling process does not need to learn, thereby reducing the network parameter
    conv = upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)
    conv = convolutional(conv, (1, 1, 768, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv_mobj_branch = convolutional(conv, (3, 3, 256, 512))

    # conv_mbox is used to predict medium-sized objects, shape = [None, 26, 26, 255]
    conv_mbox = convolutional(conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)
    conv = convolutional(conv, (1, 1, 384, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv_sobj_branch = convolutional(conv, (3, 3, 128, 256))

    # conv_sbox is used to predict small size objects, shape = [None, 52, 52, 255]
    conv_sbox = convolutional(conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbox, conv_mbox, conv_lbox]


def create_yolov3(mum_classes, input_size=416, channels=3):

    input_layer = tf.keras.layers.Input([input_size, input_size, channels])
    model_output = YOlov3(input_layer, mum_classes)
    yolov3 = tf.keras.Model(input_layer, model_output)

    return yolov3

