"""Darknet19 Model Defined in Keras."""
import functools
from functools import partial

from keras.layers import Conv2D, MaxPooling2D, MaxPooling3D, Reshape, Input, Dense, Flatten
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.initializers import glorot_uniform
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.regularizers import l2

import tensorflow as tf

from utils import compose

IMAGESIZE = 150528
RESOLUTION = [224, 224, 3]
RESTORE = 'model_progres.h5'
N_CLASSES = 3

# Partial wrapper for Convolution2D with static default argument.
_DarknetConv2D = partial(Conv2D, padding='same')


@functools.wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet weight regularizer for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs.update(kwargs)
    return _DarknetConv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def bottleneck_block(outer_filters, bottleneck_filters):
    """Bottleneck block of 3x3, 1x1, 3x3 convolutions."""
    return compose(
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)),
        DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))


def bottleneck_x2_block(outer_filters, bottleneck_filters):
    """Bottleneck block of 3x3, 1x1, 3x3, 1x1, 3x3 convolutions."""
    return compose(
        bottleneck_block(outer_filters, bottleneck_filters),
        DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))


def darknet_body():
    """Generate first 18 conv layers of Darknet-19."""
    return compose(
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(),
        bottleneck_block(128, 64),
        MaxPooling2D(),
        bottleneck_block(256, 128),
        MaxPooling2D(),
        bottleneck_x2_block(512, 256),
        MaxPooling2D(),
        bottleneck_x2_block(1024, 512))

def darknet19(inputs):
    """Generate Darknet-19 model for Imagenet classification."""
    body = darknet_body()(inputs)
    logits = DarknetConv2D(1000, (1, 1), activation='softmax')(body)
    return Model(inputs, logits)

def yolo_lstm_stage_1(input, num_anchors, num_classes, stateful = False):
    X = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, stateful=stateful, data_format='channels_last', kernel_regularizer = l2(5e-4))(input)

    dims = X.shape
    X = Reshape(( int(dims[2]), int(dims[3]), int(dims[4])))(X)

    X = BatchNormalization()(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = MaxPooling2D()(X)

    X = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer = l2(5e-4))(X)
    X = BatchNormalization()(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = MaxPooling2D()(X)

    X = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(5e-4))(X)
    X = BatchNormalization()(X)
    X = LeakyReLU(alpha=0.1, name='middle_layer')(X)

    X = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(5e-4))(X)
    X = BatchNormalization()(X)
    X = LeakyReLU(alpha=0.1)(X)

    X = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(5e-4))(X)
    X = BatchNormalization()(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = MaxPooling2D()(X)

    logits = Conv2D(filters=256, kernel_size=(1, 1), padding='same', kernel_regularizer=l2(5e-4), activation='softmax')(X)
    return Model(input, logits)


def yolo_lstm_stage_2(input, num_anchors, num_classes):
    pass

def yolo_lstm_stage_3(input, num_anchors, num_classes):
    pass

# def yolo_lstm_model(input, num_anchors, num_classes):
#     """Generate Darknet-19 model for Imagenet classification."""
#
#     X = ConvLSTM2D(filters=32, kernel_size=(3, 3),  padding='same',  return_sequences=True, stateful=True, data_format='channels_last')(input)
#     X = BatchNormalization()(X)
#     X = LeakyReLU(alpha=0.1)(X)
#
#     X = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input)
#     X = BatchNormalization()(X)
#     X = LeakyReLU(alpha=0.1)(X)
#
#     X = darknet_body()(X)
#
#
#     X = DarknetConv2D_BN_Leaky(1024, (3, 3))(X)
#     X = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(X)
#
#
#
#     return Model(input, X)