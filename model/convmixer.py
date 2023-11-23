from tensorflow.keras import layers
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x


def get_conv_mixer_256_8(image_size=32, filters=256, depth=2, kernel_size=5, patch_size=2):
    inputs = keras.Input((image_size, image_size, 3))
    x = conv_stem(inputs, filters, patch_size)
    for i in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    outputs = layers.AveragePooling2D((2, 2))(x) 
    return keras.Model(inputs, outputs)
