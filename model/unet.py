import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, Input, Add, BatchNormalization

import numpy as np
from model.convmixer import get_conv_mixer_256_8

def fused_unet():
    """
    This is the simplest model that could be made.
    Uncomment the layers below to increase the size of the model.
    Help taken from: https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5
    """

    input_layer = Input((32, 32, 3))
    # conv1 = Conv2D(16, (3, 3), activation="relu", padding="same")(input_layer)
    # # conv1 = Conv2D(16, (3, 3), activation="relu", padding="same")(conv1)
    # pool1 = MaxPooling2D((2, 2))(conv1)
    # pool1 = Dropout(0.25)(pool1)

    # conv2 = Conv2D(32, (3, 3), activation="relu", padding="same")(pool1)
    # # conv2 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv2)
    # pool2 = MaxPooling2D((2, 2))(conv2)
    # pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(64, (3, 3), activation="relu", padding="same")(input_layer)
    # conv3 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = BatchNormalization()(pool3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool3)
    # conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = BatchNormalization()(pool4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv2D(256, (3, 3), activation="relu", padding="same")(pool4)
    # convm = Conv2D(256, (3, 3), activation="relu", padding="same")(convm)
    # print("UMAR convm shape", convm.shape)

    # Get the fusion of ConvMixer and Add it to U-Net model.
    # convmixer = get_conv_mixer_256_8(image_size=32, filters=256, depth=2, kernel_size=5, patch_size=2)
    # fusion = convmixer(input_layer)
    # print("UMAR fusion shape", fusion.shape)

    # convm = Add()([convm, fusion])
    # print("UMAR convm added shape", convm.shape)
    
    
    deconv4 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = Add()([deconv4, conv4])
    # uconv4 = BatchNormalization()(uconv4)
    uconv4 = Dropout(0.5)(uconv4)
    # uconv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(uconv4)
    # uconv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = Add()([deconv3, conv3])
    # uconv3 = BatchNormalization()(uconv3)
    uconv3 = Dropout(0.5)(uconv3)
    # uconv3 = Conv2D(64, (3, 3), activation="relu", padding="same")(uconv3)
    # uconv3 = Conv2D(64, (3, 3), activation="relu", padding="same")(uconv3)

    # deconv2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(uconv3)
    # uconv2 = Add()([deconv2, conv2])
    # uconv2 = Dropout(0.5)(uconv2)
    # # uconv2 = Conv2D(32, (3, 3), activation="relu", padding="same")(uconv2)
    # # uconv2 = Conv2D(32, (3, 3), activation="relu", padding="same")(uconv2)

    # deconv1 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(uconv2)
    # uconv1 = Add()([deconv1, conv1])
    # uconv1 = Dropout(0.5)(uconv1)
    # # uconv1 = Conv2D(16, (3, 3), activation="relu", padding="same")(uconv1)
    # # uconv1 = Conv2D(16, (3, 3), activation="relu", padding="same")(uconv1)
    
    output_layer = Conv2D(3, (3,3), padding="same", activation="sigmoid")(uconv3)
    
    return keras.Model(input_layer, output_layer)