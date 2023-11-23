import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, Input, Add, BatchNormalization

import numpy as np
from model.convmixer import get_conv_mixer_256_8

def fused_unet():

    input_layer = Input((32, 32, 3))

    conv3 = Conv2D(64, (3, 3), activation="relu", padding="same")(input_layer)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = BatchNormalization()(pool3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool3)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = BatchNormalization()(pool4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv2D(256, (3, 3), activation="relu", padding="same")(pool4)
    
    deconv4 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = Add()([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)

    deconv3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = Add()([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    
    output_layer = Conv2D(3, (3,3), padding="same", activation="sigmoid")(uconv3)
    
    return keras.Model(input_layer, output_layer)
