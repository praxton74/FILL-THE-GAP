import tensorflow as tf
import os
import numpy as np
from utils.sequence_generator import data_generator
from model.unet import fused_unet
from utils.visualise import display

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

val_split = 0.1
val_indices = int(len(x_train) * val_split)
new_x_train, new_y_train = x_train[val_indices:], y_train[val_indices:] # 45000
x_val, y_val = x_train[:val_indices], y_train[:val_indices] # 5000


batch_size = 2
img_size = (32,32)
epoch = 1

# Make data generators
train_gen = data_generator(batch_size, img_size, new_x_train, new_x_train)
val_gen = data_generator(batch_size, img_size, x_val, x_val)
test_gen = data_generator(batch_size, img_size, x_test, x_test)

print("Length of training steps:", len(train_gen))
print("Length of test steps:", len(val_gen))
print("Length of test steps:", len(test_gen))


# Model 
model = fused_unet()
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True, to_file='./asset/fused_model.png')


def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (tf.keras.backend.sum(y_true_f + y_pred_f))


tf.keras.backend.clear_session()
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=[dice_coef])


model_history = model.fit(train_gen, validation_data=val_gen, epochs=epoch, batch_size = batch_size, shuffle=True, 
            steps_per_epoch=len(train_gen), validation_steps=len(val_gen))

# plotting the varoius metrics
import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(model_history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()


model.evaluate(test_gen, batch_size=batch_size)

model.save('my_model_5epochs_cifar')

# HELP FROM:
# https://github.com/ayulockin/deepimageinpainting/blob/master/Image_Inpainting_Autoencoder_Decoder_v2_0.ipynb
# https://keras.io/examples/vision/oxford_pets_image_segmentation/