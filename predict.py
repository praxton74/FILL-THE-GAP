import tensorflow as tf
from utils.visualise import display
import numpy as np
import cv2

def mask_image(self, image):
    mask = np.full((32,32,3), 255, np.uint8)
    r = np.random.choice(range(2,20)) # image size is 32, so taken 2 padding on sides for masking.
    start = (r, r) 
    end = (r+10, r+10) 
    mask = cv2.rectangle(mask, start, end, (0,0,0), -1) # drawing a black mask
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

model = tf.keras.models.load_model('my_model_5epochs_cifar')
                                    
for i in range(0,5):
    original = x_test[i] / 255.0
    masked_image = masked_image(original)
    masked_image = masked_image[np.newaxis,...]
    pred = model.predict(masked_image)  
    display(original, masked_image, pred)



