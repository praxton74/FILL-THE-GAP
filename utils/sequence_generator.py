from tensorflow import keras
import numpy as np
import cv2

class data_generator(keras.utils.Sequence):

    def __init__(self, batch_size, img_size, x_images, y_images):
        self.batch_size = batch_size
        self.img_size = img_size
        self.x_images = x_images
        self.y_images = y_images

    def _mask_image(self, image):
        mask = np.full((32,32,3), 255, np.uint8)
        r = np.random.choice(range(2,20)) # image size is 32, so taken 2 padding on sides for masking.
        start = (r, r) 
        end = (r+10, r+10) 
        mask = cv2.rectangle(mask, start, end, (0,0,0), -1) # drawing a black mask
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def __len__(self):
        return len(self.x_images) // self.batch_size # 45000//batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size

        batch_input_img = self.x_images[i : i + self.batch_size]
        batch_target_img = self.y_images[i : i + self.batch_size]
        
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        
        for i in range(batch_input_img.shape[0]): 
            img = batch_input_img[i] # each image
            img = self._mask_image(img.astype(np.uint8))
            x[i] = img

        y = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j in range(batch_target_img.shape[0]): 
            img = batch_target_img[j] # each image
            y[j] = img

        return x/255.0 , y/255.0 # normalise this /255.0
