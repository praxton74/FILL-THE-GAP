import cv2
import matplotlib.pyplot as plt
import numpy as np


def display(orig_image, test_image, pred_image):
    orig_image = np.squeeze(orig_image) * 255
    test_image = np.squeeze(test_image) * 255
    pred_image = np.squeeze(pred_image) * 255

    
    orig_image = orig_image.astype('uint8')
    test_image = test_image.astype('uint8')
    pred_image = pred_image.astype('uint8')
    
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    pred_image = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(20, 15))

    # setting values to rows and column variables
    rows = 5
    columns = 5

    fig.add_subplot(rows, columns, 1)
    plt.imshow(orig_image)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title("Original Image")

    fig.add_subplot(rows, columns, 2)
    plt.imshow(test_image)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title("Masked Image")

    fig.add_subplot(rows, columns, 3)
    plt.imshow(pred_image)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title("Reconstructed Image")

    plt.show()

