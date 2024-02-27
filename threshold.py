import cv2
import numpy as np
import easyocr
import re
import os
import time
import imutils


def convert_to_greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_threshold(image):
    # return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)[1]

def adaptive_thresholding(image):
    return cv2.adaptiveThreshold(
        image, 
        255, 
        # cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY_INV, 
        11,  # Block size (size of the neighborhood area)
        2    # Constant subtracted from the mean
    )



img = cv2.imread('./input/XYPMenu-Original.png')
image = img.copy()
if img is None:
    print('Please check input image!')
    exit()
greyed = convert_to_greyscale(img)
thresholded = adaptive_thresholding(greyed)
# thresholded = apply_threshold(img)

# # Define a horizontal kernel for morphological operations
# kernel_length = max(2, np.array(greyed).shape[1]//30)
# horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

# # Use morphological operations to emphasize horizontal lines
# processed = cv2.erode(thresholded, horizontal_kernel, iterations=1)
# processed = cv2.dilate(processed, horizontal_kernel, iterations=1)

# # Optionally, apply a smoothing filter
# # processed = cv2.GaussianBlur(processed, (5, 5), 0)

# # Show the processed image
# cv2.imshow('Horizontal Line Length Smoothing', processed)
# cv2.waitKey(0)

# contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# for contour in contours:
#     # Get the bounding rectangle for each contour
#     x, y, w, h = cv2.boundingRect(contour)
#     if w == h:
#     # Draw the bounding rectangle on the original (or a copy of the) image
#     # Change 'image' to the name of your original image variable
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# cv2.imshow('Text Areas', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


cv2.imwrite('thresholded.jpg', thresholded)

thresholded_inv = cv2.bitwise_not(thresholded)

cv2.imwrite('thresholded_inv.jpg', thresholded_inv)