import cv2
import numpy as np
import re
import os
import time


image = cv2.imread('./input/XYPMenu-Original.png')

yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
equalized_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

gray = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2GRAY)

adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)

white_background = 255 * np.ones_like(equalized_image)
result_image = np.where(adaptive_thresh[:,:,None] == 255, white_background, equalized_image)

cv2.imwrite('./test/01-contrast_increased_colored.jpg', equalized_image)
cv2.imwrite('./test/02-adaptive_threshold.jpg', adaptive_thresh)
cv2.imwrite('./test/03-final_result_colored.jpg', result_image)