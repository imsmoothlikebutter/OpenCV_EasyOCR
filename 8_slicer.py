import cv2

# Read the input image
# image = cv2.imread('./input/XYPMenu-Original.png')
image = cv2.imread('thresholded.jpg')

# Get the image dimensions (height, width, and channels)
h, w, c = image.shape

# Calculate the center point
center_h, center_w = h // 2, w // 2

# Divide the image into 4 quadrants
top_left = image[0:center_h, 0:center_w]
top_right = image[0:center_h, center_w:w]
bottom_left = image[center_h:h, 0:center_w]
bottom_right = image[center_h:h, center_w:w]

# Further split each quadrant horizontally to get 8 parts
top_left_1 = top_left[0:center_h // 2, :]
top_left_2 = top_left[center_h // 2:, :]

top_right_1 = top_right[0:center_h // 2, :]
top_right_2 = top_right[center_h // 2:, :]

bottom_left_1 = bottom_left[0:center_h // 2, :]
bottom_left_2 = bottom_left[center_h // 2:, :]

bottom_right_1 = bottom_right[0:center_h // 2, :]
bottom_right_2 = bottom_right[center_h // 2:, :]

# Save the 8 parts
cv2.imwrite('./doublequarters/1.jpg', top_left_1)
cv2.imwrite('./doublequarters/2.jpg', top_left_2)
cv2.imwrite('./doublequarters/3.jpg', top_right_1)
cv2.imwrite('./doublequarters/4.jpg', top_right_2)
cv2.imwrite('./doublequarters/5.jpg', bottom_left_1)
cv2.imwrite('./doublequarters/6.jpg', bottom_left_2)
cv2.imwrite('./doublequarters/7.jpg', bottom_right_1)
cv2.imwrite('./doublequarters/8.jpg', bottom_right_2)