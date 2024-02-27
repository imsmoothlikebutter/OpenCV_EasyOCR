import cv2

image = cv2.imread('./input/XYPMenu-Original.png')

h, w, c = image.shape

center_h, center_w = h // 2, w // 2

top_left = image[0:center_h, 0:center_w]
top_right = image[0:center_h, center_w:w]
bottom_left = image[center_h:h, 0:center_w]
bottom_right = image[center_h:h, center_w:w]

cv2.imwrite('./quarters/1.jpg', top_left)
cv2.imwrite('./quarters/2.jpg', top_right)
cv2.imwrite('./quarters/3.jpg', bottom_left)
cv2.imwrite('./quarters/4.jpg', bottom_right)
