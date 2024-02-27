import cv2
import numpy as np

image = cv2.imread('./input/XYPMenu-Original.png')

# Create a synthetic image with a solid target RGB color
# target_rgb_color = (237, 27, 54)
# target_rgb_color = (255, 255, 255)
target_rgb_color = (38, 121, 30)
# image = np.zeros((100, 100, 3), dtype=np.uint8)
# image[:] = target_rgb_color  # Fill the image with the target color

# Convert the target RGB color to HSV
target_hsv_color = cv2.cvtColor(np.uint8([[target_rgb_color]]), cv2.COLOR_RGB2HSV)[0][0]

# Convert the synthetic image to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define a range for the HSV color
sensitivity = 15
lower_hsv = np.array([target_hsv_color[0] - sensitivity, 100, 100])
upper_hsv = np.array([target_hsv_color[0] + sensitivity, 255, 255])

# Create a mask
mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

# Apply the mask to the original image (optional)
result_image = cv2.bitwise_and(image, image, mask=mask)


# Define thresholds for black and white
black_lower_hsv = np.array([0, 0, 0])
black_upper_hsv = np.array([180, 255, 50])  # Hue range is 0-180 in OpenCV, saturation 0-255, value 0-50 for black

white_lower_hsv = np.array([0, 0, 200])
white_upper_hsv = np.array([180, 55, 255])  # Low saturation (0-55) and high value (200-255) for white

# Create masks
black_mask = cv2.inRange(hsv_image, black_lower_hsv, black_upper_hsv)
white_mask = cv2.inRange(hsv_image, white_lower_hsv, white_upper_hsv)

# Optional: Apply masks to isolate black and white areas
black_areas = cv2.bitwise_and(image, image, mask=black_mask)
white_areas = cv2.bitwise_and(image, image, mask=white_mask)

# Display the masks and results
cv2.imshow('Black Mask', black_mask)
cv2.imshow('White Mask', white_mask)
cv2.imshow('Black Areas', black_areas)
cv2.imshow('White Areas', white_areas)

# Display the mask
# Note: In a Jupyter environment, consider using matplotlib to display images inline
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()