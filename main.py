import cv2
import numpy as np
import easyocr
import re
import os
import time

start_time = time.time()

reader = easyocr.Reader(['en'])
img = cv2.imread('./input/testImage25.jpeg')
if img is None:
    print('Please check input image!')
    exit()

original_img_rgb = img.copy()
contour_showcase = img.copy()

output_folder = './output'
ocr_roi_folder = './ocr_roi'
contour_folder = './contours'

output_files = os.listdir(output_folder)
ocr_roi_files = os.listdir(ocr_roi_folder)
contour_files = os.listdir(contour_folder)


for file in output_files:
    file_path = os.path.join(output_folder, file)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

for file in ocr_roi_files:
    file_path = os.path.join(ocr_roi_folder, file)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

for file in contour_files:
    file_path = os.path.join(contour_folder, file)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")


#greyscale
def convert_to_greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#gaussianBlur
def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (7,7), cv2.BORDER_DEFAULT)

#noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)

#thresholding
def apply_threshold(image):
    # return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)[1]

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

def get_contours(image):
    return cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

def sort_contours_desc(contours):
    return sorted(contours, key=cv2.contourArea, reverse=True)

#dilation
def dilation(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image>0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h,w) = image.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

def calculate_white_black_ratio(image, contour):
    # Crop the image to the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)
    cropped = image[y:y+h, x:x+w]
    
    # Convert to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get binary image
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Count white and black pixels
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)
    
    # Calculate the ratio
    ratio = white_pixels / black_pixels if black_pixels != 0 else float('inf')
    
    return ratio

def get_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

grey = convert_to_greyscale(img)
gaussian = apply_gaussian_blur(grey)
thresh = apply_threshold(gaussian)
# thresh = adaptive_thresholding(gaussian)
cv2.imwrite('./threshold/threshold.jpg', thresh)
# denoised = remove_noise(thresh)
contours = get_contours(thresh)
sorted_contours = sort_contours_desc(contours)

# N = 5
# largest_contours = sorted_contours[:N]
# largest_contours = sorted_contours
confirmed_contours = []

# Draw contours on the original image (for visualization)
# cv2.drawContours(original_img_rgb, largest_contours, -1, (0, 255, 0), 3)
ocr_results = []

# Specify text properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_thickness = 2
text = "Text: {text}, Probability: {prob}, B/W Ratio: {ratio}, Area: {area}"
text_color = (0, 0, 0)  # Black color

# Calculate the width and height of the text box
(text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
max_area_threshold = 2000
# Iterate through the largest contours
for n, contour in enumerate(sorted_contours):
    # Get the coordinates of the bounding box
    x, y, w, h = cv2.boundingRect(contour)
    ratio = calculate_white_black_ratio(original_img_rgb, contour)
    area = cv2.contourArea(contour)
    if ratio < 2 and area < max_area_threshold:
        # Crop the region from the original image
        cropped_region = thresh[y:y+h, x:x+w]
        deskewed = deskew(cropped_region)
        denoised = remove_noise(deskewed)
        # Perform OCR on the cropped region
        result = reader.readtext(denoised)
        for (bbox, text, prob) in result:
            text = re.sub(r'[^0-9]', '', text)
            if(text):
                if(prob > 0.3):
                    confirmed_contours.append(contour)
                    cv2.imwrite('./ocr_roi/ocr_cropped_region'+str(n)+'.jpg', cropped_region)
                    # Get the centroid (center) of the contour
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = 0, 0

                    # Define a region around the centroid
                    radius = 200  # Adjust the radius as needed to control the region size
                    x = max(0, cX - radius)
                    y = max(0, cY - radius)
                    w = min(thresh.shape[1], cX + radius) - x
                    h = min(thresh.shape[0], cY + radius) - y

                    # Crop the region from the original image
                    cropped_region2 = img[y:y+h, x:x+w].copy()
                    # Set the text start position
                    x, y = 0, 30  # You can change this to place the text at a different position

                    # Draw the white rectangle (background for the text)
                    cv2.rectangle(cropped_region2, (x, y - text_height - 10), (x + text_width, y + 10), (255, 255, 255), -1)

                    # Add the text on the image
                    cv2.putText(cropped_region2,  f'B/W Ratio: {ratio:.3f}, Area: {area}', (x, y), font, font_scale, text_color, font_thickness)
                    cv2.imwrite('./output/cropped'+str(n)+'.jpg', cropped_region2)
                    ocr = f'Text: {text}, Probability: {prob:.3f}'
                    ocr_results.append(ocr)
                    print(ocr)

with open('./output/OCR_Result.txt', 'w') as file:
    for result in ocr_results:
        file.write(result+'\n')

# Function to generate a bright color
def generate_bright_color(index):
    # We'll use a simple strategy to cycle through different bright colors
    colors = [
        (255, 0, index % 256),       # Red to Yellow (keeping Green at 0 and varying Blue)
        (255 - index % 256, 255, 0), # Yellow to Green (reducing Red and keeping Blue at 0)
        (0, 255, index % 256),       # Green to Cyan (keeping Red at 0 and varying Blue)
        (index % 256, 255 - index % 256, 255), # Cyan to Blue (varying Red and reducing Green)
        (255, 0, 255 - index % 256), # Magenta to Red (keeping Green at 0 and reducing Blue)
        (255, index % 256, 255),     # Pink to Magenta (varying Green and keeping Blue at 255)
    ]
    return colors[index % len(colors)]

for i, contour in enumerate(confirmed_contours):
    # Generate a bright color for each contour
    color = generate_bright_color(i)
    
    # Draw the contour on the image
    cv2.drawContours(contour_showcase, [contour], -1, color, 3)

# cv2.drawContours(contour_showcase, confirmed_contours, -1, (0, 255, 0), 3)
cv2.imwrite('./contours/contour-showcase.jpeg', contour_showcase)


end_time = time.time()  # Record the end time
execution_time = end_time - start_time  # Calculate the execution time
print(f"It took {execution_time} seconds")