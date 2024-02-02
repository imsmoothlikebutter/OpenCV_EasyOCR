import cv2
import numpy as np
import easyocr
import re
import os

reader = easyocr.Reader(['en'])
img = cv2.imread('./input/testImage.jpg')
original_img_rgb = img.copy()

output_folder = './output'
ocr_roi_folder = './ocr_roi'

output_files = os.listdir(output_folder)
ocr_roi_files = os.listdir(ocr_roi_folder)


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


#greyscale
def get_greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#gaussianBlur
def gaussian_blur(image):
    return cv2.GaussianBlur(image, (7,7), cv2.BORDER_DEFAULT)

#noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)

#thresholding
def thresholding(image):
    # return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)[1]

def get_contours(image):
    return cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

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

grey = get_greyscale(img)
gaussian = gaussian_blur(grey)
thresh = thresholding(gaussian)
cv2.imwrite('threshold.jpg', thresh)
contours = get_contours(thresh)
sorted_contours = sort_contours_desc(contours)

# N = 5
# largest_contours = sorted_contours[:N]
largest_contours = sorted_contours
confirmed_contours = []

# Draw contours on the original image (for visualization)
# cv2.drawContours(original_img_rgb, largest_contours, -1, (0, 255, 0), 3)
ocr_results = []

# Iterate through the largest contours
for n, contour in enumerate(largest_contours):
    # Get the coordinates of the bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Crop the region from the original image
    cropped_region = thresh[y:y+h, x:x+w]
    denoised = remove_noise(cropped_region)
    # Perform OCR on the cropped region
    result = reader.readtext(denoised)
    for (bbox, text, prob) in result:
        text = re.sub(r'[^0-9]', '', text)
        if(text):
            confirmed_contours.append(contour)
            cv2.imwrite('./ocr_roi/ocr_cropped_region'+str(n)+'.jpg', denoised)
            # Get the centroid (center) of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # Define a region around the centroid
            radius = 400  # Adjust the radius as needed to control the region size
            x = max(0, cX - radius)
            y = max(0, cY - radius)
            w = min(thresh.shape[1], cX + radius) - x
            h = min(thresh.shape[0], cY + radius) - y

            # Crop the region from the original image
            cropped_region2 = img[y:y+h, x:x+w]
            cv2.imwrite('./output/cropped'+str(n)+'.jpg', cropped_region2)
            ocr = f'Text: {text}, Probability: {prob}'
            ocr_results.append(ocr)
            print(ocr)

with open('./output/OCR_Result.txt', 'w') as file:
    for result in ocr_results:
        file.write(result+'\n')



cv2.drawContours(original_img_rgb, confirmed_contours, -1, (0, 255, 0), 3)