import cv2
import numpy as np
import easyocr
import re
import os
import time
import imutils
import keras_ocr
import glob
import keras_ocr.tools

start_time = time.time()
pipeline = keras_ocr.pipeline.Pipeline()
reader = easyocr.Reader(['en'])
img = cv2.imread('./input/XYPMenu-Original.png')
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

def process_image_hsv_color(image):
    raw_image = imutils.resize(image, width=1000)
    raw_image2 = raw_image.copy()
    img_hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)
    cv2.imwrite('./hsv_color/img_hsv.jpg', img_hsv)
    # cv2.imshow("image_hsv",img_hsv)
    # cv2.waitKey(0)
    mask = cv2.inRange(img_hsv, (170, 170, 170), (255,255,255))
    cv2.imwrite('./hsv_color/img_mask.jpg',mask)
    # cv2.imshow("mask",mask)
    # cv2.waitKey(0)

    contours = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    for n, contour in enumerate(contours):
        rect = cv2.boundingRect(contour)
        x,y,w,h = rect
        area = w*h
        if area >= 500:
            # cv2.rectangle(raw_image, (x,y), (x+w, y+h), (0,255,0),2)
            cv2.rectangle(raw_image2, (x,y), (x+w, y+h), (0,255,0),2)
            cropped_region = raw_image[y:y+h, x:x+w]
            cv2.imwrite('./contours/cropped'+str(n)+'.jpg', cropped_region)
            confirmed_contours.append(contour)
            radius = 220 
            cropped_region2 = raw_image[y-10:y+radius-72, x:x+radius].copy()
            cv2.imwrite('./output/cropped'+str(n)+'.jpg', cropped_region2)
            # cv2.imshow("raw_image2", cropped_region)
            # cv2.waitKey(0)
            # deskewed = deskew(cropped_region)
            # denoised = remove_noise(cropped_region)
            # Perform OCR on the cropped region
            # result = reader.readtext(cropped_region)
            # for (bbox, text, prob) in result:
            #     cv2.imwrite('./ocr_roi/ocr_cropped_region'+str(n)+'.jpg', cropped_region)

            #     ocr = f'Text: {text}, Probability: {prob:.3f}'
            #     ocr_results.append(ocr)
            #     print(ocr)
    
    # # cv2.imshow("image",raw_image)
    # # cv2.waitKey(0)
    # return confirmed_coordinates
                
def perform_keras_ocr():
    # List of image paths in the folder
    image_paths = glob.glob(os.path.join('contours', '*.jpg'))  # You can add more patterns if needed
    # Read images using keras_ocr
    output_images = [keras_ocr.tools.read(image_path) for image_path in image_paths]
    predictions = pipeline.recognize(output_images)
    # Print out the results
    for image_path, prediction in zip(image_paths, predictions):
        print(f"Results for {os.path.basename(image_path)}:")
        for text, box in prediction:
            ocr = f'- Detected text: {text}'
            print(ocr)
            ocr_results.append(ocr)




                        
confirmed_contours = []
ocr_results = []
contours_coordinates = process_image_hsv_color(img)
perform_keras_ocr()


with open('./output/OCR_Result.txt', 'w') as file:
    for result in ocr_results:
        file.write(result+'\n')


end_time = time.time()  # Record the end time
execution_time = end_time - start_time  # Calculate the execution time
print(f"It took {execution_time} seconds")