import cv2
import pytesseract
import numpy as np
from pytesseract import Output

# img = cv2.imread('test2.png')
img = cv2.imread('test3.jpg')
original_img_rgb = img.copy()

#Custom Options
custom_config = r'--oem 3 --psm 6'
# result = pytesseract.image_to_string(img,config=custom_config);
# print(result)

#greyscale
def get_greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
thresh = thresholding(grey)
cv2.imwrite('threshold.jpg', thresh)

cv2.imshow('thresh', thresh)
cv2.waitKey(0)
# erosion = erode(thresh)
# dilate = dilation(erosion)
# opening = opening(dilate)

contours = get_contours(thresh)
sorted_contours = sort_contours_desc(contours)

N = 1
largest_contours = sorted_contours[:N]

# Draw contours on the original image (for visualization)
cv2.drawContours(original_img_rgb, largest_contours, -1, (0, 255, 0), 3)

# Display the result
cv2.imshow('Output', original_img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

result = pytesseract.image_to_string(thresh,config=custom_config);
print(result)

# h, w, c = opening.shape
# boxes = pytesseract.image_to_data(opening, output_type=Output.DICT)
# for b in boxes.splitlines():
#     b = b.split(' ')
#     img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0,255,0),2)
#     cv2.imshow('img', img)
#     cv2.waitKey(0)


# d = pytesseract.image_to_data(thresh, output_type=Output.DICT)
# print(d.keys())
# print(d['text'])
# n_boxes = len(d['text'])
# for i in range(n_boxes):
#     if int(d['conf'][i]) > 60:
#         (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#         img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# cv2.imshow('img', img)
# cv2.waitKey(0)