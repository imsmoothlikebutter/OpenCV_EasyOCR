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

pipeline = keras_ocr.pipeline.Pipeline()


def perform_keras_ocr():
    # List of image paths in the folder
    image_paths = glob.glob(os.path.join('input2', '*.png'))  # You can add more patterns if needed
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

ocr_results = []
perform_keras_ocr()