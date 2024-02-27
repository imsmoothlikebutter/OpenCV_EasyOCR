import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

start = time.time()

def extract_color_palette(image_path, k=8):
    # Load the image
    image = cv2.imread(image_path)
    # Convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to be a list of pixels
    pixels = image.reshape((-1, 3))

    # Apply k-means clustering to find k colors
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Get the RGB values of the cluster centers
    palette = kmeans.cluster_centers_
    print(palette)

    # Display the color palette
    plt.figure(figsize=(8, 2))
    plt.imshow([palette.astype(np.uint8)])
    plt.axis('off')
    plt.show()

# Path to your image
# image_path = './input/XYPMenu-Original.png'
image_path = './input/SecondMenu-02.jpg'
# Extract and display the color palette with k dominant colors
extract_color_palette(image_path, k=8)

end = time.time()

print(f'it took {end-start} seconds')
