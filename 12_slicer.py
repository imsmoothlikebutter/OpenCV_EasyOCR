import cv2

# Read the input image
image = cv2.imread('./input/XYPMenu-Original.png')

# Get the image dimensions (height, width)
h, w = image.shape[:2]

# Calculate the size of each part (3 rows, 4 columns)
part_h, part_w = h // 3, w // 4

# Create a list to hold the parts
parts = []

# Loop over the image in a 3x4 grid
for i in range(3):
    for j in range(4):
        # Calculate the start and end points of the current part
        start_h, start_w = i * part_h, j * part_w
        end_h, end_w = (i + 1) * part_h, (j + 1) * part_w
        
        # Slice the image to create the part
        part = image[start_h:end_h, start_w:end_w]
        
        # Add the part to the list
        parts.append(part)
        
        # Save the part to disk
        cv2.imwrite(f'./twelve/part_{i*4+j+1}.jpg', part)
