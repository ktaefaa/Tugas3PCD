import cv2
import numpy as np
import glob

# Load the images
images = []
for img_file in glob.glob('img_avg.jfif'):
    img = cv2.imread(img_file)
    images.append(img)

# Calculate the average image
average = np.zeros(images[0].shape, np.float32)
for img in images:
    average += img / len(images)

# Convert the average image to uint8 format
average = np.uint8(average)

cv2.imshow('Original Image', img)
# Show the average image
cv2.imshow('Average Image', average)
cv2.waitKey(0)
cv2.destroyAllWindows()