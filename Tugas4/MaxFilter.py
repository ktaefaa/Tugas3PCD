import cv2
import numpy as np

# Load image
img = cv2.imread("Afifa.jpeg")

# Define kernel size
kernel_size = (3, 3)

# Apply maximum filter to the image
max_filtered_img = cv2.dilate(img, np.ones(kernel_size), iterations=1)

# Show output image
cv2.imshow('Original Image', img)
cv2.imshow('Max Filtered Image', max_filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()