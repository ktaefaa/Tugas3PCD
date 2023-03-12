import cv2
import numpy as np

# Load image
img = cv2.imread('Afifa.jpeg')

# Define kernel size
kernel_size = (3, 3)

# Define minimum filter kernel
min_kernel = np.ones(kernel_size)

# Normalize kernel
min_kernel /= np.sum(min_kernel)

# Apply minimum filter to the image
filtered_img = cv2.filter2D(img, -1, min_kernel)

#Show output image
cv2.imshow('Original Image', img)
cv2.imshow('Min Filtered Image', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
