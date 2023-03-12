import cv2

# Load image
img = cv2.imread('Afifa.jpeg', cv2.IMREAD_GRAYSCALE)

# Define kernel size
kernel_size = 3

# Apply median filter
median_filtered_img = cv2.medianBlur(img, kernel_size)

# Show output image
cv2.imshow('Original Image', img)
cv2.imshow('Median Filtered Image', median_filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()