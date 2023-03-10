import cv2

# Load the two images
img1 = cv2.imread('img1.jpg')
img2 = cv2.imread('img2.jpg')

# Convert the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Calculate the absolute difference between the two grayscale images
diff = cv2.absdiff(gray1, gray2)

# Show the difference image
cv2.imshow('Difference Image', diff)
cv2.waitKey(0)
cv2.destroyAllWindows()