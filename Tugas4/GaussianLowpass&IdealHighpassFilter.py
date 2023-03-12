import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
img = cv2.imread('ImageAfifa.jpeg', 0)

# Compute the 2D FFT
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Define the dimensions of the filter
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# Create a Gaussian lowpass filter
d0 = 50  # Cut-off frequency
filter_type = 'lowpass'
if filter_type == 'lowpass':
    # Gaussian lowpass filter
    H = np.zeros((rows, cols, 2), np.float32)
    for i in range(rows):
        for j in range(cols):
            H[i, j] = np.exp(-((i - crow) ** 2 + (j - ccol) ** 2) / (2 * d0 ** 2))
elif filter_type == 'highpass':
    # Ideal highpass filter
    H = np.ones((rows, cols, 2), np.float32)
    for i in range(rows):
        for j in range(cols):
            if (i-crow)**2 + (j-ccol)**2 < d0**2:
                H[i, j] = 0

# Apply the filter to the Fourier transform of the image
F = H * dft_shift

# Compute the inverse FFT
f_ishift = np.fft.ifftshift(F)
img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Display the results
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Output Image'), plt.xticks([]), plt.yticks([])
plt.show()