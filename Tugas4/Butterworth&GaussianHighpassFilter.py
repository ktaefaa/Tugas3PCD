import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load gambar
img = cv2.imread('ImageAfifa.jpeg',0)

# Transformasi Fourier
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Parameter filter
D0 = 50   # Cutoff frequency
n = 2     # Order filter

# Filter Butterworth highpass
rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)
u, v = np.meshgrid(np.arange(cols), np.arange(rows))
D = np.sqrt((u - ccol)**2 + (v - crow)**2)
H = 1 / (1 + (D0 / D)**(2*n))
butterworth_hpf = 1 - H

# Filter Gaussian highpass
sigma = 10   # Standard deviation
gaussian_hpf = 1 - np.exp(-(D**2) / (2 * (sigma**2)))

# Apply filter
fshift_butterworth = fshift * butterworth_hpf
fshift_gaussian = fshift * gaussian_hpf

# Transformasi Fourier inverse
img_back_butterworth = np.fft.ifft2(np.fft.ifftshift(fshift_butterworth))
img_back_gaussian = np.fft.ifft2(np.fft.ifftshift(fshift_gaussian))

# Konversi tipe data untuk visualisasi
img_back_butterworth = np.uint8(np.abs(img_back_butterworth))
img_back_gaussian = np.uint8(np.abs(img_back_gaussian))

# Tampilkan gambar asli dan hasil filter
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Gambar Asli'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(butterworth_hpf, cmap='gray')
plt.title('Filter Butterworth Highpass'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_back_butterworth, cmap='gray')
plt.title('Hasil Filter Butterworth Highpass'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Gambar Asli'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(gaussian_hpf, cmap='gray')
plt.title('Gaussian Highpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_back_gaussian, cmap='gray')
plt.title('Hasil Gaussian Highpass Filter'), plt.xticks([]), plt.yticks([])
plt.show()
