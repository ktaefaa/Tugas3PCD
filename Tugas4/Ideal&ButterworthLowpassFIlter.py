import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
img = cv2.imread('ImgAfifa.jpeg',0)

# Transformasi Fourier
dft = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Ukuran kernel filter
rows, cols = img.shape
crow,ccol = rows//2 , cols//2

# Filter Lowpass Ideal
mask = np.zeros((rows,cols,2),np.float32)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

# Filter Lowpass Butterworth
n = 2 # Orde filter
D0 = 30 # Cutoff frequency
nrows, ncols = img.shape
u, v = np.meshgrid(np.arange(ncols), np.arange(nrows))
du = u - ncols/2
dv = v - nrows/2
D = np.sqrt(du**2 + dv**2)
H = 1 / (1 + (D/D0)**(2*n))
H = np.fft.fftshift(H)
H = np.dstack([H]*2)
fshift_b = dft_shift * H
f_ishift_b = np.fft.ifftshift(fshift_b)
img_back_b = cv2.idft(f_ishift_b)
img_back_b = cv2.magnitude(img_back_b[:,:,0],img_back_b[:,:,1])

# Plot hasil filter
plt.subplot(2,2,1),plt.imshow(img, cmap = 'gray')
plt.title('Gambar Asli'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(img_back, cmap = 'gray')
plt.title('Ideal Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(img, cmap = 'gray')
plt.title('Gambar Asli'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(img_back_b, cmap = 'gray')
plt.title('Butterworth Lowpass Filter'), plt.xticks([]), plt.yticks([])

plt.show()
