import cv2
import numpy as np
from matplotlib import pyplot as plt

#Membaca gambar dari file
img = cv2.imread('Afifa.jpeg');

#Mengubah gambar ke dalam skala keabuan
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Menampilkan gambar asli dan gambar dalam skala keabuan
plt.subplot(1,2,1); plt.imshow(img);
plt.subplot(1,2,2); plt.imshow(img_gray);

#Melakukan operasi FFT pada gambar dalam skala keabuan
img_fft = np.fft.fft2(img_gray);

#Menggeser frekuensi rendah ke pusat gambar
img_fft_shifted = np.fft.fftshift(img_fft);

#Menghitung spektrum amplitudo dan fasa
img_fft_amp = np.log(abs(img_fft_shifted));
img_fft_phase = np.angle(img_fft_shifted);

#Menampilkan spektrum amplitudo dan fasa
plt.subplot(1,2,1); plt.imshow(img_fft_amp, cmap = 'gray');
plt.subplot(1,2,2); plt.imshow(img_fft_phase, cmap = 'gray');

#Memulihkan gambar dari operasi FFT
img_fft_restored = np.fft.ifft2(np.fft.ifftshift(img_fft_shifted));

#Menampilkan gambar yang telah dipulihkan
plt.imshow(np.abs(img_fft_restored), cmap = 'gray');
plt.show();