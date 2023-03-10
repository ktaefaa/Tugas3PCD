import cv2
import matplotlib.pyplot as plt

#membaca gambar dan melakukan equalisasi pada gambar
img = cv2.imread ('img2.jfif', 0)
equalization_img = cv2.equalizeHist (img)

#menghitung histogram gambar asli dan gambar setelah equalisasi
histogram_img = cv2.calcHist ([img], [0], None, [256], [0, 256])
equalization_histogram = cv2.calcHist ([equalization_img], [0], None, [256], [0, 256])

#menampilkan gambar asli dan setelahnya
plt.subplot(2,2,1)
plt.imshow (img, cmap='gray')
plt.title ('Gambar Asli')
plt.subplot(2,2,2)
plt.imshow (equalization_img, cmap='gray')
plt.title ('Histogram Equalization')
#menampilkan histogram gambar asli dan setelah proses equalisasi
plt.subplot (2,2,3)
plt.plot (histogram_img, color='gray')
plt.title ('Histogram Gambar Asli')
plt.xlabel ('Intensitas Pixel')
plt.ylabel ('Jumlah Pixel')
plt.subplot(2,2,4)
plt.plot (equalization_histogram, color='gray')
plt.title ('Histogram Setelah Equalization')
plt.xlabel ('Intensitas Pixel')
plt.ylabel ('Jumlah Pixel')

plt.show()