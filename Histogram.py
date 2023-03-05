import cv2
import matplotlib.pyplot as plt

#membaca gambar
gmbr = cv2.imread ('img3.png', 0)

#menghitung histogram
histogram = cv2.calcHist([gmbr], [0], None, [250], [0, 256])

#menampilkan histogram
plt.plot (histogram, color='gray')
plt.title ('Histogram Gambar Kontras yang Tinggi')
plt.xlabel ('Intensitas Pixel')
plt.ylabel ('Jumlah Pixel')
plt.show()