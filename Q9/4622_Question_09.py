import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

img = cv2.imread('../images/Image_5.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    img = cv2.imread('../images/Image_5.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error")
        exit()

eq_img = cv2.equalizeHist(img)

min_val = np.min(img)
max_val = np.max(img)
stretched_img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

gaussian_img = cv2.GaussianBlur(img, (5, 5), 0)

if not os.path.exists('result'):
    os.makedirs('result')

plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title('Original MRI (Image 5)')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(eq_img, cmap='gray')
plt.title('Histogram Equalization')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(stretched_img, cmap='gray')
plt.title('Contrast Stretching')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(gaussian_img, cmap='gray')
plt.title('Gaussian Filtering')
plt.axis('off')

plt.tight_layout()
plt.savefig('result/Question_09_output.png')
plt.show()