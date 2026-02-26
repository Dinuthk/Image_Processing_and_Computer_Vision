import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os

img = cv2.imread('../images/Image_3.jpg', cv2.IMREAD_GRAYSCALE)
watermark_img = cv2.imread('../images/watermark.png', cv2.IMREAD_GRAYSCALE)

if img is None or watermark_img is None:
    print("Error")
    exit()

coeffs = pywt.dwt2(img, 'haar')
cA, (cH, cV, cD) = coeffs

watermark_resized = cv2.resize(watermark_img, (cA.shape[1], cA.shape[0]))
_, watermark_bin = cv2.threshold(watermark_resized, 127, 1, cv2.THRESH_BINARY)
watermark_bin = watermark_bin.astype(np.float64)

alpha = 30.0
cA_wm = cA + (alpha * watermark_bin)

watermarked_img = pywt.idwt2((cA_wm, (cH, cV, cD)), 'haar')
watermarked_img = np.clip(watermarked_img, 0, 255).astype(np.uint8)

coeffs_ext = pywt.dwt2(watermarked_img, 'haar')
cA_ext, _ = coeffs_ext

extracted_continuous = (cA_ext - cA) / alpha
extracted_bin = (extracted_continuous > 0.5).astype(np.uint8) * 255

if not os.path.exists('result'):
    os.makedirs('result')

plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(watermark_resized, cmap='gray')
plt.title('Original Watermark')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(watermarked_img, cmap='gray')
plt.title('Watermarked Image')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(extracted_bin, cmap='gray')
plt.title('Extracted Watermark')
plt.axis('off')

plt.tight_layout()
plt.savefig('result/Question_07_output.png')
plt.show()