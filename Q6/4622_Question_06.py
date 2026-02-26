import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import random
import os

img = cv2.imread('../images/Image_3.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error")
    exit()

row, col = img.shape
SP = np.zeros((row, col), dtype=np.float64)
noise_ratio = 0.05
num_noisy = int(row * col * noise_ratio)

for _ in range(num_noisy // 2):
    SP[random.randint(0, row-1), random.randint(0, col-1)] = 255

for _ in range(num_noisy // 2):
    SP[random.randint(0, row-1), random.randint(0, col-1)] = -255

laplacian = cv2.Laplacian(img, cv2.CV_64F)

I_prime = img.astype(np.float64) + SP + laplacian
I_prime = np.clip(I_prime, 0, 255).astype(np.uint8)

coeffs = pywt.wavedec2(I_prime, 'haar', level=3)
LL3 = coeffs[0]

new_coeffs = [coeffs[0]]
for detail in coeffs[1:]:
    new_coeffs.append(tuple(np.zeros_like(d) for d in detail))

reconstructed = pywt.waverec2(new_coeffs, 'haar')
reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

if not os.path.exists('result'):
    os.makedirs('result')

plt.figure(figsize=(18, 10))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image (I)')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(SP, cmap='gray')
plt.title('Salt & Pepper Noise (SP)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian L(I)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(I_prime, cmap='gray')
plt.title("Degraded Image I'")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(LL3, cmap='gray')
plt.title('Wavelet LL Subband (level 3) [haar]')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(reconstructed, cmap='gray')
plt.title('Reconstructed Smooth Image')
plt.axis('off')

plt.tight_layout()
plt.savefig('result/Question_06_output.png')
plt.show()