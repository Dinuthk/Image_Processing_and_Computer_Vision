import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

img = cv2.imread('../images/Image_6.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    img = cv2.imread('../images/Image_6.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error")
        exit()

_, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

kernel = np.ones((5, 5), np.uint8)

erosion = cv2.erode(binary_img, kernel, iterations=1)
dilation = cv2.dilate(binary_img, kernel, iterations=1)
opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Extracted Morphological Features (Based on Opened Image):")
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    print(f"Object {i+1}: Area = {area:.2f}, Perimeter = {perimeter:.2f}")

if not os.path.exists('result'):
    os.makedirs('result')

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(binary_img, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(erosion, cmap='gray')
plt.title('Erosion')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(dilation, cmap='gray')
plt.title('Dilation')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(opening, cmap='gray')
plt.title('Opening')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(closing, cmap='gray')
plt.title('Closing')
plt.axis('off')

plt.tight_layout()
plt.savefig('result/Question_10_output.png')
plt.show()