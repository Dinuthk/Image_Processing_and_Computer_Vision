import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

img = cv2.imread('../images/Image_4.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    img = cv2.imread('../images/Image_4.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error")
        exit()

blur = cv2.GaussianBlur(img, (5, 5), 0)

mask_organs = cv2.inRange(blur, 80, 190)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned_organs = cv2.morphologyEx(mask_organs, cv2.MORPH_OPEN, kernel, iterations=2)
cleaned_organs = cv2.morphologyEx(cleaned_organs, cv2.MORPH_CLOSE, kernel, iterations=2)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_organs, connectivity=8)

output_colored = np.full((img.shape[0], img.shape[1], 3), (40, 40, 40), dtype=np.uint8)

np.random.seed(42)
colors = np.random.randint(50, 255, size=(num_labels, 3), dtype=np.uint8)

for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] > 300:
        output_colored[labels == i] = colors[i]

_, mask_bones = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
mask_bones = cv2.morphologyEx(mask_bones, cv2.MORPH_OPEN, kernel, iterations=1)

num_bone_labels, bone_labels, bone_stats, _ = cv2.connectedComponentsWithStats(mask_bones, connectivity=8)

for i in range(1, num_bone_labels):
    if bone_stats[i, cv2.CC_STAT_AREA] > 100:
        output_colored[bone_labels == i] = (255, 50, 150) 

if not os.path.exists('result'):
    os.makedirs('result')

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Intensity Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(output_colored, cv2.COLOR_BGR2RGB))
plt.title('Isolated Organs')
plt.axis('off')

plt.tight_layout()
plt.savefig('result/Question_08_output.png')
plt.show()