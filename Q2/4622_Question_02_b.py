import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../images/Image_2.jpg') 

if img is None:
    print("Error")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel_sizes = [3, 5, 11]
filtered_images = []

for k in kernel_sizes:
    filtered_img = cv2.medianBlur(img_rgb, k)
    filtered_images.append((k, filtered_img))

plt.figure(figsize=(16, 8))

plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

for i, (k, filtered_img) in enumerate(filtered_images):
    plt.subplot(1, 4, i + 2)
    plt.imshow(filtered_img)
    plt.title(f'Median Filter {k}x{k}')
    plt.axis('off')

plt.tight_layout()
plt.show()