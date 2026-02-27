import cv2
import random
import matplotlib.pyplot as plt

def add_salt_and_pepper_noise(image, noise_ratio):
    noisy_image = image.copy()
    
    row = len(noisy_image)
    col = len(noisy_image[0])
    
    total_pixels = row * col
    num_noisy_pixels = int(total_pixels * noise_ratio)
    
    num_salt = num_noisy_pixels // 2
    num_pepper = num_noisy_pixels - num_salt
    
    for _ in range(num_salt):
        y = random.randint(0, row - 1)
        x = random.randint(0, col - 1)
        noisy_image[y][x] = 255 
        
    for _ in range(num_pepper):
        y = random.randint(0, row - 1)
        x = random.randint(0, col - 1)
        noisy_image[y][x] = 0
        
    return noisy_image

img = cv2.imread('../images/Image_2.jpg')

if img is None:
    print("Error: Could not load image. Please check if 'image_2.png' is in your folder.")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Adding 10% noise...")
noisy_10 = add_salt_and_pepper_noise(img_rgb, 0.10)

print("Adding 20% noise...")
noisy_20 = add_salt_and_pepper_noise(img_rgb, 0.20)

print("Displaying results...")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title('Original Image 2')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noisy_10)
plt.title('10% Salt & Pepper Noise')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(noisy_20)
plt.title('20% Salt & Pepper Noise')
plt.axis('off')

plt.tight_layout()
plt.show()