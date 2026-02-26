import cv2
import random
import matplotlib.pyplot as plt

def add_salt_and_pepper_noise(image, noise_ratio):
    """
    Adds salt and pepper noise to an image from scratch.
    'image' is the loaded image array.
    'noise_ratio' is a float (e.g., 0.10 for 10%, 0.20 for 20%).
    """
    noisy_image = image.copy()
    
    # Get the height (rows) and width (cols) of the image
    row = len(noisy_image)
    col = len(noisy_image[0])
    
    # Calculate the exact number of pixels to modify
    total_pixels = row * col
    num_noisy_pixels = int(total_pixels * noise_ratio)
    
    # Split the noise evenly between salt (white) and pepper (black)
    num_salt = num_noisy_pixels // 2
    num_pepper = num_noisy_pixels - num_salt
    
    # Add Salt (White pixels: 255)
    for _ in range(num_salt):
        y = random.randint(0, row - 1)
        x = random.randint(0, col - 1)
        noisy_image[y][x] = 255 
        
    # Add Pepper (Black pixels: 0)
    for _ in range(num_pepper):
        y = random.randint(0, row - 1)
        x = random.randint(0, col - 1)
        noisy_image[y][x] = 0
        
    return noisy_image

# ==========================================
# Main Execution for Part (a)
# ==========================================

# 1. Load the original image
# Make sure 'image_2.png' or 'image_2.jpg' is in your directory!
img = cv2.imread('../images/Image_2.jpg')

if img is None:
    print("Error: Could not load image. Please check if 'image_2.png' is in your folder.")
    exit()

# Convert from BGR to RGB so colors look right in Matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Adding 10% noise...")
# (i) Add salt and pepper noise to 10% of all pixels [cite: 32, 33]
noisy_10 = add_salt_and_pepper_noise(img_rgb, 0.10)

print("Adding 20% noise...")
# (ii) Add salt and pepper noise to 20% of all pixels [cite: 32, 34]
noisy_20 = add_salt_and_pepper_noise(img_rgb, 0.20)

print("Displaying results...")

# 2. Visualize the results
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