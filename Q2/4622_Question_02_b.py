import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def add_salt_and_pepper_noise(image, noise_ratio):
    """Reusing the noise function to generate the test image."""
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

def custom_median_filter(image, kernel_size):
    """
    Applies a median filter to an image from scratch.
    """
    pad = kernel_size // 2
    output_image = np.zeros_like(image, dtype=np.uint8)
    
    # Pad the image to handle the edges
    if len(image.shape) == 3:
        padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    else:
        padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')
        
    height, width = image.shape[:2]
    
    # Slide the kernel over the image
    for y in range(height):
        for x in range(width):
            if len(image.shape) == 3:
                # Process each color channel independently
                for c in range(3):
                    roi = padded_image[y : y + kernel_size, x : x + kernel_size, c]
                    output_image[y, x, c] = np.median(roi)
            else:
                roi = padded_image[y : y + kernel_size, x : x + kernel_size]
                output_image[y, x] = np.median(roi)
                
    return output_image

# ==========================================
# Main Execution for Part (b)
# ==========================================

# 1. Load the original image using the correct relative path!
img = cv2.imread('../images/Image_2.jpg') 

if img is None:
    print("Error: Could not load image. Check the file path.")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Generate the 20% noisy image to test our filters on
print("Generating 20% noisy image for filtering...")
noisy_20 = add_salt_and_pepper_noise(img_rgb, 0.20)

# 3. Apply the custom median filters
kernel_sizes = [3, 5, 11]
filtered_images = []

print("Applying custom median filters...")
print("Note: The 11x11 kernel will take a minute or two to process due to the nested loops!")

for k in kernel_sizes:
    print(f"Processing {k}x{k} median filter...")
    filtered_img = custom_median_filter(noisy_20, k)
    filtered_images.append((k, filtered_img))

print("Processing complete! Generating plots...")

# 4. Visualize the Results
plt.figure(figsize=(16, 8))

plt.subplot(1, 4, 1)
plt.imshow(noisy_20)
plt.title('Original 20% Noisy Image')
plt.axis('off')

for i, (k, filtered_img) in enumerate(filtered_images):
    plt.subplot(1, 4, i + 2)
    plt.imshow(filtered_img)
    plt.title(f'Median Filter {k}x{k}')
    plt.axis('off')

plt.tight_layout()
plt.show()