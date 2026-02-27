import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    sum_val = 0.0
    
    for y in range(size):
        for x in range(size):
            rx = x - center
            ry = y - center
            kernel[y, x] = np.exp(-(rx**2 + ry**2) / (2 * sigma**2))
            sum_val += kernel[y, x]
            
    kernel /= sum_val
    return kernel

def custom_gaussian_filter(image, kernel):
    k_size = kernel.shape[0]
    pad = k_size // 2
    output_image = np.zeros_like(image, dtype=np.float32)
    
    if len(image.shape) == 3:
        padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    else:
        padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')
        
    height, width = image.shape[:2]
    
    for y in range(height):
        for x in range(width):
            if len(image.shape) == 3:
                roi = padded_image[y : y + k_size, x : x + k_size, :]
                output_image[y, x] = np.sum(roi * kernel[:, :, np.newaxis], axis=(0, 1))
            else:
                roi = padded_image[y : y + k_size, x : x + k_size]
                output_image[y, x] = np.sum(roi * kernel)
                
    return np.clip(output_image, 0, 255).astype(np.uint8)

img = cv2.imread('../images/Image_3.jpg') 
if img is None:
    print("Error: Could not load image. Check the file path.")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel_sizes = [3, 5, 11, 15]
fixed_sigma = 2.0
size_results = []

print("Running Kernel Size variations... (The 15x15 will take a moment!)")
for k in kernel_sizes:
    print(f"Applying {k}x{k} Gaussian filter (sigma={fixed_sigma})...")
    gauss_kernel = generate_gaussian_kernel(k, fixed_sigma)
    filtered_img = custom_gaussian_filter(img_rgb, gauss_kernel)
    size_results.append((k, filtered_img))

fixed_kernel_size = 15
sigma_values = [1.0, 3.0, 6.0]
sigma_results = []

print(f"\nRunning Sigma variations on a {fixed_kernel_size}x{fixed_kernel_size} kernel...")
for sig in sigma_values:
    print(f"Applying Gaussian filter with sigma={sig}...")
    gauss_kernel = generate_gaussian_kernel(fixed_kernel_size, sig)
    filtered_img = custom_gaussian_filter(img_rgb, gauss_kernel)
    sigma_results.append((sig, filtered_img))

print("Processing complete! Generating plots...")

plt.figure(figsize=(18, 10))

plt.subplot(2, 5, 1)
plt.imshow(img_rgb)
plt.title('Original Image 3')
plt.axis('off')

for i, (k, filtered_img) in enumerate(size_results):
    plt.subplot(2, 5, i + 2)
    plt.imshow(filtered_img)
    plt.title(f'Size: {k}x{k}, $\sigma$={fixed_sigma}')
    plt.axis('off')

plt.subplot(2, 5, 6)
plt.imshow(img_rgb)
plt.title('Original Image 3')
plt.axis('off')

for i, (sig, filtered_img) in enumerate(sigma_results):
    plt.subplot(2, 5, i + 7)
    plt.imshow(filtered_img)
    plt.title(f'Size: {fixed_kernel_size}x{fixed_kernel_size}, $\sigma$={sig}')
    plt.axis('off')

plt.tight_layout()
plt.show()