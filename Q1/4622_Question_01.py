import cv2
import numpy as np
import matplotlib.pyplot as plt

def custom_average_filter(image, kernel_size):
    # Calculate padding based on the kernel size
    pad = kernel_size // 2
    
    # Create an empty output image with the same dimensions and type
    output_image = np.zeros_like(image, dtype=np.float32)
    
    # Pad the image to handle the edges based on whether it is RGB or Grayscale
    if len(image.shape) == 3:
        padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    else:
        padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')
        
    height, width = image.shape[:2]
    
    # Slide the kernel over every pixel
    for y in range(height):
        for x in range(width):
            if len(image.shape) == 3:
                # Extract the region of interest for RGB
                roi = padded_image[y : y + kernel_size, x : x + kernel_size, :]
                output_image[y, x] = np.mean(roi, axis=(0, 1))
            else:
                # Extract the region of interest for Grayscale
                roi = padded_image[y : y + kernel_size, x : x + kernel_size]
                output_image[y, x] = np.mean(roi)
                
    return output_image.astype(np.uint8)

# 1. Load the original image
# Ensure 'image_1.png' or 'image_1.jpg' is in your working directory
img = cv2.imread('images/image_1.jpg') 

if img is None:
    raise FileNotFoundError("Image not found. Please check the file path.")

# Convert from BGR (OpenCV default) to RGB for correct color display in Matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Define the requested kernel sizes [cite: 27, 28]
# Extracting just the integer since our custom function takes a single int
kernel_sizes = [3, 5, 11, 15] 

# 3. Apply the custom filters and store the results
filtered_images = []
print("Processing images... this might take a few moments for larger kernels!")

for k in kernel_sizes:
    print(f"Applying {k}x{k} average filter...")
    blurred_img = custom_average_filter(img_rgb, k) 
    filtered_images.append((k, blurred_img))

print("Processing complete! Generating plot...")

# 4. Visualize the Observations and Results
plt.figure(figsize=(15, 8))

# Plot Original Image
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# Plot Filtered Images
for i, (k, blurred_img) in enumerate(filtered_images):
    plt.subplot(2, 3, i + 2)
    plt.imshow(blurred_img)
    plt.title(f'Custom Average Filter {k}x{k}')
    plt.axis('off')

plt.tight_layout()
plt.show()