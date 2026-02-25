import cv2
import matplotlib.pyplot as plt

# 1. Load the original image
# Note: Update the extension if image_1 is a .jpg or .tif
img = cv2.imread('images/image_1.jpg') 

# Convert from BGR (OpenCV default) to RGB for correct color display in Matplotlib
# (If it's a grayscale image, you can load it with cv2.IMREAD_GRAYSCALE)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply the Average Filters

# 2. Define the requested kernel sizes
kernel_sizes = [(3, 3), (5, 5), (11, 11), (15, 15)]

# 3. Apply the filters and store the results
filtered_images = []
for k in kernel_sizes:
    blurred_img = cv2.blur(img_rgb, k)
    filtered_images.append((k, blurred_img))

# Visualize the Observations and Results

# 4. Set up the Matplotlib figure
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
    plt.title(f'Average Filter {k[0]}x{k[1]}')
    plt.axis('off')

plt.tight_layout()
plt.show()