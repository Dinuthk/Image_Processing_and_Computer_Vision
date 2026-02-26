import cv2
import matplotlib.pyplot as plt

# ==========================================
# Main Execution for Question 04
# ==========================================

# 1. Load the original image
img = cv2.imread('../images/Image_3.jpg') 
if img is None:
    print("Error: Could not load image. Check the file path.")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Generate 3-level Gaussian Pyramid ---
print("Generating Gaussian Pyramid...")
gaussian_pyramid = [img_rgb]
G = img_rgb.copy()

# Generate Level 1, Level 2, and Level 3
for i in range(3):
    G = cv2.pyrDown(G)
    gaussian_pyramid.append(G)

# --- Generate 3-level Laplacian Pyramid ---
print("Generating Laplacian Pyramid...")
# The top of the Laplacian pyramid is the same as the top of the Gaussian pyramid
laplacian_pyramid = [gaussian_pyramid[-1]]

# Reconstruct downwards and calculate the difference
for i in range(3, 0, -1):
    gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i])
    
    # Dimensions might be slightly off by 1 pixel after upsampling due to odd dimensions.
    # We resize the expanded image to exactly match the target Gaussian level before subtraction.
    target_size = (gaussian_pyramid[i-1].shape[1], gaussian_pyramid[i-1].shape[0])
    gaussian_expanded = cv2.resize(gaussian_expanded, target_size)
    
    # Laplacian is the difference between the Gaussian level and the expanded upper level
    # We use cv2.subtract to prevent negative overflow artifacts in uint8 arrays
    laplacian = cv2.subtract(gaussian_pyramid[i-1], gaussian_expanded)
    laplacian_pyramid.append(laplacian)

# Reverse the Laplacian list so it goes from Level 0 to Level 3 (to match Gaussian array)
laplacian_pyramid = laplacian_pyramid[::-1]

print("Processing complete! Generating plots...")

# ==========================================
# Visualization
# ==========================================
plt.figure(figsize=(16, 8))

# Plot Gaussian Pyramid
for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.imshow(gaussian_pyramid[i])
    plt.title(f'Gaussian Level {i}')
    plt.axis('off')

# Plot Laplacian Pyramid
for i in range(4):
    plt.subplot(2, 4, i + 5)
    
    # Laplacian images have lots of zero (black) and negative values, 
    # so we normalize them for better visualization in Matplotlib.
    vis_laplacian = cv2.normalize(laplacian_pyramid[i], None, 0, 255, cv2.NORM_MINMAX)
    
    plt.imshow(vis_laplacian)
    plt.title(f'Laplacian Level {i}')
    plt.axis('off')

plt.tight_layout()
plt.show()