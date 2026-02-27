import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../images/Image_3.jpg') 
if img is None:
    print("Error: Could not load image. Check the file path.")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Generating Gaussian Pyramid...")
gaussian_pyramid = [img_rgb]
G = img_rgb.copy()

for i in range(3):
    G = cv2.pyrDown(G)
    gaussian_pyramid.append(G)

print("Generating Laplacian Pyramid...")
laplacian_pyramid = [gaussian_pyramid[-1]]

for i in range(3, 0, -1):
    gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i])
    
    target_size = (gaussian_pyramid[i-1].shape[1], gaussian_pyramid[i-1].shape[0])
    gaussian_expanded = cv2.resize(gaussian_expanded, target_size)
    
    laplacian = cv2.subtract(gaussian_pyramid[i-1], gaussian_expanded)
    laplacian_pyramid.append(laplacian)

laplacian_pyramid = laplacian_pyramid[::-1]

print("Processing complete! Generating plots...")

plt.figure(figsize=(16, 8))

for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.imshow(gaussian_pyramid[i])
    plt.title(f'Gaussian Level {i}')
    plt.axis('off')

for i in range(4):
    plt.subplot(2, 4, i + 5)
    
    vis_laplacian = cv2.normalize(laplacian_pyramid[i], None, 0, 255, cv2.NORM_MINMAX)
    
    plt.imshow(vis_laplacian)
    plt.title(f'Laplacian Level {i}')
    plt.axis('off')

plt.tight_layout()
plt.show()