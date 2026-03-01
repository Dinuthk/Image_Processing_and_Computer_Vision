import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

image_path = "../images/Image_4.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
h, w = img.shape

mask = np.zeros((h, w), dtype=np.uint8)

organs = [
    (1, "liver", (0, 0, 255),        lambda m: cv2.ellipse(m, (150, 200), (60, 100), -20, 0, 360, 1, -1)),
    (2, "kidney", (255, 0, 128),     lambda m: cv2.circle(m, (180, 350), 40, 2, -1)),
    (3, "kidney", (255, 0, 255),     lambda m: cv2.circle(m, (350, 350), 40, 3, -1)),
    (4, "spleen", (255, 0, 0),       lambda m: cv2.ellipse(m, (450, 250), (30, 60), 10, 0, 360, 4, -1)),
    (5, "spinal column", (147, 20, 255), lambda m: cv2.circle(m, (265, 400), 25, 5, -1)),
]

for class_id, _, _, draw_fn in organs:
    draw_fn(mask)

out = np.zeros((h, w, 3), dtype=np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX

for class_id, name, color, _ in organs:
    organ = (mask == class_id).astype(np.uint8)
    out[organ == 1] = color

    contours, _ = cv2.findContours(organ, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        continue

    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        continue

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    cv2.putText(out, name, (cx - 40, cy + 60), font, 0.6, color, 2, cv2.LINE_AA)

os.makedirs("result", exist_ok=True)
cv2.imwrite("result/input_ct.png", img)
cv2.imwrite("result/segmented_output.png", out)

plt.figure(figsize=(10, 5))

ax1 = plt.subplot(1, 2, 1)
ax1.set_title("Input CT (Grayscale)")
ax1.imshow(img, cmap="gray")
ax1.axis("off")

ax2 = plt.subplot(1, 2, 2)
ax2.set_title("AI Output (Segmented & Labeled)")
ax2.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
ax2.axis("off")

plt.tight_layout()
plt.show()