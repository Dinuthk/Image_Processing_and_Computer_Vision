import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

IMG1_PATH = "../images/Image_1.jpg"
IMG2_PATH = "../images/Image_2.jpg"
IMG3_PATH = "../images/Image_3.jpg"

AVG_K = 15
MED_K = 11
GAUSS_K = 15
IMG2_SCALE_PERCENT = 20

SAVE_DIR = "results"
SAVE_NAME = "Q5_result_image_5.png"
DPI = 150

def read_rgb(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def get_difference(imgA, imgB):
    diff = np.abs(imgA.astype(np.float32) - imgB.astype(np.float32))
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def resize_percent(img, percent: int):
    h, w = img.shape[:2]
    nw = int(w * percent / 100)
    nh = int(h * percent / 100)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def manual_average_filter(img, k):
    kernel = np.ones((k, k), np.float32) / (k * k)
    return cv2.filter2D(img, -1, kernel)

def manual_gaussian_filter(img, k, sigma=0):
    g1d = cv2.getGaussianKernel(k, sigma)
    kernel = g1d @ g1d.T
    return cv2.filter2D(img, -1, kernel)

def manual_median_filter_fast(img, k):
    if k % 2 == 0:
        raise ValueError("Median kernel size must be odd.")

    pad = k // 2
    
    if len(img.shape) == 2:
        channels = [img]
    else:
        channels = [img[:,:,i] for i in range(img.shape[2])]
    
    out_channels = []
    for channel in channels:
        padded = cv2.copyMakeBorder(channel, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        
        windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k), axis=(0, 1))
        
        filtered = np.median(windows, axis=(2, 3)).astype(channel.dtype)
        out_channels.append(filtered)
    
    if len(img.shape) == 2:
        return out_channels[0]
    else:
        return np.stack(out_channels, axis=2)

def show_grid(rows, cols, images, titles, cmap_list=None, facecolor="black"):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15), facecolor=facecolor)
    axes = np.array(axes).reshape(rows, cols)

    if cmap_list is None:
        cmap_list = [None] * (rows * cols)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            axes[r, c].imshow(images[idx], cmap=cmap_list[idx])
            axes[r, c].set_title(titles[idx], color="white")
            axes[r, c].axis("off")
            idx += 1

    plt.tight_layout()
    return fig

try:
    img1 = read_rgb(IMG1_PATH)
    img2 = read_rgb(IMG2_PATH)
    img3 = read_rgb(IMG3_PATH)
except FileNotFoundError as e:
    print(e)
    raise SystemExit(1)

out1_A = manual_average_filter(img1, AVG_K)
out1_B = cv2.blur(img1, (AVG_K, AVG_K))
diff1 = get_difference(out1_A, out1_B)

img2_small = resize_percent(img2, IMG2_SCALE_PERCENT)

print("Processing Manual Median Filter (fast version)...")
out2_A = manual_median_filter_fast(img2_small, MED_K)
out2_B = cv2.medianBlur(img2_small, MED_K)
diff2 = get_difference(out2_A, out2_B)

out3_A = manual_gaussian_filter(img3, GAUSS_K, sigma=0)
out3_B = cv2.GaussianBlur(img3, (GAUSS_K, GAUSS_K), 0)
diff3 = get_difference(out3_A, out3_B)

images = [
    out1_A, out1_B, diff1,
    out2_A, out2_B, diff2,
    out3_A, out3_B, diff3
]
titles = [
    "Q1: Manual Average (A)", "Q1: Built-in cv2.blur (B)", "Difference |A - B|",
    "Q2: Manual Median (A)", "Q2: Built-in cv2.medianBlur (B)", "Difference |A - B|",
    "Q3: Manual Gaussian (A)", "Q3: Built-in cv2.GaussianBlur (B)", "Difference |A - B|"
]
cmap_list = [None, None, "gray", None, None, "gray", None, None, "gray"]

fig = show_grid(3, 3, images, titles, cmap_list=cmap_list, facecolor="black")

os.makedirs(SAVE_DIR, exist_ok=True)
save_path = os.path.join(SAVE_DIR, SAVE_NAME)
plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())

plt.show()