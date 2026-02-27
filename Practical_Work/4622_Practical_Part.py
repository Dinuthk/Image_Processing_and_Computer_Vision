import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random


# --------------------------------------------------
# Import image paths
# --------------------------------------------------

FUNDUS_TRAIN = r"../images/Database_For_Practical_Part/For Designing/Fundus images for Designing"
GT_TRAIN     = r"../images/Database_For_Practical_Part/For Designing/Ground Truths For Designing"

VALIDATION_FUNDUS = r"../images/Database_For_Practical_Part/For Validation/Foundus Images For Validation"
VALIDATION_GT     = r"../images/Database_For_Practical_Part/For Validation/Ground Truth For Validation"

exts = ('.png', '.jpg', '.jpeg')

val_paths = sorted([
    os.path.join(VALIDATION_FUNDUS, f)
    for f in os.listdir(VALIDATION_FUNDUS)
    if f.endswith(exts)
])

val_gt = sorted([
    os.path.join(VALIDATION_GT, f)
    for f in os.listdir(VALIDATION_GT)
    if f.endswith(exts)
])

print(f"Validation images: {len(val_paths)}  |  Validation ground truths: {len(val_gt)}")


# --------------------------------------------------
# Image processing pipeline
# --------------------------------------------------

def noise_reduction(image_path):
    """Extract green channel and apply Gaussian blur."""
    img = cv2.imread(image_path)
    green = img[:, :, 1]
    return cv2.GaussianBlur(green, (5, 5), 0)


def contrast_enhance(image):
    """CLAHE followed by gamma correction."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(image)

    gamma = 1.0
    table = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(enhanced, table)


def adaptive_threshold(image):
    """Blur + adaptive threshold + morphological cleaning."""
    img = image.astype(np.uint8)

    img = cv2.medianBlur(img, 5)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25,
        C=2
    )

    kernel = np.ones((3, 3), np.uint8)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return thresh


def morphological_cleanup(binary):
    """Morphological operations and component filtering."""

    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_med   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_small)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_med)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    out = np.zeros_like(closed)

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 150:
            out[labels == i] = 255

    return out


def final_cleanup(vessel_mask):
    """Apply FOV mask and final component filtering."""

    h, w = vessel_mask.shape[:2]

    center = (w // 2, h // 2)
    radius = int(min(center) * 0.90)

    fov = np.zeros((h, w), np.uint8)
    cv2.circle(fov, center, radius, 255, -1)

    masked = cv2.bitwise_and(vessel_mask, fov)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    opened = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)

    cleaned = np.zeros_like(opened)

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 310:
            cleaned[labels == i] = 255

    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)


def connect_vessels(mask):
    """Bridge nearby vessel segments."""
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, bridge_kernel)


def segment(image_path):
    """Full segmentation pipeline."""
    img = noise_reduction(image_path)
    img = contrast_enhance(img)
    img = adaptive_threshold(img)
    img = morphological_cleanup(img)
    img = final_cleanup(img)
    img = connect_vessels(img)
    return img


# --------------------------------------------------
# Evaluation metrics
# --------------------------------------------------

def dice(pred, gt):
    p, g = pred > 0, gt > 0
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else (2.0 * inter) / denom


def jaccard(pred, gt):
    p, g = pred > 0, gt > 0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return 1.0 if union == 0 else inter / union


# --------------------------------------------------
# Run pipeline for entire dataset
# --------------------------------------------------

predictions = [segment(p) for p in val_paths]

val_names = [os.path.basename(p) for p in val_paths]
gt_names = [f for f in os.listdir(VALIDATION_GT) if f.endswith(exts)]

common = sorted(set(val_names) & set(gt_names))
name_to_idx = {n: i for i, n in enumerate(val_names)}

print(f"Matched pairs for evaluation: {len(common)}")


dice_scores = []
jaccard_scores = []
eval_names = []

for name in common:

    idx = name_to_idx[name]
    pred = predictions[idx]

    gt_img = cv2.imread(os.path.join(VALIDATION_GT, name), cv2.IMREAD_GRAYSCALE)

    if gt_img is None:
        continue

    _, gt_bin = cv2.threshold(gt_img, 128, 255, cv2.THRESH_BINARY)

    if pred.shape != gt_bin.shape:
        pred = cv2.resize(pred, (gt_bin.shape[1], gt_bin.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    dice_scores.append(dice(pred, gt_bin))
    jaccard_scores.append(jaccard(pred, gt_bin))
    eval_names.append(name)


# --------------------------------------------------
# Random subset analysis
# --------------------------------------------------

subset_size = min(50, len(dice_scores))
subset_indices = random.sample(range(len(dice_scores)), subset_size)

subset_dice = [dice_scores[i] for i in subset_indices]
subset_jaccard = [jaccard_scores[i] for i in subset_indices]


print(f"\n{'='*50}")
print(f"  Evaluation Results")
print(f"{'='*50}")
print(f"  Dice  — Mean: {np.mean(subset_dice):.4f}  Std: {np.std(subset_dice):.4f}"
      f"  Min: {np.min(subset_dice):.4f}  Max: {np.max(subset_dice):.4f}")

print(f"  IoU   — Mean: {np.mean(subset_jaccard):.4f}  Std: {np.std(subset_jaccard):.4f}"
      f"  Min: {np.min(subset_jaccard):.4f}  Max: {np.max(subset_jaccard):.4f}")
print(f"{'='*50}")


# --------------------------------------------------
# Visualisation
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].bar(range(subset_size), subset_dice)
axes[0].axhline(np.mean(subset_dice), ls='--')
axes[0].set(title="Dice per Image", xlabel="Image Index", ylabel="DSC")

axes[1].bar(range(subset_size), subset_jaccard)
axes[1].axhline(np.mean(subset_jaccard), ls='--')
axes[1].set(title="Jaccard per Image", xlabel="Image Index", ylabel="IoU")

plt.tight_layout()
plt.show()


# --------------------------------------------------
# Save 10 random visual results
# --------------------------------------------------

sample_indices = random.sample(subset_indices, min(10, len(subset_indices)))

output_dir = os.path.join("practical work", "results")
os.makedirs(output_dir, exist_ok=True)

for i, idx in enumerate(sample_indices):

    name = eval_names[idx]

    pred = predictions[name_to_idx[name]]

    gt_img = cv2.imread(os.path.join(VALIDATION_GT, name), cv2.IMREAD_GRAYSCALE)
    _, gt_bin = cv2.threshold(gt_img, 128, 255, cv2.THRESH_BINARY)

    orig = cv2.cvtColor(
        cv2.imread(os.path.join(VALIDATION_FUNDUS, name)),
        cv2.COLOR_BGR2RGB
    )

    if pred.shape != gt_bin.shape:
        pred = cv2.resize(pred,
                          (gt_bin.shape[1], gt_bin.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(orig)
    axes[0].set_title(f"Original ({name})")
    axes[0].axis('off')

    axes[1].imshow(pred, cmap='gray')
    axes[1].set_title("Predicted")
    axes[1].axis('off')

    axes[2].imshow(gt_bin, cmap='gray')
    axes[2].set_title("Ground Truth")
    axes[2].axis('off')

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"result_{i+1:02d}_{name}")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {save_path}")


print(f"\nAll result images saved to '{output_dir}/'")