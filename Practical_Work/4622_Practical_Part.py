import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random

FUNDUS_TRAIN = r"../images/Database_For_Practical_Part/For_Designing/Fundus_images_for_Designing"
GT_TRAIN = r"../images/Database_For_Practical_Part/For_Designing/Ground_Truths_For_Designing"
VALIDATION_FUNDUS = r"../images/Database_For_Practical_Part/For_Validation/Foundus_Images_For_Validation"
VALIDATION_GT = r"../images/Database_For_Practical_Part/For_Validation/Ground_Truth_For_Validation"
EXTS = (".png", ".jpg", ".jpeg")


def list_images(folder, exts=EXTS):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    )


def noise_reduction(image_path):
    img = cv2.imread(image_path)
    green = img[:, :, 1]
    return cv2.GaussianBlur(green, (5, 5), 0)


def contrast_enhance(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(image)
    gamma = 1.0
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype(
        "uint8"
    )
    return cv2.LUT(enhanced, table)


def adaptive_threshold(image):
    img = image.astype(np.uint8)
    img = cv2.medianBlur(img, 5)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25,
        C=2,
    )
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh


def morphological_cleanup(binary):
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_small), cv2.MORPH_CLOSE, k_med)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    out = np.zeros_like(closed)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 150:
            out[labels == i] = 255
    return out


def final_cleanup(vessel_mask):
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
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, bridge_kernel)


def segment(image_path):
    x = noise_reduction(image_path)
    x = contrast_enhance(x)
    x = adaptive_threshold(x)
    x = morphological_cleanup(x)
    x = final_cleanup(x)
    x = connect_vessels(x)
    return x


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


def read_gt_bin(gt_path):
    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt_img is None:
        return None
    _, gt_bin = cv2.threshold(gt_img, 128, 255, cv2.THRESH_BINARY)
    return gt_bin


def match_pairs(val_paths, gt_folder):
    val_names = [os.path.basename(p) for p in val_paths]
    gt_names = [f for f in os.listdir(gt_folder) if f.lower().endswith(EXTS)]
    common = sorted(set(val_names) & set(gt_names))
    name_to_idx = {n: i for i, n in enumerate(val_names)}
    return common, name_to_idx


def evaluate(common_names, name_to_idx, val_paths, gt_folder, predictions):
    dice_scores, jaccard_scores, eval_names = [], [], []
    for name in common_names:
        idx = name_to_idx[name]
        pred = predictions[idx]
        gt_bin = read_gt_bin(os.path.join(gt_folder, name))
        if gt_bin is None:
            continue
        if pred.shape != gt_bin.shape:
            pred = cv2.resize(pred, (gt_bin.shape[1], gt_bin.shape[0]), interpolation=cv2.INTER_NEAREST)
        dice_scores.append(dice(pred, gt_bin))
        jaccard_scores.append(jaccard(pred, gt_bin))
        eval_names.append(name)
    return dice_scores, jaccard_scores, eval_names


def plot_subset(subset_dice, subset_jaccard):
    subset_size = len(subset_dice)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].bar(range(subset_size), subset_dice)
    axes[0].axhline(np.mean(subset_dice), ls="--")
    axes[0].set(title="Dice per Image", xlabel="Image Index", ylabel="DSC")

    axes[1].bar(range(subset_size), subset_jaccard)
    axes[1].axhline(np.mean(subset_jaccard), ls="--")
    axes[1].set(title="Jaccard per Image", xlabel="Image Index", ylabel="IoU")

    plt.tight_layout()
    plt.show()


def save_visual_results(sample_indices, subset_indices, eval_names, predictions, name_to_idx, fundus_folder, gt_folder, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for i, idx in enumerate(sample_indices):
        name = eval_names[idx]
        pred = predictions[name_to_idx[name]]

        gt_bin = read_gt_bin(os.path.join(gt_folder, name))
        if gt_bin is None:
            continue

        orig_bgr = cv2.imread(os.path.join(fundus_folder, name))
        if orig_bgr is None:
            continue
        orig = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

        if pred.shape != gt_bin.shape:
            pred = cv2.resize(pred, (gt_bin.shape[1], gt_bin.shape[0]), interpolation=cv2.INTER_NEAREST)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(orig)
        axes[0].set_title(f"Original ({name})")
        axes[0].axis("off")

        axes[1].imshow(pred, cmap="gray")
        axes[1].set_title("Predicted")
        axes[1].axis("off")

        axes[2].imshow(gt_bin, cmap="gray")
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"result_{i+1:02d}_{name}")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")


def main():
    val_paths = list_images(VALIDATION_FUNDUS)
    val_gt_paths = list_images(VALIDATION_GT)

    print(f"Validation images: {len(val_paths)}  |  Validation ground truths: {len(val_gt_paths)}")

    predictions = [segment(p) for p in val_paths]

    common, name_to_idx = match_pairs(val_paths, VALIDATION_GT)
    print(f"Matched pairs for evaluation: {len(common)}")

    dice_scores, jaccard_scores, eval_names = evaluate(common, name_to_idx, val_paths, VALIDATION_GT, predictions)

    if len(dice_scores) == 0:
        print("No valid image pairs found for evaluation.")
        return

    subset_size = min(50, len(dice_scores))
    subset_indices = random.sample(range(len(dice_scores)), subset_size)

    subset_dice = [dice_scores[i] for i in subset_indices]
    subset_jaccard = [jaccard_scores[i] for i in subset_indices]

    print(f"\n{'='*50}")
    print("  Evaluation Results")
    print(f"{'='*50}")
    print(
        f"  Dice  — Mean: {np.mean(subset_dice):.4f}  Std: {np.std(subset_dice):.4f}"
        f"  Min: {np.min(subset_dice):.4f}  Max: {np.max(subset_dice):.4f}"
    )
    print(
        f"  IoU   — Mean: {np.mean(subset_jaccard):.4f}  Std: {np.std(subset_jaccard):.4f}"
        f"  Min: {np.min(subset_jaccard):.4f}  Max: {np.max(subset_jaccard):.4f}"
    )
    print(f"{'='*50}")

    plot_subset(subset_dice, subset_jaccard)

    sample_indices = random.sample(subset_indices, min(10, len(subset_indices)))
    output_dir = os.path.join(".", "results")
    save_visual_results(sample_indices, subset_indices, eval_names, predictions, name_to_idx, VALIDATION_FUNDUS, VALIDATION_GT, output_dir)

    print(f"\nAll result images saved to '{output_dir}/'")


if __name__ == "__main__":
    main()