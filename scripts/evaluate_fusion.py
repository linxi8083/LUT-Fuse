import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.metrics import structural_similarity


VISIBLE_DIR = Path("dataset/test/Visible")
INFRARED_DIR = Path("dataset/test/Infrared")
FUSED_DIR = Path("results/author_pretrained/LLVIP")
OUTPUT_FILE = FUSED_DIR / "metrics.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PAPER_RESULTS = {
    "MI": 2.446,
    "EN": 7.545,
    "CC": 0.719,
    "SSIM": 0.892,
    "QABF": 0.597,
}


def image_files(directory):
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def entropy(image):
    histogram = np.histogram(image, bins=256, range=(0, 256))[0].astype(np.float64)
    probability = histogram / histogram.sum()
    probability = probability[probability > 0]
    return float(-np.sum(probability * np.log2(probability)))


def mutual_information(first, second):
    joint = np.histogram2d(
        first.ravel(), second.ravel(), bins=256, range=((0, 256), (0, 256))
    )[0]
    joint /= joint.sum()
    first_probability = joint.sum(axis=1, keepdims=True)
    second_probability = joint.sum(axis=0, keepdims=True)
    independent = first_probability @ second_probability
    valid = joint > 0
    return float(np.sum(joint[valid] * np.log2(joint[valid] / independent[valid])))


def correlation(first, second):
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = math.sqrt(
        float(np.sum(first_centered ** 2) * np.sum(second_centered ** 2))
    )
    if denominator == 0:
        return 0.0
    return float(np.sum(first_centered * second_centered) / denominator)


def gradient(image):
    horizontal = ndimage.sobel(image, axis=1, mode="reflect")
    vertical = ndimage.sobel(image, axis=0, mode="reflect")
    magnitude = np.hypot(horizontal, vertical)
    orientation = np.arctan2(vertical, horizontal)
    return magnitude, orientation


def edge_quality(source_magnitude, source_orientation, fused_magnitude, fused_orientation):
    maximum = np.maximum(source_magnitude, fused_magnitude)
    minimum = np.minimum(source_magnitude, fused_magnitude)
    relative_strength = np.divide(
        minimum, maximum, out=np.zeros_like(maximum), where=maximum > 0
    )

    angle_difference = np.abs(source_orientation - fused_orientation)
    angle_difference = np.minimum(angle_difference, 2 * np.pi - angle_difference)
    relative_orientation = 1 - np.minimum(angle_difference, np.pi / 2) / (np.pi / 2)

    strength_quality = 0.9994 / (1 + np.exp(-15 * (relative_strength - 0.5)))
    orientation_quality = 0.9879 / (1 + np.exp(-22 * (relative_orientation - 0.8)))
    return strength_quality * orientation_quality


def qabf(visible, infrared, fused):
    visible_magnitude, visible_orientation = gradient(visible)
    infrared_magnitude, infrared_orientation = gradient(infrared)
    fused_magnitude, fused_orientation = gradient(fused)

    visible_quality = edge_quality(
        visible_magnitude, visible_orientation, fused_magnitude, fused_orientation
    )
    infrared_quality = edge_quality(
        infrared_magnitude, infrared_orientation, fused_magnitude, fused_orientation
    )
    denominator = np.sum(visible_magnitude + infrared_magnitude)
    if denominator == 0:
        return 0.0
    numerator = np.sum(
        visible_quality * visible_magnitude + infrared_quality * infrared_magnitude
    )
    return float(numerator / denominator)


def evaluate_pair(visible, infrared, fused):
    if visible.shape != infrared.shape or visible.shape != fused.shape:
        raise ValueError(
            f"Image shapes do not match: visible={visible.shape}, "
            f"infrared={infrared.shape}, fused={fused.shape}"
        )
    return {
        "MI": mutual_information(visible, fused) + mutual_information(infrared, fused),
        "EN": entropy(fused),
        "CC": (correlation(visible, fused) + correlation(infrared, fused)) / 2,
        "SSIM": (
            structural_similarity(visible, fused, data_range=255)
            + structural_similarity(infrared, fused, data_range=255)
        ) / 2,
        "QABF": qabf(visible, infrared, fused),
    }


def main():
    visible_files = image_files(VISIBLE_DIR)
    infrared_files = image_files(INFRARED_DIR)
    fused_files = image_files(FUSED_DIR)
    common_names = sorted(set(visible_files) & set(infrared_files) & set(fused_files))

    if not common_names:
        raise ValueError("No matching visible, infrared, and fused images were found")
    if len(common_names) != len(visible_files):
        raise ValueError(
            f"Only {len(common_names)} complete triplets found for "
            f"{len(visible_files)} visible test images"
        )

    per_image = {}
    for index, name in enumerate(common_names, start=1):
        values = evaluate_pair(
            load_gray(visible_files[name]),
            load_gray(infrared_files[name]),
            load_gray(fused_files[name]),
        )
        per_image[name] = values
        print(f"[{index:03d}/{len(common_names):03d}] {name}")

    averages = {
        metric: float(np.mean([values[metric] for values in per_image.values()]))
        for metric in PAPER_RESULTS
    }
    differences = {
        metric: averages[metric] - PAPER_RESULTS[metric]
        for metric in PAPER_RESULTS
    }
    report = {
        "image_count": len(common_names),
        "average": averages,
        "paper_llvip": PAPER_RESULTS,
        "difference": differences,
        "per_image": per_image,
    }
    OUTPUT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nMetric       Current       Paper       Difference")
    for metric in PAPER_RESULTS:
        print(
            f"{metric:<8} {averages[metric]:>12.6f} "
            f"{PAPER_RESULTS[metric]:>11.6f} {differences[metric]:>16.6f}"
        )
    print(f"\nSaved report to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
