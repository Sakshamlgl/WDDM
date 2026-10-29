"""
Generates a tiny fake source/target 'dataset' (noise images, random multi-hot
labels with >= 1 lesion per image, like the paper's filtered datasets) so the
full pipeline -- data loading -> backbone -> clustering -> class-aware
sampling -> losses -> training loop -> checkpoint/resume -> eval -- can be
smoke-tested without real chest X-rays. It does NOT validate model quality.

Usage:
    python tests/make_synthetic_data.py
    python train.py --source-manifest tests/synthetic/source_manifest.csv \\
        --source-root tests/synthetic/source_imgs \\
        --target-manifest tests/synthetic/target_manifest.csv \\
        --target-root tests/synthetic/target_imgs \\
        --val-manifest tests/synthetic/val_manifest.csv --val-root tests/synthetic/val_imgs \\
        --epochs 2 --batch-size 4 --num-workers 0 --no-pretrained \\
        --n-critic 1 --cluster-source-samples 0 --log-every 1 --out-dir runs/smoke
"""
import csv
import os

import numpy as np
from PIL import Image

OUT_DIR = os.path.join(os.path.dirname(__file__), "synthetic")
LESIONS = ["Atelectasis", "Cardiomegaly", "Effusion", "Consolidation", "Edema"]


def make_fake_domain(name: str, n: int, seed: int, brightness: int):
    img_dir = os.path.join(OUT_DIR, f"{name}_imgs")
    os.makedirs(img_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        labels = rng.integers(0, 2, len(LESIONS))
        if labels.sum() == 0:
            labels[rng.integers(0, len(LESIONS))] = 1
        # Grayscale noise with a per-domain brightness shift and a faint
        # label-dependent pattern, saved as single-channel PNG like real CXRs.
        arr = rng.normal(brightness, 40, (256, 256))
        for j, on in enumerate(labels):
            if on:
                arr[40 * j:40 * j + 30, :] += 40
        fname = f"img{i}.png"
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(os.path.join(img_dir, fname))
        rows.append([fname, *labels.tolist()])
    with open(os.path.join(OUT_DIR, f"{name}_manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", *LESIONS])
        w.writerows(rows)


if __name__ == "__main__":
    make_fake_domain("source", n=24, seed=1, brightness=110)
    make_fake_domain("target", n=24, seed=2, brightness=140)
    make_fake_domain("val", n=12, seed=3, brightness=140)
    print(f"Synthetic data written to {OUT_DIR}")
