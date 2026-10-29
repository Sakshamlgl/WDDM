"""
Generic multi-label chest X-ray dataset, driven by a CSV manifest.

Expects a harmonized manifest CSV per domain/split with columns:

    path,Atelectasis,Cardiomegaly,Effusion,Consolidation,Edema

(extra columns such as `patient_id` are allowed and ignored). `path` is
relative to `root`; the 5 label columns are 0/1. `build_wddm_manifests.py`
produces these for CheXpert and ChestX-ray14. Check the printed positive
rates against Fig. 7 of the paper (e.g. CheXpert: 24.09% Atelectasis, 62.10%
Effusion, ...) before training anything on them.
"""
import io
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Allow `python data/datasets.py ...` as well as `python -m data.datasets`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.cache import read_uncached
from models.classifier import LESION_TYPES


class ChestXrayDataset(Dataset):
    """
    Args:
        manifest_csv: harmonized manifest (see module docstring).
        root: directory the manifest's `path` column is relative to.
        transform: torchvision transform applied to the PIL image.
        return_index: if True, __getitem__ returns (img, label, idx). train.py
            uses the index to look up per-sample target pseudo-labels.
    """

    cache = None
    tensor_transform = None

    def __init__(self, manifest_csv: str, root: str, transform=None, return_index: bool = False):
        self.root = Path(root)
        self.df = pd.read_csv(manifest_csv)
        missing = [c for c in ["path", *LESION_TYPES] if c not in self.df.columns]
        if missing:
            raise ValueError(f"Manifest {manifest_csv} is missing columns: {missing}")
        labels = self.df[LESION_TYPES].to_numpy()
        if not np.isin(labels, (0, 1)).all():
            raise ValueError(f"Manifest {manifest_csv} has label values other than 0/1")

        # Plain Python/NumPy containers: much faster than df.iloc per item, and
        # cheap to share with forked DataLoader workers.
        self.paths = self.df["path"].astype(str).tolist()
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.transform = transform
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.paths)

    def use_cache(self, cache, tensor_transform) -> "ChestXrayDataset":
        """Serve images from a data.cache.ImageCache. `tensor_transform` must be
        `transform` minus its leading Resize + ToTensor (see data/transforms.py)."""
        self.cache, self.tensor_transform = cache, tensor_transform
        return self

    def _image(self, idx: int):
        x = self.cache.get(idx) if self.cache is not None else None
        if x is not None:
            return self.tensor_transform(x) if self.tensor_transform is not None else x
        # Same bytes as Image.open(path), but the file is dropped from the page
        # cache afterwards (on Kaggle, page cache counts against the RAM limit).
        with Image.open(io.BytesIO(read_uncached(self.root / self.paths[idx]))) as im:
            img = im.convert("RGB")
        return self.transform(img) if self.transform is not None else img

    def verify_cache(self, n: int = 32, seed: int = 0) -> int:
        """Compare `n` random items from the cache against the uncached disk
        pipeline (same RNG seed, so random augmentation matches too). Returns
        the number checked; raises if any tensor differs."""
        if self.cache is None or n <= 0:
            return 0
        idx = np.random.default_rng(seed).choice(len(self), min(n, len(self)), replace=False)
        cache, bad = self.cache, []
        try:
            for i in idx.tolist():
                torch.manual_seed(seed + i)
                a = self._image(i)
                self.cache = None
                torch.manual_seed(seed + i)
                b = self._image(i)
                self.cache = cache
                if a.shape != b.shape or not torch.equal(a, b):
                    bad.append((self.paths[i], float((a - b).abs().max()) if a.shape == b.shape else "shape"))
        finally:
            self.cache = cache
        if bad:
            raise RuntimeError(f"cached images differ from the disk pipeline for {len(bad)}/{len(idx)} samples, "
                               f"e.g. {bad[:3]}")
        return len(idx)

    def __getitem__(self, idx: int):
        img = self._image(idx)
        label = self.labels[idx]
        if self.return_index:
            return img, label, idx
        return img, label

    def label_prevalence(self) -> pd.Series:
        """Positive rate per lesion type -- compare this against Fig. 7 of the paper."""
        return self.df[LESION_TYPES].mean()

    def missing_files(self, limit: int = 10) -> list:
        """First `limit` manifest paths that don't exist under root (empty = all good)."""
        out = []
        for p in self.paths:
            if not (self.root / p).is_file():
                out.append(p)
                if len(out) >= limit:
                    break
        return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data/datasets.py <manifest.csv> <root_dir>")
        sys.exit(1)
    ds = ChestXrayDataset(sys.argv[1], sys.argv[2])
    print(f"{len(ds)} samples")
    print((ds.label_prevalence() * 100).round(2).to_string())
    bad = ds.missing_files()
    print("All image files found." if not bad else f"Missing files (first {len(bad)}): {bad}")
