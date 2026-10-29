"""
In-RAM decoded-image cache: selected manifest images are decoded once and
resized to 224x224 into a uint8 numpy array, so the expensive 1024x1024
ChestX-ray14 PNGs aren't decoded again every epoch. The array lives in the
training process's RAM (not in a file: on Kaggle, file page cache counts
against the 30 GB container limit, and a file in /tmp is charged about
twice). DataLoader workers forked afterwards share it copy-on-write.

Exactness: the pipeline's first steps are
    Image.open(p).convert("RGB") -> Resize((224, 224)) -> ToTensor
and chest X-rays are grayscale, so the three RGB channels are identical. The
cache applies the same torchvision Resize to that single channel, and
RamImageCache.get() rebuilds the identical 3-channel float tensor before the
remaining normalization/augmentation transforms. Images whose RGB channels are
NOT identical are recorded as fallbacks and always read from disk through the
full pipeline. ChestXrayDataset.verify_cache() checks a random sample against
the uncached pipeline bit for bit.

Reads go through read_uncached(): many files are read in parallel (Kaggle's
/kaggle/input is NFS, ~70 files/s serially, >1,400 files/s with 32 reads in
flight) and each file's page cache is dropped after reading.
"""
import hashlib
import io
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from data.transforms import IMG_SIZE

_CHUNK = 256
IO_THREADS = 12   # reads in flight per decoding process


def read_uncached(path) -> bytes:
    """Read a whole file, then tell the kernel to drop it from the page cache."""
    with open(path, "rb") as f:
        data = f.read()
        if hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
    return data


def cache_key(paths: Sequence[str], root, size: int = IMG_SIZE) -> str:
    h = hashlib.sha1(f"{size}|{Path(root).resolve()}|".encode())
    h.update("\n".join(paths).encode())
    return h.hexdigest()[:16]


def load_gray(path_or_file, resize) -> tuple:
    """(uint8 HxW array after `resize`, source mode), or (None, mode) if the
    image's RGB channels differ (then it must be read through the full RGB path)."""
    with Image.open(path_or_file) as img:
        mode = img.mode
        if mode == "L":
            gray = img.copy()
        else:
            rgb = np.asarray(img.convert("RGB"))
            if not ((rgb[..., 0] == rgb[..., 1]).all() and (rgb[..., 1] == rgb[..., 2]).all()):
                return None, mode
            gray = Image.fromarray(np.ascontiguousarray(rgb[..., 0]))
    return np.asarray(resize(gray), dtype=np.uint8), mode


_W: dict = {}


def _init_worker(root, paths, size):
    _W.update(root=Path(root), paths=paths, resize=transforms.Resize((size, size)))


def _work(chunk):
    start, stop = chunk
    root, paths, resize = _W["root"], _W["paths"], _W["resize"]
    out, modes = [], Counter()
    files = [root / paths[i] for i in range(start, stop)]
    with ThreadPoolExecutor(IO_THREADS) as pool:
        # map() submits every read at once and yields in order, so decoding
        # image i overlaps with reading the images after it.
        for i, (path, blob) in zip(range(start, stop), pool.map(lambda p: (p, read_uncached(p)), files)):
            try:
                arr, mode = load_gray(io.BytesIO(blob), resize)
            except Exception as e:  # keep the offending path in the error message
                raise RuntimeError(f"cache build failed on {path}: {e!r}") from None
            modes[mode] += 1
            out.append((i, arr))
    return out, dict(modes)


class RamImageCache:
    """Decoded images for one manifest (index-aligned with its rows)."""

    def __init__(self, data: np.ndarray, fallback: frozenset, name: str = ""):
        self.data, self.fallback, self.name = data, fallback, name

    def get(self, idx: int) -> Optional[torch.Tensor]:
        """(3, H, W) float tensor identical to ToTensor(Resize(img.convert('RGB'))), or None."""
        if idx in self.fallback:
            return None
        gray = torch.from_numpy(self.data[idx].copy())
        return gray.unsqueeze(0).expand(3, -1, -1).contiguous().float().div(255)

    @property
    def nbytes(self) -> int:
        return self.data.nbytes


def build_ram_cache(name: str, paths: List[str], root, workers: int = 4, size: int = IMG_SIZE,
                    log: Callable = print) -> RamImageCache:
    n = len(paths)
    data = np.zeros((n, size, size), dtype=np.uint8)
    fallback, modes, done, next_report, t0 = [], Counter(), 0, 0.2, time.time()
    chunks = [(s, min(s + _CHUNK, n)) for s in range(0, n, _CHUNK)]
    init_args = (str(root), list(paths), size)
    if workers <= 1:
        _init_worker(*init_args)
        results, pool = map(_work, chunks), None
    else:
        ctx = mp.get_context("spawn" if sys.platform == "win32" else "fork")
        pool = ctx.Pool(workers, initializer=_init_worker, initargs=init_args)
        results = pool.imap_unordered(_work, chunks)
    try:
        for out, md in results:
            for i, arr in out:
                if arr is None:
                    fallback.append(i)
                else:
                    data[i] = arr
            done += len(out)
            modes.update(md)
            if done / n >= next_report or done == n:
                el = time.time() - t0
                log(f"  cache {name}: {done:,}/{n:,} images ({el:.0f}s, {done / max(el, 1e-6):.0f} img/s)")
                next_report += 0.2
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        _W.clear()
    log(f"  cache {name}: {n:,} images in RAM ({data.nbytes / 1e9:.2f} GB) in {time.time() - t0:.0f}s | "
        f"modes {dict(modes)} | {len(fallback)} non-grayscale (read from disk)")
    return RamImageCache(data, frozenset(fallback), name)
