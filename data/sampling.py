"""
Per-epoch batch plans for the source and target loaders.

Paper (Network optimization section): "class-aware sampling is used for the
source and target domains, i.e., data is sampled from both domains for each
class within a randomly sampled subset of classes." This is CAN's
class-aware sampling: each iteration picks a random subset of categories and
draws the same number of samples per picked category from both domains
(source by ground truth, target by cluster pseudo-label).

Plans are plain lists of index lists, built in the main process and handed
to two DataLoaders as `batch_sampler`s. DataLoader preserves batch order, so
source batch i and target batch i always come from the same category subset,
regardless of worker count.

Sampling modes:
  - "class-aware" (default, paper): needs target category masks, i.e. only
    once clustering has run. Falls back to random batches otherwise.
  - "random" (opt-in): shuffled source batches, target drawn at random.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np

SAMPLING_MODES = ("class-aware", "random")


def random_plan(n_source: int, n_target: int, batch_size: int, n_batches: int,
                rng: np.random.Generator) -> Tuple[List[List[int]], List[List[int]]]:
    def stream(n: int) -> np.ndarray:
        reps = int(np.ceil(n_batches * batch_size / n))
        return np.concatenate([rng.permutation(n) for _ in range(reps)])[: n_batches * batch_size]

    src = stream(n_source).reshape(n_batches, batch_size)
    tgt = stream(n_target).reshape(n_batches, batch_size)
    return src.tolist(), tgt.tolist()


def _draw(pool: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(pool, size=k, replace=len(pool) < k)


def class_aware_plan(
    source_masks: np.ndarray,          # (Ns, C) bool, ground-truth categories
    target_masks: np.ndarray,          # (Nt, C) bool, cluster categories (after filtering)
    units: Sequence[Sequence[int]],    # category columns sampled together
    batch_size: int,
    n_batches: int,
    units_per_batch: int,
    rng: np.random.Generator,
    min_target_per_category: int = 1,
    min_per_category: int = 2,
) -> Optional[Tuple[List[List[int]], List[List[int]]]]:
    """
    `units`: for the paper's multiclass setting every category is its own unit
    ([[0], [1], ..., [4]]); for the binary variant a unit is a lesion's
    (negative, positive) pair so both polarities are always sampled together.

    Each batch picks up to `units_per_batch` eligible units at random and
    draws batch_size // (#picked categories) samples per category from each
    domain. The number of picked units is capped so every category gets at
    least `min_per_category` samples (otherwise the per-batch D_cdd would
    have nothing to compare). Returns None if no unit is eligible (caller
    falls back to random batches).
    """
    src_pools = [np.flatnonzero(source_masks[:, c]) for c in range(source_masks.shape[1])]
    tgt_pools = [np.flatnonzero(target_masks[:, c]) for c in range(target_masks.shape[1])]
    eligible = [
        u for u in units
        if all(len(src_pools[c]) > 0 and len(tgt_pools[c]) >= min_target_per_category for c in u)
    ]
    if not eligible:
        return None

    unit_size = max(len(u) for u in eligible)
    m_max = max(1, batch_size // (unit_size * max(1, min_per_category)))
    m = min(units_per_batch, len(eligible), m_max)
    src_batches, tgt_batches = [], []
    for _ in range(n_batches):
        picked = rng.choice(len(eligible), size=m, replace=False)
        cats = [c for i in picked for c in eligible[i]]
        k = max(1, batch_size // len(cats))
        src_batches.append(np.concatenate([_draw(src_pools[c], k, rng) for c in cats]).tolist())
        tgt_batches.append(np.concatenate([_draw(tgt_pools[c], k, rng) for c in cats]).tolist())
    return src_batches, tgt_batches


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sm = rng.random((100, 5)) > 0.6
    tm = np.zeros((80, 5), dtype=bool)
    tm[np.arange(80), rng.integers(0, 5, 80)] = True
    plan = class_aware_plan(sm, tm, [[c] for c in range(5)], batch_size=16, n_batches=3,
                            units_per_batch=4, rng=rng)
    assert plan is not None
    for s, t in zip(*plan):
        assert len(s) == len(t) == 16
    s, t = random_plan(100, 80, 16, 3, rng)
    assert len(s) == len(t) == 3 and all(len(b) == 16 for b in s + t)
    print("OK: class-aware and random plans have matching, full-size batches")
