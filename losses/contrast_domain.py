"""
Contrast domain difference D_cdd (Eq. 4 of the WDDM paper).

    D_cdd = 1/N * sum_c D^{cc}(y^t, sigma)
          - 1/(N(N-1)) * sum_c sum_{c' != c} D^{cc'}(y^t, sigma)

D^{cc'} is the discrepancy between the kernel mean embeddings (in an RKHS)
of the source samples of category c and the target samples of category c',
i.e. an MMD, exactly as in CAN (Kang et al. 2019, the paper's ref. 46).
Minimizing D_cdd pulls each source class toward the same target class and
pushes it away from the other target classes.

Target categories y^t come from K-means clustering of target features,
seeded with source class centers, computed with the network fixed ("First,
the network parameters are fixed and the sample clusters in the target
domain are updated"). After clustering, "fuzzy target domain data far from
the cluster center and fuzzy classes containing few target samples ... are
set to zero".

Category definition (`mode`):
  - "multiclass" (default, paper): N = 5 categories = the 5 lesion types.
    The paper excludes images with none of the 5 lesions, so every image
    belongs to at least one category. Source membership is the (multi-hot)
    ground truth; target membership is the K=5 K-means cluster.
  - "binary" (opt-in variant): each lesion is its own N=2 problem
    (negative vs positive), clustered with its own 2-means, and the five
    binary D_cdd values are averaged. Better suited to multi-label data but
    not what the paper describes.

Implementation details the paper doesn't pin down (documented choices):
  - Spherical K-means (cosine similarity), as in CAN.
  - Fuzzy-sample filter: per cluster, keep the `keep_ratio` fraction of
    target samples closest to the center.
  - Fuzzy-class filter: a cluster with fewer than `min_class_size` kept
    samples is dropped entirely.
  - Multi-bandwidth Gaussian kernel, base bandwidth = detached mean pairwise
    squared distance in the batch (as in CAN / DAN).

Representation used throughout: a category is a column of a boolean mask
matrix. `category_groups(mode)` says which columns form one Eq. 4 problem.
"""
from typing import List, Tuple

import torch
import torch.nn.functional as F

CDD_MODES = ("multiclass", "binary")


# --------------------------------------------------------------------------- #
# Category bookkeeping
# --------------------------------------------------------------------------- #
def category_groups(mode: str, n_lesions: int) -> List[List[int]]:
    """Columns of the category-mask matrices that form one Eq. 4 problem."""
    if mode == "multiclass":
        return [list(range(n_lesions))]
    if mode == "binary":
        # column j = "lesion j negative", column n_lesions + j = "lesion j positive"
        return [[j, n_lesions + j] for j in range(n_lesions)]
    raise ValueError(f"Unknown CDD mode {mode!r}; expected one of {CDD_MODES}")


def source_category_masks(labels: torch.Tensor, mode: str) -> torch.Tensor:
    """(N, L) 0/1 source labels -> (N, C) bool category membership."""
    y = labels.bool()
    if mode == "multiclass":
        return y
    if mode == "binary":
        return torch.cat([~y, y], dim=1)
    raise ValueError(f"Unknown CDD mode {mode!r}; expected one of {CDD_MODES}")


# --------------------------------------------------------------------------- #
# Clustering (network fixed, whole target train set)
# --------------------------------------------------------------------------- #
def _spherical_kmeans(t: torch.Tensor, centers: torch.Tensor, n_iter: int):
    """t, centers: L2-normalized. Returns (assign, cosine sim to own center)."""
    for _ in range(n_iter):
        assign = (t @ centers.T).argmax(dim=1)
        new_centers = torch.stack([
            F.normalize(t[assign == k].mean(0), dim=0) if (assign == k).any() else centers[k]
            for k in range(centers.size(0))
        ])
        converged = torch.allclose(new_centers, centers, atol=1e-6)
        centers = new_centers
        if converged:
            break
    sim = t @ centers.T
    assign = sim.argmax(dim=1)
    return assign, sim.gather(1, assign.unsqueeze(1)).squeeze(1)


def _filter_fuzzy(assign: torch.Tensor, sim: torch.Tensor, n_clusters: int,
                  keep_ratio: float, min_class_size: int) -> torch.Tensor:
    """(Nt, n_clusters) bool membership after dropping fuzzy samples/classes."""
    dist = 1.0 - sim
    member = torch.zeros(assign.size(0), n_clusters, dtype=torch.bool, device=assign.device)
    for k in range(n_clusters):
        m = assign == k
        if not m.any():
            continue
        keep = m & (dist <= torch.quantile(dist[m], keep_ratio))
        if int(keep.sum()) >= min_class_size:
            member[:, k] = keep
    return member


@torch.no_grad()
def cluster_target(
    source_feat: torch.Tensor,     # (Ns, D)
    source_labels: torch.Tensor,   # (Ns, L) 0/1
    target_feat: torch.Tensor,     # (Nt, D)
    mode: str = "multiclass",
    n_iter: int = 20,
    keep_ratio: float = 0.8,
    min_class_size: int = 3,
) -> torch.Tensor:
    """
    Returns target category masks (Nt, C) bool, same column layout as
    `source_category_masks`. A False row/column entry means "not a (valid)
    member of that category" -- filtered samples are False everywhere in
    their group, dropped classes are False for every sample.
    """
    s = F.normalize(source_feat.float(), dim=1)
    t = F.normalize(target_feat.float(), dim=1)
    n_lesions = source_labels.size(1)
    src_masks = source_category_masks(source_labels, mode)
    out = torch.zeros(t.size(0), src_masks.size(1), dtype=torch.bool, device=t.device)

    for cols in category_groups(mode, n_lesions):
        # Seed one center per category that has source samples.
        seeded = [c for c in cols if src_masks[:, c].any()]
        if len(seeded) < 2:
            continue
        centers = F.normalize(torch.stack([s[src_masks[:, c]].mean(0) for c in seeded]), dim=1)
        assign, sim = _spherical_kmeans(t, centers, n_iter)
        member = _filter_fuzzy(assign, sim, len(seeded), keep_ratio, min_class_size)
        for k, c in enumerate(seeded):
            out[:, c] = member[:, k]
    return out


# --------------------------------------------------------------------------- #
# Differentiable D_cdd on a mini-batch
# --------------------------------------------------------------------------- #
def _multi_bandwidth_gaussian_kernel(feat: torch.Tensor, kernel_num: int = 5,
                                     kernel_mul: float = 2.0) -> torch.Tensor:
    """Mean of `kernel_num` Gaussian kernels with bandwidths spread around the
    mean pairwise squared distance. No sqrt, so no NaN gradient at d=0."""
    sq = (feat * feat).sum(dim=1)
    d2 = (sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * feat @ feat.T).clamp_min(0.0)
    n = feat.size(0)
    base = (d2.detach().sum() / max(n * n - n, 1)).clamp_min(1e-8)
    k = torch.zeros_like(d2)
    for i in range(kernel_num):
        k = k + torch.exp(-d2 / (base * kernel_mul ** (i - kernel_num // 2)))
    return k / kernel_num


def contrast_domain_discrepancy(
    source_feat: torch.Tensor,     # (Ns, D), requires grad
    source_masks: torch.Tensor,    # (Ns, C) bool, from source_category_masks
    target_feat: torch.Tensor,     # (Nt, D), requires grad
    target_masks: torch.Tensor,    # (Nt, C) bool, rows of cluster_target output
    groups: List[List[int]],
    min_samples: int = 2,
    kernel_num: int = 5,
    kernel_mul: float = 2.0,
) -> Tuple[torch.Tensor, int]:
    """
    Eq. 4 on one batch. A category c takes part only if both the source and
    target side have >= `min_samples` members in the batch. Per group:
        D = mean_c MMD^2(S_c, T_c) - mean_{c != c'} MMD^2(S_c, T_c')
    (intra-class term only when a single category is present). Returns the
    mean over groups with at least one active category, and how many groups
    that was.
    """
    n_s, n_t = source_feat.size(0), target_feat.size(0)
    n_cat = source_masks.size(1)
    feat = torch.cat([source_feat, target_feat], dim=0).float()
    kern = _multi_bandwidth_gaussian_kernel(feat, kernel_num, kernel_mul)

    # Block membership over the concatenated [source; target] index:
    # columns 0..C-1 = source categories, C..2C-1 = target categories.
    m = torch.zeros(n_s + n_t, 2 * n_cat, dtype=kern.dtype, device=kern.device)
    m[:n_s, :n_cat] = source_masks.to(kern.dtype)
    m[n_s:, n_cat:] = target_masks.to(kern.dtype)
    cnt = m.sum(dim=0)
    e = (m.T @ kern @ m) / (cnt.unsqueeze(1) * cnt.unsqueeze(0)).clamp_min(1.0)
    counts = cnt.tolist()

    def mmd2(c_src: int, c_tgt: int) -> torch.Tensor:
        a, b = c_src, n_cat + c_tgt
        return e[a, a] + e[b, b] - 2.0 * e[a, b]

    terms = []
    for cols in groups:
        active = [c for c in cols if counts[c] >= min_samples and counts[n_cat + c] >= min_samples]
        if not active:
            continue
        intra = torch.stack([mmd2(c, c) for c in active]).mean()
        if len(active) >= 2:
            inter = torch.stack([mmd2(c, c2) for c in active for c2 in active if c2 != c]).mean()
            terms.append(intra - inter)
        else:
            terms.append(intra)

    if not terms:
        return feat.new_zeros(()), 0
    return torch.stack(terms).mean(), len(terms)


if __name__ == "__main__":
    torch.manual_seed(0)
    n_lesions, dim = 5, 768

    # Synthetic domains: each lesion adds its own direction to the feature.
    shift = torch.randn(n_lesions, dim) * 2.0
    def make(n, offset):
        lab = (torch.rand(n, n_lesions) > 0.6).float()
        lab[lab.sum(1) == 0, torch.randint(0, n_lesions, (1,))] = 1.0  # >=1 lesion, as in the paper
        return torch.randn(n, dim) * 0.3 + lab @ shift + offset, lab
    src, src_lab = make(300, 0.0)
    tgt, tgt_lab = make(200, 0.1)

    for mode in CDD_MODES:
        groups = category_groups(mode, n_lesions)
        tm = cluster_target(src, src_lab, tgt, mode=mode)
        truth = source_category_masks(tgt_lab, mode)
        prec = (truth[tm].float().mean().item()) if tm.any() else float("nan")
        print(f"[{mode}] clustering: {tm.any(1).float().mean().item():.2f} of target kept, "
              f"membership precision {prec:.3f}")

        fs = torch.randn(16, dim, requires_grad=True)
        ft = torch.randn(16, dim, requires_grad=True)
        sm = source_category_masks(src_lab[:16], mode)
        d, n_active = contrast_domain_discrepancy(fs, sm, ft, truth[:16], groups)
        assert d.dim() == 0 and torch.isfinite(d)
        d.backward()
        assert fs.grad is not None and ft.grad is not None
        assert torch.isfinite(fs.grad).all() and torch.isfinite(ft.grad).all()
        print(f"[{mode}] OK: D_cdd = {d.item():.4f} over {n_active} group(s); gradients reach both domains")
