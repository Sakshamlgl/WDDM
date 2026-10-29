"""
Utilities for the Wasserstein-distance term of the loss (Eq. 1-3).

The critic itself lives in models/critic.py. This module implements the
paper's "select the source domain sample that is closest to the target
domain sample" step: for each target-batch sample, find its nearest
source-batch sample in feature space. train.py uses it when run with
`--w-source-subset closest` (the paper's literal description); the default
`all` uses the whole source batch, which is the standard unbiased estimate of
the dual form E[g(Xs)] - E[g(Xt)].
"""
import torch


def select_closest_source_samples(source_feat: torch.Tensor, target_feat: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    Returns indices into source_feat (shape (N_target * k,), may repeat)
    of the k nearest source samples for each target sample, in Euclidean
    feature-space distance. Computed with no_grad since this is a selection
    step, not part of the differentiable loss itself.
    """
    with torch.no_grad():
        dists = torch.cdist(target_feat, source_feat)          # (N_target, N_source)
        idx = dists.topk(k, dim=1, largest=False).indices      # (N_target, k)
    return idx.reshape(-1)


if __name__ == "__main__":
    src = torch.randn(10, 768)
    tgt = torch.randn(6, 768)
    idx = select_closest_source_samples(src, tgt, k=2)
    assert idx.shape == (12,), idx.shape
    assert idx.min() >= 0 and idx.max() < 10
    print(f"OK: selected source indices shape {tuple(idx.shape)}")
