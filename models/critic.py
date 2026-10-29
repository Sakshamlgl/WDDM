"""
Critic g(x) used to estimate the Wasserstein distance via its
Kantorovich-Rubenstein dual form (Eq. 2 in the paper):

    W(Xs, Xt) = sup_{||g||_L <= 1} E[g(Ds)] - E[g(Dt)]

g is trained to (locally) maximize this gap subject to a 1-Lipschitz
constraint, enforced here with a gradient penalty (WGAN-GP), which is
smoother to optimize than hard weight clipping.
"""
import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, in_dim: int = 768, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)  # (N,)


def gradient_penalty(critic: Critic, source_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
    """Standard WGAN-GP penalty, interpolating between source and target features."""
    n = min(source_feat.size(0), target_feat.size(0))
    src, tgt = source_feat[:n], target_feat[:n]
    eps = torch.rand(n, 1, device=src.device)
    interp = (eps * src + (1 - eps) * tgt).detach().requires_grad_(True)
    scores = critic(interp)
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=interp,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_norm = grads.norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()


if __name__ == "__main__":
    critic = Critic()
    src = torch.randn(8, 768, requires_grad=False)
    tgt = torch.randn(6, 768, requires_grad=False)
    scores = critic(src)
    assert scores.shape == (8,), scores.shape
    gp = gradient_penalty(critic, src, tgt)
    assert gp.dim() == 0
    gp.backward()  # confirms the penalty is differentiable end-to-end
    print(f"OK: critic scores shape {tuple(scores.shape)}, gradient_penalty={gp.item():.4f}")
