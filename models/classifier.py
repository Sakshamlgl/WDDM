"""
Classifier head on top of the BiFormer feature vector.

Paper (Multi-scale feature extraction section): "the BiFormer network is
followed by a Layer Norm layer, a global pooling layer, and a fully
connected layer to obtain the final output." LayerNorm + GAP live in
models/backbone.py; this module is the fully connected layer.

Multi-label (5 lesion types, a study can have several), so the training loop
uses BCEWithLogitsLoss, not softmax cross-entropy.

head="linear" (default) is the paper's single FC layer. head="mlp" is an
opt-in variant (Linear -> ReLU -> Dropout -> Linear) kept for experiments.
"""
import torch
import torch.nn as nn

LESION_TYPES = ["Atelectasis", "Cardiomegaly", "Effusion", "Consolidation", "Edema"]
HEAD_TYPES = ("linear", "mlp")


class ClassifierHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        n_classes: int = len(LESION_TYPES),
        head: str = "linear",
        hidden_dim: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()
        if head == "linear":
            self.net = nn.Linear(in_dim, n_classes)
        elif head == "mlp":
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
        else:
            raise ValueError(f"Unknown head {head!r}; expected one of {HEAD_TYPES}")

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)  # logits, (N, n_classes)


if __name__ == "__main__":
    for h in HEAD_TYPES:
        logits = ClassifierHead(head=h)(torch.randn(4, 768))
        assert logits.shape == (4, len(LESION_TYPES)), logits.shape
        print(f"OK ({h}): logits shape {tuple(logits.shape)}")
