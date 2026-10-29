"""
Wraps the vendored BiFormer-B as a feature extractor matching Fig. 2 of the
WDDM paper: four BiFormer stages -> LayerNorm -> global average pool ->
768-d feature vector.

BiFormer's own `norm` after stage 4 is a BatchNorm2d (as are the stem and
every downsampling layer). It is kept so ImageNet pre-trained weights load
correctly. Because of all that BatchNorm, callers should push source and
target images through the extractor in ONE forward pass (see train.py), so
training-time batch statistics match what the running statistics see at
eval time. A LayerNorm (`self.ln`) is applied before global average pooling,
matching the paper's description.
"""
import sys
from pathlib import Path
from typing import Optional

_BIFORMER_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "biformer"
if str(_BIFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(_BIFORMER_ROOT))

import torch
import torch.nn as nn

from biformer_models.biformer import (  # noqa: E402
    biformer_base,
    load_imagenet_state_dict,
    safe_load_checkpoint,
)


class BiFormerFeatureExtractor(nn.Module):
    """Outputs a (N, 768) feature vector for a (N, 3, 224, 224) chest X-ray batch."""

    FEATURE_DIM = 768

    def __init__(self, pretrained: bool = True, weights_path: Optional[str] = None):
        """
        Args:
            pretrained: Load ImageNet-1k pre-trained BiFormer-B weights. Use True
                for training; use False when the weights are about to be
                overwritten anyway (eval.py, resuming) or for offline tests.
            weights_path: Optional local BiFormer-B ImageNet checkpoint (e.g. the
                official `biformer_base_best.pth`). Used instead of downloading.
        """
        super().__init__()
        try:
            # pretrained=True uses torch.hub's cache (~/.cache/torch/hub/checkpoints/
            # biformer_base_in1k.pth) and only downloads if that file is absent.
            self.backbone = biformer_base(pretrained=pretrained and not weights_path, num_classes=0)
        except Exception as e:  # urllib HTTPError / URLError, etc.
            raise RuntimeError(
                "Could not download the ImageNet BiFormer-B weights. The upstream OneDrive link "
                "no longer allows scripted downloads. Download 'BiFormer-B' from "
                "https://github.com/rayleizhu/BiFormer (Results and Pre-trained Models -> model) "
                "in a browser, then either pass --pretrained-weights /path/to/file.pth or copy it "
                f"to {Path(torch.hub.get_dir()) / 'checkpoints' / 'biformer_base_in1k.pth'}. "
                f"Original error: {e}"
            ) from e
        if weights_path:
            load_imagenet_state_dict(self.backbone, safe_load_checkpoint(weights_path))
        self.ln = nn.LayerNorm(self.FEATURE_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone.forward_features(x)   # (N, 768, 7, 7)
        feat_map = feat_map.permute(0, 2, 3, 1)        # -> (N, 7, 7, 768)
        feat_map = self.ln(feat_map)
        return feat_map.mean(dim=(1, 2))               # GAP -> (N, 768)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="BiFormer-B feature extractor self-test")
    p.add_argument("--pretrained", action="store_true",
                   help="Also download/load ImageNet weights (needs network or cache)")
    p.add_argument("--weights-path", default=None, help="Local biformer_base_in1k.pth")
    a = p.parse_args()

    m = BiFormerFeatureExtractor(pretrained=a.pretrained, weights_path=a.weights_path).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (2, BiFormerFeatureExtractor.FEATURE_DIM), out.shape
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    loaded = "ImageNet weights" if (a.pretrained or a.weights_path) else "random init"
    print(f"OK ({loaded}): output shape {tuple(out.shape)}, {n_params:.1f}M params")
