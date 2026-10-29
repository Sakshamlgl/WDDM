"""
Evaluate a trained WDDM checkpoint on a labeled test split (Eq. 6-12 of the
paper): per-lesion AUC, Accuracy, Sensitivity, Specificity, PPV, NPV at a
0.5 threshold (AUC is threshold-free), plus the 5-lesion mean that the
paper's Table 2 reports.

Normalization and head type are read from the checkpoint's saved config, so
evaluation always matches how the model was trained.

Usage:
    python eval.py --checkpoint runs/wddm_fold0/best.pt \\
        --manifest manifests/cxr14_fold0_test.csv --root /data/cxr14 \\
        --out-csv runs/wddm_fold0/test_metrics.csv
"""
import argparse
import warnings

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.datasets import ChestXrayDataset
from data.transforms import NORM_MODES, build_eval_transform
from metrics import compute_metrics, get_predictions
from models.backbone import BiFormerFeatureExtractor
from models.classifier import ClassifierHead


def _head_from_state_dict(sd: dict) -> str:
    return "linear" if "net.weight" in sd else "mlp"


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = ckpt.get("config") or {}

    norm = args.norm or cfg.get("norm")
    if norm is None:
        norm = "imagenet"
        warnings.warn("Checkpoint has no saved config (pre-fix run); assuming --norm imagenet. "
                      "Pass --norm explicitly if that's wrong.")
    head = cfg.get("head") or _head_from_state_dict(ckpt["classifier"])

    # pretrained=False: the ImageNet weights would be overwritten immediately.
    extractor = BiFormerFeatureExtractor(pretrained=False).to(device)
    classifier = ClassifierHead(in_dim=BiFormerFeatureExtractor.FEATURE_DIM, head=head).to(device)
    extractor.load_state_dict(ckpt["extractor"])
    classifier.load_state_dict(ckpt["classifier"])

    ds = ChestXrayDataset(args.manifest, args.root, transform=build_eval_transform(norm))
    missing = ds.missing_files()
    if missing:
        raise FileNotFoundError(f"image files not found under {args.root}, e.g. {missing}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    probs, labels = get_predictions(extractor, classifier, loader, device)
    metrics = compute_metrics(probs, labels, threshold=args.threshold)
    metrics.loc["mean"] = metrics.mean(numeric_only=True)

    print(f"checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, norm={norm}, head={head})")
    print(f"test set:   {args.manifest} ({len(ds)} images)\n")
    print((metrics * 100).round(2).to_string())
    if args.out_csv:
        metrics.to_csv(args.out_csv)
        print(f"\nwrote {args.out_csv}")
    if args.out_probs:
        cols = {f"prob_{c}": probs[:, j] for j, c in enumerate(metrics.index[:-1])}
        pd.DataFrame({"path": ds.paths, **cols}).to_csv(args.out_probs, index=False)
        print(f"wrote {args.out_probs}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a WDDM checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--norm", choices=NORM_MODES, default=None, help="override the checkpoint's norm mode")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-csv", default=None, help="save the metrics table")
    p.add_argument("--out-probs", default=None, help="save per-image probabilities (for ROC plots)")
    main(p.parse_args())
