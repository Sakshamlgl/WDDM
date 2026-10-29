"""
Evaluation metrics matching Eq. 6-12 of the paper: per-lesion AUC, Accuracy,
Sensitivity, Specificity, PPV, NPV. Thresholding only affects the
confusion-matrix metrics; AUC is threshold-free. Shared by train.py
(per-epoch validation / best-model selection) and eval.py.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from models.classifier import LESION_TYPES


@torch.no_grad()
def get_predictions(extractor, classifier, loader, device):
    extractor.eval()
    classifier.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        probs = torch.sigmoid(classifier(extractor(x))).float().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    preds = (probs >= threshold).astype(int)
    rows = []
    for j, name in enumerate(LESION_TYPES):
        y_true, y_pred, y_prob = labels[:, j], preds[:, j], probs[:, j]
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
        rows.append({
            "lesion": name,
            "AUC": auc,
            "Accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),   # Eq. 8
            "Sensitivity": tp / max(tp + fn, 1),                 # Eq. 9
            "Specificity": tn / max(tn + fp, 1),                 # Eq. 10
            "PPV": tp / max(tp + fp, 1),                         # Eq. 11
            "NPV": tn / max(tn + fn, 1),                         # Eq. 12
        })
    return pd.DataFrame(rows).set_index("lesion")
