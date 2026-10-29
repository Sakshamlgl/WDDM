"""
Build a results report for one training run from files train.py/eval.py wrote:
metrics.csv (per epoch), test_probs.csv (per-image test probabilities),
config.json, plus the test manifest (labels).

Outputs in --out-dir:
    report.md               summary tables (test metrics vs the paper, per epoch)
    roc_test.png            per-lesion ROC curves on the test set
    training_curves.png     validation AUC / accuracy and losses per epoch
    test_metrics_ci.csv     test metrics with 95% bootstrap CIs (patient-level)

Everything is computed from the saved probabilities at the 0.5 threshold, the
same as eval.py; no threshold is tuned on the test set.

Usage:
    python make_report.py --run-dir results/wddm-cx-nih-25ep/runs/fold0 \\
        --test-manifest kaggle_runner/out/wddm-timing/manifests/cxr14_fold0_test.csv
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from metrics import compute_metrics  # noqa: E402
from models.classifier import LESION_TYPES  # noqa: E402

# Table 2 of He et al., Sci Rep 2024 (5-lesion means, %).
PAPER = {
    "Paper Exp 2: CheXpert -> ChestX-ray14, no adaptation": [62.7, 58.7, 58.4, 59.9, 31.5, 79.6],
    "Paper Exp 3: CheXpert -> ChestX-ray14, WDDM": [80.5, 74.3, 73.8, 74.4, 45.3, 87.2],
}
COLS = ["AUC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV"]


def bootstrap_ci(probs, labels, groups, n_boot, seed):
    """Patient-level bootstrap of every metric (resample patients, keep all their images)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_of = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    samples = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_of[g] for g in pick])
        m = compute_metrics(probs[idx], labels[idx])
        m.loc["mean"] = m.mean(numeric_only=True)
        samples.append(m.to_numpy())
    s = np.stack(samples)                       # (B, lesions+1, metrics)
    return np.nanpercentile(s, 2.5, axis=0), np.nanpercentile(s, 97.5, axis=0)


def fmt_table(df, digits=1, scale=100):
    head = "| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    body = ["| " + " | ".join([str(i)] + [f"{v * scale:.{digits}f}" if pd.notna(v) else "-" for v in r]) + " |"
            for i, r in df.iterrows()]
    return "\n".join([head, sep, *body])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--test-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <run-dir>/../../report")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = a.out_dir or a.run_dir.parent.parent / "report"
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((a.run_dir / "config.json").read_text())
    hist = pd.read_csv(a.run_dir / "metrics.csv")
    probs_df = pd.read_csv(a.run_dir / "test_probs.csv")
    man = pd.read_csv(a.test_manifest)
    if not (man["path"].values == probs_df["path"].values).all():
        raise ValueError("test manifest and test_probs.csv rows are not aligned")
    probs = probs_df[[f"prob_{c}" for c in LESION_TYPES]].to_numpy()
    labels = man[LESION_TYPES].to_numpy()

    # ---- test metrics + CIs -------------------------------------------------
    test = compute_metrics(probs, labels)
    test.loc["mean"] = test.mean(numeric_only=True)
    lo, hi = bootstrap_ci(probs, labels, man["patient_id"].to_numpy(), a.n_boot, a.seed)
    ci = test.copy()
    for j, col in enumerate(test.columns):
        ci[f"{col}_lo"], ci[f"{col}_hi"] = lo[:, j], hi[:, j]
    ci.to_csv(out / "test_metrics_ci.csv")

    best_ep = int(hist.loc[hist["val_mean_auc"].idxmax(), "epoch"])
    prev = man[LESION_TYPES].mean()
    majority = pd.Series({c: max(p, 1 - p) for c, p in prev.items()})

    # ---- ROC curves -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for j, c in enumerate(LESION_TYPES):
        fpr, tpr, _ = roc_curve(labels[:, j], probs[:, j])
        ax.plot(fpr, tpr, lw=1.8, label=f"{c}: AUC = {roc_auc_score(labels[:, j], probs[:, j]):.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01),
           title=f"CheXpert -> ChestX-ray14 (WDDM), test set\nmean AUC = {test.loc['mean', 'AUC']:.4f}")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "roc_test.png", dpi=160)
    plt.close(fig)

    # ---- training curves ------------------------------------------------------
    ep = hist["epoch"] + 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(ep, hist["val_mean_auc"], "o-", label="val mean AUC")
    for c in LESION_TYPES:
        ax.plot(ep, hist[f"val_auc_{c}"], lw=0.9, alpha=0.6, label=c)
    ax.axvline(best_ep + 1, color="k", ls=":", lw=1, label=f"best epoch ({best_ep + 1})")
    ax.set(title="NIH validation AUC", xlabel="epoch", ylabel="AUC")
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1]
    ax.plot(ep, hist["val_mean_acc"] * 100, "o-", label="val mean accuracy")
    ax.plot(ep, hist["acc_s"] * 100, "s-", ms=3, label="source train accuracy")
    ax.axhline(majority.mean() * 100, color="r", ls="--", lw=1, label="always-majority baseline")
    ax.axvline(best_ep + 1, color="k", ls=":", lw=1)
    ax.set(title="Accuracy (threshold 0.5)", xlabel="epoch", ylabel="%")
    ax.legend(fontsize=8)
    ax = axes[2]
    ax.plot(ep, hist["l_c"], label="l_c (BCE)")
    ax.plot(ep, hist["W"], label="W (Wasserstein)")
    ax.plot(ep, hist["D_cdd"], label="D_cdd")
    ax.plot(ep, hist["total"], "k-", lw=1.5, label="total")
    ax.set(title="Training losses (epoch mean)", xlabel="epoch")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=140)
    plt.close(fig)

    # ---- markdown -------------------------------------------------------------
    cmp = pd.DataFrame(PAPER, index=COLS).T / 100
    cmp.loc["This run: CheXpert -> ChestX-ray14, WDDM"] = test.loc["mean", COLS].to_numpy()
    cmp.index.name = "5-lesion mean (%)"
    per = test.drop(index="mean")[COLS].copy()
    per["Prevalence"] = prev
    per["Majority-class acc."] = majority
    per.index.name = "Lesion"
    m = test.loc["mean"]
    ci_line = ", ".join(f"{c} {m[c] * 100:.1f} [{ci.loc['mean', c + '_lo'] * 100:.1f}, "
                        f"{ci.loc['mean', c + '_hi'] * 100:.1f}]" for c in ("AUC", "Accuracy", "Sensitivity", "Specificity"))
    epochs = hist[["epoch", "val_mean_auc", "val_mean_acc", "acc_s", "l_c", "W", "D_cdd"]].copy()
    epochs["epoch"] += 1
    ep_rows = "\n".join(
        f"| {int(r.epoch)}{' **best**' if int(r.epoch) == best_ep + 1 else ''} | {r.val_mean_auc:.4f} | "
        f"{r.val_mean_acc * 100:.2f} | {r.acc_s * 100:.2f} | {r.l_c:.4f} | {r.W:+.4f} | {r.D_cdd:+.4f} |"
        for r in epochs.itertuples())

    md = f"""# WDDM: CheXpert → ChestX-ray14 results

Reimplementation of He et al., *Sci Rep* 14:2690 (2024). Source: CheXpert-v1.0-small
(labeled, 138,757 images). Target: ChestX-ray14 (NIH), labels not used for training.
Evaluated on a held-out, patient-disjoint NIH test split ({len(man):,} images,
{man['patient_id'].nunique():,} patients) with the checkpoint of the best validation
epoch (epoch {best_ep + 1} of {cfg['epochs']}, validation mean AUC {hist['val_mean_auc'].max():.4f}),
as in the paper.

## Test results vs the paper

{fmt_table(cmp)}

95% patient-level bootstrap CIs ({a.n_boot} resamples): {ci_line}.

![ROC curves](roc_test.png)

## Per-lesion test results (threshold 0.5)

{fmt_table(per)}

Accuracy has to be read against the "majority-class" column: always predicting the
more common answer for each lesion already scores {majority.mean() * 100:.1f}% on average,
because Cardiomegaly, Consolidation and Edema are rare. AUC is threshold-free and is the
metric used for model selection.

## Training history (NIH validation set)

![Training curves](training_curves.png)

| Epoch | Val AUC | Val acc (%) | Source train acc (%) | l_c | W | D_cdd |
|---|---|---|---|---|---|---|
{ep_rows}

## Setup

| Setting | Value |
|---|---|
| Backbone | BiFormer-B, ImageNet-1k init, LayerNorm + GAP + single FC head |
| Loss | BCE + λ·W + η·D_cdd, λ = {cfg['lam']}, η = {cfg['eta']} (paper Fig. 5) |
| Optimizer | Adam, lr {cfg['lr']}, weight decay {cfg['weight_decay']}, batch {cfg['batch_size']} per domain |
| Epochs | {cfg['epochs']}; one epoch = one pass over the NIH train split ({int(hist['steps'].iloc[0])} steps) |
| Target clustering | K-means every {cfg['cluster_every']} epochs, class-aware sampling |
| Normalization | {cfg['norm']} |
| Precision | {'fp16 mixed precision' if cfg.get('amp') else 'fp32'} (1× Tesla T4, Kaggle) |

Differences from the paper: 25 instead of 100 epochs, target-length epochs, re-clustering
every 5 epochs instead of every epoch, a single fold instead of 5-fold cross-validation,
and mixed precision.
"""
    (out / "report.md").write_text(md, encoding="utf-8")
    print(f"wrote {out / 'report.md'}, roc_test.png, training_curves.png, test_metrics_ci.csv")
    print(f"test mean: {ci_line}")


if __name__ == "__main__":
    main()
