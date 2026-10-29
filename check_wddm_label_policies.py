"""
Compare CheXpert label-harmonization choices against the paper's reported
subset (138,894 images; Fig. 7 prevalences) using the same train.csv that
build_wddm_manifests.py reads, so the check validates the labels you
actually train on.

Candidates: views {all, frontal} x uncertain (-1) policy {zero, one, ignore}.
("ignore" drops any image with an uncertain label among the five, since the
manifest format is strictly 0/1.)

Usage:
    python check_wddm_label_policies.py --chexpert-csv /data/chexpert/train.csv \\
        [--chexpert-valid-csv /data/chexpert/valid.csv]
"""
import argparse
from pathlib import Path

import pandas as pd

LABS = ["Atelectasis", "Cardiomegaly", "Pleural Effusion", "Consolidation", "Edema"]
PAPER_COUNT = 138_894
PAPER_PREV = [24.09, 19.49, 62.10, 10.67, 37.65]


def evaluate(df: pd.DataFrame, views: str, policy: str):
    d = df if views == "all" else df[df["Frontal/Lateral"] == "Frontal"]
    v = d[LABS]
    if policy == "ignore":
        d, v = d[~v.eq(-1).any(axis=1)], v[~v.eq(-1).any(axis=1)]
    pos = v.eq(1) | (v.eq(-1) if policy == "one" else False)
    kept = pos[pos.any(axis=1)]
    return len(kept), (kept.mean() * 100).round(2).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chexpert-csv", type=Path, required=True)
    ap.add_argument("--chexpert-valid-csv", type=Path, default=None)
    args = ap.parse_args()

    csvs = [args.chexpert_csv] + ([args.chexpert_valid_csv] if args.chexpert_valid_csv else [])
    df = pd.concat([pd.read_csv(p, low_memory=False) for p in csvs], ignore_index=True)
    print(f"{len(df):,} rows from {', '.join(map(str, csvs))}")
    print(f"paper: {PAPER_COUNT:,} images, prevalences {PAPER_PREV}\n")
    print(f"{'views':8s} {'uncertain':9s} {'kept':>9s} {'vs paper':>9s}  prevalences (At, Ca, Ef, Co, Ed) | max abs diff")

    rows = []
    for views in ("all", "frontal"):
        for policy in ("zero", "one", "ignore"):
            n, prev = evaluate(df, views, policy)
            dmax = max(abs(a - b) for a, b in zip(prev, PAPER_PREV))
            rows.append((abs(n - PAPER_COUNT), dmax, views, policy))
            print(f"{views:8s} U-{policy:7s} {n:9,d} {n - PAPER_COUNT:+9,d}  {prev} | {dmax:.2f}")
    best = min(rows)
    print(f"\nclosest to the paper: views={best[2]}, uncertain={best[3]} "
          f"(build_wddm_manifests.py defaults: views=all, uncertain=zero)")


if __name__ == "__main__":
    main()
