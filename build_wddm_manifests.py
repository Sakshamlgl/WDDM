"""
Build harmonized manifests + 5-fold splits for CheXpert and ChestX-ray14,
following the paper's "Data set and evaluation indicators" and
"Implementation details" sections.

Label harmonization (paper):
  - Five lesion types shared by the datasets: Atelectasis, Cardiomegaly,
    Effusion (CheXpert: "Pleural Effusion"), Consolidation, Edema.
  - "After excluding the chest radiographs that did not contain these five
    types of lesions, 138,894 samples remained in the CheXpert dataset ...
    27,167 samples remained in the Chest X-ray14 dataset".
  - CheXpert: all views, uncertain (-1) and blank -> 0 ("U-Zeros"). This is
    the combination that reproduces the paper's count and Fig. 7 prevalences
    on CheXpert train.csv (138,757 images, prevalences within 0.04 points).
    Frontal-only or U-Ones give clearly different numbers, so they're opt-in.
  - ChestX-ray14: every image in Data_Entry_2017.csv (the paper's 27,167
    matches the full dataset, not the official train/val list).

Splits (paper): "fivefold cross-validation ... based on the proportions of
the five lesion types ... one part is randomly selected as the test set, and
the remaining 4 parts are integrated with the source domain instances as the
training set", with a validation set used to pick the best epoch, and "the
proportion of lesion types in the training set, test set, and validation set
close". So per fold k:
    test  = fold k                                        (~20%)
    val   = a stratified slice of the other 4 folds       (~--val-frac, 10%)
    train = the rest                                      (~70%)
Stratification is on the 5-lesion label pattern. By default folds are
grouped by patient so one patient's images never sit on both sides of a
split (the paper says "randomly divided" without specifying; use
--split-unit image for a plain per-image split).

Outputs (in --out-dir):
    chexpert_all.csv, cxr14_all.csv                      full filtered sets
    {dataset}_fold{k}_{train,val,test}.csv               k = 0..folds-1
Columns: path, patient_id, Atelectasis, Cardiomegaly, Effusion, Consolidation, Edema
`path` is relative to the dataset root you pass (and later give train.py).

Example (EC2):
    python build_wddm_manifests.py \\
        --chexpert-csv /data/chexpert/train.csv --chexpert-root /data/chexpert \\
        --nih-csv /data/cxr14/Data_Entry_2017.csv --nih-root /data/cxr14 \\
        --out-dir manifests
"""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Consolidation", "Edema"]
CHEX_COLS = ["Atelectasis", "Cardiomegaly", "Pleural Effusion", "Consolidation", "Edema"]

PAPER = {
    "chexpert": (138_894, [24.09, 19.49, 62.10, 10.67, 37.65]),
    "cxr14": (27_167, [42.46, 10.20, 48.98, 17.18, 8.48]),
}


# --------------------------------------------------------------------------- #
# Loading / harmonization
# --------------------------------------------------------------------------- #
def load_chexpert(csv_paths: list[Path], root: Path, views: str, uncertain: str,
                  check_files: bool) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p, low_memory=False) for p in csv_paths], ignore_index=True)
    missing_cols = [c for c in ["Path", "Frontal/Lateral", *CHEX_COLS] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CheXpert CSV is missing columns {missing_cols}")
    print(f"CheXpert: {len(df):,} rows in {', '.join(str(p) for p in csv_paths)}")
    if views == "frontal":
        df = df[df["Frontal/Lateral"] == "Frontal"]
        print(f"  frontal only: {len(df):,} rows")

    out = pd.DataFrame(index=df.index)
    for src, dst in zip(CHEX_COLS, LABELS):
        pos = df[src] == 1
        if uncertain == "one":
            pos |= df[src] == -1
        out[dst] = pos.astype("int8")

    # "CheXpert-v1.0-small/train/patient00001/..." -> "train/patient00001/..."
    out["path"] = df["Path"].astype(str).str.replace(r"^[^/]*/(?=(?:train|valid)/)", "", regex=True)
    out["patient_id"] = df["Path"].astype(str).str.extract(r"(patient\d+)", expand=False)
    out = out[out[LABELS].eq(1).any(axis=1)].reset_index(drop=True)
    print(f"  with >=1 of the 5 lesions (uncertain -> {1 if uncertain == 'one' else 0}): {len(out):,}")

    if check_files:
        exists = out["path"].map(lambda p: (root / p).is_file())
        if not exists.all():
            bad = out.loc[~exists, "path"].head(5).tolist()
            raise FileNotFoundError(f"{(~exists).sum():,} CheXpert images not found under {root}, e.g. {bad}")
    return out[["path", "patient_id", *LABELS]]


def load_cxr14(meta_csv: Path, root: Path, image_list: Path | None, check_files: bool) -> pd.DataFrame:
    meta = pd.read_csv(meta_csv, usecols=["Image Index", "Finding Labels", "Patient ID"])
    print(f"ChestX-ray14: {len(meta):,} rows in {meta_csv}")
    if image_list is not None:
        keep = {ln.strip() for ln in image_list.read_text(encoding="utf-8").splitlines() if ln.strip()}
        meta = meta[meta["Image Index"].isin(keep)]
        print(f"  restricted to {image_list.name}: {len(meta):,} rows")

    found = meta["Finding Labels"].fillna("").map(lambda s: {x.strip() for x in s.split("|")})
    out = pd.DataFrame({lab: found.map(lambda f, lab=lab: int(lab in f)).astype("int8") for lab in LABELS})
    out["patient_id"] = meta["Patient ID"].map(lambda x: f"cxr14_{int(x):05d}").to_numpy()
    out["name"] = meta["Image Index"].to_numpy()
    out = out[out[LABELS].eq(1).any(axis=1)].reset_index(drop=True)
    print(f"  with >=1 of the 5 lesions: {len(out):,}")

    # Kaggle layout is images_001/images/*.png ... images_012; NIH box is images/*.png.
    # Index whatever is there by filename.
    image_map: dict[str, str] = {}
    for p in root.rglob("*.png"):
        if p.name in image_map:
            raise RuntimeError(f"Duplicate ChestX-ray14 filename under {root}: {p.name}")
        image_map[p.name] = p.relative_to(root).as_posix()
    out["path"] = out["name"].map(image_map)
    missing = out["path"].isna()
    if missing.any():
        msg = f"{missing.sum():,} ChestX-ray14 images not found under {root}, e.g. {out.loc[missing, 'name'].head(5).tolist()}"
        if check_files:
            raise FileNotFoundError(msg)
        warnings.warn(msg + " -- dropping them (--skip-file-check)")
        out = out[~missing].reset_index(drop=True)
    return out[["path", "patient_id", *LABELS]]


def subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Label-pattern-proportional subsample (for cheap experiments only)."""
    if n <= 0 or n >= len(df):
        return df
    pattern = df[LABELS].astype(str).agg("".join, axis=1)
    groups = dict(list(df.groupby(pattern)))
    raw = {p: n * len(g) / len(df) for p, g in groups.items()}
    take = {p: min(len(groups[p]), math.floor(r)) for p, r in raw.items()}
    for p in sorted(raw, key=lambda p: raw[p] - math.floor(raw[p]), reverse=True):
        if sum(take.values()) >= n:
            break
        if take[p] < len(groups[p]):
            take[p] += 1
    parts = [g.sample(n=take[p], random_state=seed) for p, g in groups.items() if take[p]]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def _pattern(df: pd.DataFrame) -> np.ndarray:
    return (df[LABELS].to_numpy().astype(int) * (1 << np.arange(len(LABELS)))).sum(axis=1)


def _split(df: pd.DataFrame, n_splits: int, seed: int, by_patient: bool):
    """Yield test-index arrays of a stratified (optionally patient-grouped) K-fold."""
    y = _pattern(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # rare label patterns (< n_splits members)
        if by_patient:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            parts = splitter.split(np.zeros(len(df)), y, df["patient_id"].to_numpy())
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            parts = splitter.split(np.zeros(len(df)), y)
        for _, test_idx in parts:
            yield test_idx


def make_folds(df: pd.DataFrame, n_folds: int, val_frac: float, seed: int, by_patient: bool):
    fold_of = np.full(len(df), -1)
    for k, idx in enumerate(_split(df, n_folds, seed, by_patient)):
        fold_of[idx] = k
    n_val_splits = max(2, round((1 - 1 / n_folds) / val_frac))
    for k in range(n_folds):
        test = df[fold_of == k]
        rest = df[fold_of != k].reset_index(drop=True)
        val_idx = next(_split(rest, n_val_splits, seed + 1 + k, by_patient))
        is_val = np.zeros(len(rest), dtype=bool)
        is_val[val_idx] = True
        yield k, rest[~is_val], rest[is_val], test


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def report(df: pd.DataFrame, name: str, key: str | None = None) -> None:
    prev = (df[LABELS].mean() * 100).round(2)
    line = "  ".join(f"{lab} {prev[lab]:5.2f}%" for lab in LABELS)
    print(f"  {name:<28s} {len(df):>7,} imgs {df['patient_id'].nunique():>6,} pts | {line}")
    if key:
        n_paper, prev_paper = PAPER[key]
        diff = len(df) - n_paper
        print(f"  {'paper':<28s} {n_paper:>7,} imgs  (diff {diff:+,}, {diff / n_paper * 100:+.2f}%) | "
              + "  ".join(f"{lab} {p:5.2f}%" for lab, p in zip(LABELS, prev_paper)))


def write_dataset(df: pd.DataFrame, key: str, args) -> None:
    out = Path(args.out_dir)
    df.to_csv(out / f"{key}_all.csv", index=False)
    print(f"\n{key}:")
    report(df, f"{key}_all.csv", key)
    by_patient = args.split_unit == "patient"
    for k, tr, va, te in make_folds(df, args.folds, args.val_frac, args.seed, by_patient):
        for split, part in (("train", tr), ("val", va), ("test", te)):
            part.to_csv(out / f"{key}_fold{k}_{split}.csv", index=False)
            report(part, f"{key}_fold{k}_{split}.csv")
        if by_patient:
            assert not (set(tr.patient_id) & set(te.patient_id)), "patient leak train/test"
            assert not (set(va.patient_id) & set(te.patient_id)), "patient leak val/test"
            assert not (set(tr.patient_id) & set(va.patient_id)), "patient leak train/val"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--chexpert-csv", type=Path, help="CheXpert train.csv")
    ap.add_argument("--chexpert-valid-csv", type=Path, default=None,
                    help="optionally also include CheXpert valid.csv images")
    ap.add_argument("--chexpert-root", type=Path, help="directory that contains train/ (and valid/)")
    ap.add_argument("--chexpert-views", choices=("all", "frontal"), default="all", help="paper: all")
    ap.add_argument("--chexpert-uncertain", choices=("zero", "one"), default="zero", help="paper: zero")
    ap.add_argument("--nih-csv", type=Path, help="ChestX-ray14 Data_Entry_2017.csv")
    ap.add_argument("--nih-root", type=Path, help="directory containing the ChestX-ray14 PNGs (any depth)")
    ap.add_argument("--nih-image-list", type=Path, default=None,
                    help="optional: restrict to e.g. train_val_list.txt (paper uses all images)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.1, help="validation share of the whole dataset")
    ap.add_argument("--split-unit", choices=("patient", "image"), default="patient")
    ap.add_argument("--max-images", type=int, default=0, help="subsample each dataset (0 = all; paper: all)")
    ap.add_argument("--skip-file-check", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("manifests"))
    args = ap.parse_args()

    if not (args.chexpert_csv or args.nih_csv):
        ap.error("give --chexpert-csv/--chexpert-root and/or --nih-csv/--nih-root")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    check = not args.skip_file_check

    if args.chexpert_csv:
        if not args.chexpert_root:
            ap.error("--chexpert-root is required with --chexpert-csv")
        csvs = [args.chexpert_csv] + ([args.chexpert_valid_csv] if args.chexpert_valid_csv else [])
        chex = load_chexpert(csvs, args.chexpert_root, args.chexpert_views, args.chexpert_uncertain, check)
        write_dataset(subsample(chex, args.max_images, args.seed), "chexpert", args)

    if args.nih_csv:
        if not args.nih_root:
            ap.error("--nih-root is required with --nih-csv")
        nih = load_cxr14(args.nih_csv, args.nih_root, args.nih_image_list, check)
        write_dataset(subsample(nih, args.max_images, args.seed), "cxr14", args)

    print(f"\nManifests written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
