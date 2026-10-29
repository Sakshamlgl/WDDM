"""
Optional sanity check for CheXpert-Plus users: prints 10 CheXbert-positive
samples next to their report text so you can eyeball label quality.
Not part of the WDDM pipeline (the paper uses CheXpert v1.0 train.csv labels;
see build_wddm_manifests.py).

Usage:
    python verify_chexpert_plus_samples.py --chexpert-plus-csv df_chexpert_plus_240401.csv \\
        --findings-json findings_fixed.json
"""
import argparse
import json
from pathlib import Path
import pandas as pd

_ap = argparse.ArgumentParser(description="Eyeball CheXpert-Plus CheXbert labels against report text")
_ap.add_argument("--chexpert-plus-csv", type=Path, required=True)
_ap.add_argument("--findings-json", type=Path, required=True, help="CheXbert labels, one JSON object per line")
_args = _ap.parse_args()
CHEXPERT_CSV = _args.chexpert_plus_csv
FINDINGS_JSON = _args.findings_json

TARGETS = ["Atelectasis", "Cardiomegaly", "Edema", "Consolidation", "Pleural Effusion"]

# Get 10 positive CheXbert examples, preserving the order found.
examples = []
with FINDINGS_JSON.open("r", encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        if any(x.get(k) == 1.0 for k in TARGETS):
            examples.append(x)
            if len(examples) == 10:
                break

paths = [x["path_to_image"] for x in examples]
wanted = set(paths)
rows = {}

print(f"Looking up {len(paths)} samples in the 3.95 GB CSV using chunks...")
for chunk in pd.read_csv(CHEXPERT_CSV, chunksize=50000, low_memory=False):
    hit = chunk[chunk["path_to_image"].isin(wanted)]
    for _, row in hit.iterrows():
        rows[row["path_to_image"]] = row.to_dict()
    if len(rows) == len(wanted):
        break

for i, ann in enumerate(examples, 1):
    path = ann["path_to_image"]
    row = rows.get(path)

    print("\n" + "=" * 100)
    print(f"SAMPLE {i}")
    print("IMAGE PATH :", path)
    print("FOUND IN CSV:", row is not None)

    print("\nCHEXBERT LABELS:")
    for k in TARGETS:
        print(f"  {k:18s}: {ann.get(k)!r}")
    print(f"  {'No Finding':18s}: {ann.get('No Finding')!r}")

    if row is None:
        continue

    print("\nCSV METADATA:")
    print("  split:", row.get("split"))
    print("  frontal_lateral:", row.get("frontal_lateral"))
    print("  ap_pa:", row.get("ap_pa"))

    for field in ["section_findings", "section_impression", "section_summary"]:
        value = row.get(field)
        print(f"\n{field}:")
        print(str(value)[:4000] if pd.notna(value) else "NaN")

    # Highlight obvious target words in the report for a quick human check.
    report = " ".join(
        str(row.get(field, "")) for field in
        ["section_findings", "section_impression", "section_summary"]
        if pd.notna(row.get(field))
    ).lower()

    print("\nTARGET WORD CHECK:")
    for k in TARGETS:
        key = k.lower()
        aliases = {
            "atelectasis": ["atelectasis"],
            "cardiomegaly": ["cardiomegaly"],
            "edema": ["edema"],
            "consolidation": ["consolidation"],
            "pleural effusion": ["pleural effusion", "effusion"],
        }[key]
        hits = [a for a in aliases if a in report]
        print(f"  {k:18s}: {'mentioned' if hits else 'not obviously mentioned'} {hits}")

print("\n" + "=" * 100)
print(f"Matched {len(rows)}/{len(wanted)} sample paths in the CSV.")
if len(rows) < len(wanted):
    missing = wanted - set(rows)
    print("Missing paths:")
    for p in sorted(missing):
        print(" ", p)
