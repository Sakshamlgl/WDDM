# WDDM: Wasserstein Distance + Discrepancy Metric domain adaptation

Reimplementation of He, Chen, Zhu & Xu, "Domain adaptation via Wasserstein
distance and discrepancy metric for chest X-ray image classification",
*Sci Rep* 14:2690 (2024), https://doi.org/10.1038/s41598-024-53311-w.
The main setting is CheXpert (source, labeled) → ChestX-ray14 (target,
unlabeled), which is Exp. 3 in Table 2 of the paper (mean AUC 80.5).

Every default follows the paper. Where the paper doesn't say how to do
something, the choice made here is listed below. Anything that goes beyond
the paper is an opt-in flag.

## Paper → code

| Paper | Code |
|---|---|
| Input module: per-image mean/std normalization, crop, random rotation ±25° | `data/transforms.py` |
| BiFormer-B `[4,4,18,4]`, dims `[96,192,384,768]` → LayerNorm → GAP → FC (Fig. 2–3, Table 1) | `models/backbone.py`, `models/classifier.py`, `third_party/biformer/` |
| Wasserstein distance, dual form with a 1-Lipschitz critic (Eq. 1–3) | `models/critic.py`, `train.py` |
| "Select the source samples closest to the target samples" | `losses/wasserstein.py` |
| Contrast domain difference D_cdd, CAN-style MMD, K-means target labels, fuzzy samples/classes set to zero (Eq. 4) | `losses/contrast_domain.py` |
| Class-aware sampling for both domains | `data/sampling.py` |
| Objective `l = l_c + λW + ηD_cdd` (Eq. 5) | `train.py` |
| Best model chosen by validation AUC each epoch; AUC/Acc/Sens/Spec/PPV/NPV (Eq. 6–12) | `train.py`, `metrics.py`, `eval.py` |
| Five shared lesions, images with none of them excluded, 5-fold CV stratified by lesion proportions | `build_wddm_manifests.py` |

## Settings: paper vs. defaults

| Setting | Paper | Default here |
|---|---|---|
| Backbone | BiFormer-B | BiFormer-B (56.0M params) |
| Head | LayerNorm + GAP + one FC layer | `--head linear` |
| Normalization | per-image mean/std | `--norm per-image` |
| Augmentation | cropping + random rotation ≤ ±25° | pad-16 reflect random crop + rotation ±25° |
| Optimizer | Adam, momentum 0.9, lr 1e-4, decay 1e-4 | Adam(β=0.9, 0.999), `--lr 1e-4`, `--weight-decay 1e-4`, same lr for all layers |
| LR schedule | none mentioned | `--scheduler none` |
| Epochs / batch | 100 / 16 | `--epochs 100 --batch-size 16` (per domain) |
| λ, η | 0.03, 0.2 (Fig. 5) | `--lam 0.03 --eta 0.2` |
| Source samples in W and D_cdd | the closest to the target samples | `--source-selection closest` |
| D_cdd categories | N = the 5 lesion types | `--cdd-mode multiclass` |
| Target labels | K-means, network fixed, fuzzy samples/classes dropped | re-clustered every epoch (`--cluster-every 1`) |
| Sampling | class-aware, both domains | `--sampling class-aware` |
| CheXpert subset | 138,894 images | all views, uncertain → 0: **138,757** on `train.csv` (prevalences within 0.04 points of Fig. 7) |
| ChestX-ray14 subset | 27,167 images | all images with ≥1 of the 5 lesions: **27,195** in `Data_Entry_2017.csv` (prevalences within 0.04 points of Fig. 7) |
| Splits | 5-fold CV, one fold is the test set, the rest is training; plus a validation set | fold k = test (20%), 10% stratified val, 70% train |

### Where the paper is silent

These are this implementation's choices. Report them if you write the work up.

- **Lipschitz constraint:** gradient penalty (WGAN-GP, weight 10) with `--n-critic 5` critic steps per iteration. The critic is a 768→256→256→1 MLP.
- **Pretrained initialization:** BiFormer-B starts from ImageNet-1k weights (`--pretrained`, on by default). The paper doesn't say, but it's standard practice.
- **Multi-label D_cdd:** Eq. 4 is written for categorical labels, but the data is multi-label.
  - A source image belongs to every lesion category it's positive for.
  - Each target image gets one K-means cluster (K = 5).
- **K-means:** spherical (cosine), seeded with the source class centers from `--cluster-source-samples 10000` source images. Runs for `--kmeans-iters 20`.
- **Fuzzy filtering:**
  - Within each cluster, the `--cluster-keep-ratio 0.8` fraction of samples closest to the center is kept.
  - A cluster with fewer than `--cluster-min-size 3` samples is dropped.
  - In each batch, a category takes part only if both domains have at least `--cdd-min-samples 2` samples of it.
- **Kernel:** mean of 5 Gaussian kernels whose bandwidths are spread ×2 around the mean pairwise squared distance in the batch (as in CAN).
- **Class-aware batches:** each batch draws up to `--cas-units 4` random categories, with the same number of images per category from each domain.
- **"Cross-entropy" l_c:** implemented as binary cross-entropy with logits, since the task is multi-label.
- **Validation split:** a stratified 10% slice of the four training folds. It's used for best-epoch selection only; target labels never enter the loss.
- **Split unit:** folds are grouped by patient, so a patient never appears on both sides of a split. The paper only says "randomly divided"; `--split-unit image` gives a plain per-image split.
- **BatchNorm:** BiFormer uses BatchNorm in its stem, downsampling layers and final norm. Source and target go through the network in one forward pass so the batch statistics are shared.

### Opt-in variants (not in the paper)

| Flag | Effect |
|---|---|
| `--norm imagenet` | ImageNet mean/std instead of per-image normalization |
| `--head mlp` | Linear→ReLU→Dropout(0.5)→Linear head |
| `--pos-weight` | class-imbalance weights in the BCE loss |
| `--scheduler cosine` | cosine LR decay |
| `--backbone-lr 1e-5` | smaller LR for the pretrained backbone |
| `--source-selection all` | use the whole source batch for W and D_cdd (the standard unbiased dual estimate) |
| `--cdd-mode binary` | per-lesion negative/positive D_cdd, averaged over lesions |
| `--sampling random` | ordinary random batches |
| `--cdd-start-epoch N` | turn D_cdd on only after N epochs |
| `build_wddm_manifests.py --chexpert-views frontal` / `--chexpert-uncertain one` | other label/view policies; these do not match the paper's counts |

## Setup

```bash
pip install -r requirements.txt
```

### BiFormer-B ImageNet weights (manual download)

The official weights are only on OneDrive, and that link no longer allows
scripted downloads. The vendored URL now returns HTTP 400/401.

1. Open https://github.com/rayleizhu/BiFormer in a browser.
2. Go to "Results and Pre-trained Models" → BiFormer-B → **model** and download the file. It's called `biformer_base_best.pth` and is about 868 MB.
3. Then do one of:
   - pass `--pretrained-weights biformer_base_best.pth` to `train.py`, or
   - copy the file to `~/.cache/torch/hub/checkpoints/biformer_base_in1k.pth`. `--pretrained` then finds it without downloading.

The file is large because it's a full training checkpoint:
- `model` holds the weights actually used: 516 backbone tensors plus the ImageNet head.
- It also contains `model_ema`, optimizer and scheduler state, and the training `args` (epoch 287, 84.31% top-1).

Loading details:
- The file is loaded with `torch.load(weights_only=True)`.
- Only `argparse.Namespace` is allowlisted, plus `PosixPath`/`WindowsPath` mapped to their `Pure*` versions. A real `PosixPath` can't be created on Windows.
- The 1000-class head is dropped, and every other tensor loads strictly, so any other mismatch raises an error.

To check it:

```bash
python models/backbone.py --weights-path biformer_base_best.pth
# OK (ImageNet weights): output shape (2, 768), 56.0M params
```

## Data

The paper's sources:
- CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert/ (the downsampled `CheXpert-v1.0-small` is enough, since images are resized to 224).
- ChestX-ray14: https://www.kaggle.com/datasets/nih-chest-xrays/data

Build the manifests and 5-fold splits:

```bash
python build_wddm_manifests.py \
    --chexpert-csv /data/chexpert/CheXpert-v1.0-small/train.csv \
    --chexpert-root /data/chexpert/CheXpert-v1.0-small \
    --nih-csv /data/cxr14/Data_Entry_2017.csv --nih-root /data/cxr14 \
    --out-dir manifests
```

This writes `{chexpert,cxr14}_all.csv` and `{chexpert,cxr14}_fold{0..4}_{train,val,test}.csv`.
It also prints image counts and prevalences next to the paper's Fig. 7 numbers.

- `check_wddm_label_policies.py` compares all view/uncertainty policies against the paper.
- The 137-image CheXpert gap may be the `valid.csv` images; `--chexpert-valid-csv` adds them. That's unverified.
- The ChestX-ray14 filter gives 27,195 images on the real `Data_Entry_2017.csv` (paper: 27,167), with prevalences 42.50/10.21/48.97/17.16/8.47% (paper: 42.46/10.20/48.98/17.18/8.48%).

## Training on Kaggle (free GPU)

`kaggle_runner/` runs the pipeline as a Kaggle script kernel. Kaggle gives about 30 GPU hours a week, in sessions of at most 12 hours.

- **Code:** packed into the kernel script at generation time, so each push runs your current local code.
- **Data:** the kernel mounts `nih-chest-xrays/data` (ChestX-ray14) and `ashery/chexpert` (CheXpert-v1.0-small). The latter's `train.csv` is byte-identical to Stanford's, but it's a third-party upload, so check your course is OK with it.
- **Weights:** come from a private dataset you create once.
- **Precision:** the kernels train with `--amp` (fp16 backbone, fp32 losses) to make use of the T4's tensor cores. The paper doesn't state its precision.

```bash
# once: needs ~/.kaggle/access_token (Kaggle -> Settings -> API)
pip install kaggle==2.2.4
python kaggle_runner/prepare_weights.py --src biformer_base_best.pth --owner <user>
cd kaggle_runner/build/biformer-base-in1k && kaggle datasets create && cd -   # run from inside the folder (Windows path bug)

# speed/memory check: 200 steps of fold 0
python kaggle_runner/make_kernel.py --owner <user> --slug wddm-timing --preset timing
cd kaggle_runner/build/wddm-timing && kaggle kernels push -p . && cd -
kaggle kernels status <user>/wddm-timing
kaggle kernels output <user>/wddm-timing -p out/wddm-timing

# paper run, fold 0, chained across sessions (each stops before the 11 h mark)
python kaggle_runner/make_kernel.py --owner <user> --slug wddm-f0-s1 --preset train --fold 0
python kaggle_runner/make_kernel.py --owner <user> --slug wddm-f0-s2 --preset train --fold 0 \
    --resume-kernel <user>/wddm-f0-s1
```

How the sessions fit together:
- Each session writes `runs/fold<k>/` (`last.pt`, `best.pt`, `metrics.csv`) and `manifests/` to the kernel output.
- The next session attaches that output, copies it in and continues with `--resume auto`.
- The session that reaches the last epoch also evaluates `best.pt` on `cxr14_fold<k>_test.csv`.
- Anything after `--` is passed to `train.py`.
- Set `WDDM_KAGGLE_INPUT` and `WDDM_KAGGLE_WORKING` to dry-run a generated script locally.

## Training on EC2

Suggested setup:
- **Instance:** a single-GPU instance with 24 GB of GPU memory, e.g. g5.2xlarge (A10G, 8 vCPU), launched from the AWS Deep Learning AMI (Ubuntu, PyTorch).
- **Storage:** a gp3 volume of about 150 GB. CheXpert-small is ~11 GB and ChestX-ray14 is ~45 GB.
- Memory use and speed haven't been measured, so time a short run first.

```bash
# on the instance
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # Linux wheels from PyPI include CUDA
python -c "import torch; print(torch.cuda.is_available())"

# copy the BiFormer-B .pth up from your laptop (scp, or via an S3 bucket)
mkdir -p ~/.cache/torch/hub/checkpoints
cp biformer_base_best.pth ~/.cache/torch/hub/checkpoints/biformer_base_in1k.pth

python build_wddm_manifests.py ... --out-dir manifests      # as above

# 1) time it: 200 steps, 1 epoch
python train.py \
    --source-manifest manifests/chexpert_all.csv --source-root /data/chexpert/CheXpert-v1.0-small \
    --target-manifest manifests/cxr14_fold0_train.csv --target-root /data/cxr14 \
    --val-manifest manifests/cxr14_fold0_val.csv \
    --epochs 1 --max-steps 200 --num-workers 8 --out-dir runs/timing

# 2) the real run (paper settings), inside tmux so it survives disconnects
python train.py \
    --source-manifest manifests/chexpert_all.csv --source-root /data/chexpert/CheXpert-v1.0-small \
    --target-manifest manifests/cxr14_fold0_train.csv --target-root /data/cxr14 \
    --val-manifest manifests/cxr14_fold0_val.csv \
    --num-workers 8 --out-dir runs/wddm_fold0 --resume auto 2>&1 | tee -a runs/wddm_fold0.log

# 3) test-set metrics of the best epoch
python eval.py --checkpoint runs/wddm_fold0/best.pt \
    --manifest manifests/cxr14_fold0_test.csv --root /data/cxr14 \
    --out-csv runs/wddm_fold0/test_metrics.csv --out-probs runs/wddm_fold0/test_probs.csv
```

What each output is for:
- `--resume auto` restarts from `runs/.../last.pt` if it exists, so after an interruption or spot reclaim you just rerun the same command. `last.pt` holds the model, critic, optimizers, epoch and best AUC, and is written atomically.
- `best.pt` is the epoch with the highest validation mean AUC.
- `metrics.csv` has one row per epoch: losses, validation AUC per lesion, epoch time.
- `config.json` records every setting used.
- The clustering log line shows the precision of the target pseudo-labels for monitoring. Target labels are never used in the loss.

For the paper's full protocol, repeat with folds 1–4 and average the test metrics.

**Cost:** one epoch is `138,757 / 16 ≈ 8,700` iterations, each with a 32-image forward/backward pass plus 5 critic steps. Use `epoch_sec` from the timing run to estimate 100 epochs × 5 folds before committing to the full protocol.

The no-adaptation baseline (Exp. 2) is `--lam 0 --eta 0` with the same splits.

## Smoke test (CPU, synthetic data)

```bash
python tests/make_synthetic_data.py
python train.py \
    --source-manifest tests/synthetic/source_manifest.csv --source-root tests/synthetic/source_imgs \
    --target-manifest tests/synthetic/target_manifest.csv --target-root tests/synthetic/target_imgs \
    --val-manifest tests/synthetic/val_manifest.csv --val-root tests/synthetic/val_imgs \
    --epochs 2 --batch-size 4 --num-workers 0 --no-pretrained --n-critic 1 \
    --cluster-source-samples 0 --log-every 1 --out-dir runs/smoke
python eval.py --checkpoint runs/smoke/best.pt \
    --manifest tests/synthetic/val_manifest.csv --root tests/synthetic/val_imgs --num-workers 0
```

Each module also has a `__main__` self-test, e.g. `python losses/contrast_domain.py`.

## Verified vs. not verified

Verified locally (CPU, Python 3.13, the versions in `requirements.txt`):
- Module self-tests pass.
- The weight loader: a checkpoint in the official format (with the 1000-class head) loads exactly, and a checkpoint with a missing tensor is rejected.
- Synthetic end-to-end run: clustering, class-aware batches, critic/W/D_cdd steps, validation, `best.pt`/`last.pt`, resume, and `eval.py`. Covered for the paper defaults and for all opt-in flags together, with `--num-workers 0` and `2`.
- One Windows run with `--num-workers 2` crashed once with an access violation (0xC0000005) and didn't reproduce on rerun. If it happens on Windows, use `--num-workers 0`. Linux (EC2) DataLoader workers use a different start method (fork), so this is unlikely to apply there, but it hasn't been tested.
- The manifest builder on the real CheXpert `train.csv` (138,757 images, patient-disjoint stratified folds). The ChestX-ray14 label filter on the real `Data_Entry_2017.csv` (27,195 images); its image-path indexing was tested on fake data locally.
- CUDA with `--amp` on an RTX 2050, and the Kaggle runner dry-run locally against a fake `/kaggle/input` tree: manifests, training, a time-limit stop, a resumed second session, and the final test-set evaluation.

- The real `biformer_base_best.pth`, loaded both through `--pretrained-weights` and through the torch hub cache path: all 516 backbone tensors match the checkpoint exactly. A synthetic `train.py` run from these weights completes.

Not verified:
- Automatic download of the weights (the upstream link is dead; see above).
- GPU memory and speed.
- Classification quality. Nothing here shows the paper's 80.5 mean AUC is reachable; that needs the real data and a full run.

## Old checkpoints

The `bench_*`, `checkpoints*` folders and the root-level manifest CSVs were
produced by the previous version of this code, before these fixes. That
version used a different head and normalization, and most likely did not
load ImageNet weights. `eval.py` can still open those checkpoints (the head
type is inferred; the normalization falls back to `imagenet` with a warning),
but don't use them for results.
