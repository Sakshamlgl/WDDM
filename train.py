"""
WDDM training loop, following He et al., "Domain adaptation via Wasserstein
distance and discrepancy metric for chest X-ray image classification",
Sci Rep 14:2690 (2024).

Objective (Eq. 5):   min_theta  l = l_c + lambda * W(Xs, Xt) + eta * D_cdd

Per epoch ("Network optimization and hyperparameters setting"):
  1. Network fixed: extract features, K-means the target train set seeded
     with source class centers, drop fuzzy samples / fuzzy classes -> target
     categories (losses/contrast_domain.py).
  2. Class-aware batches from both domains (data/sampling.py).
  3. Each iteration:
       a. one forward pass over [source; target] so BatchNorm statistics
          are shared by both domains (BiFormer has BN in its stem,
          downsampling layers and final norm);
       b. for each target sample select the closest source sample -- "the
          Wasserstein distance ... is calculated to determine the data
          samples to participate in the contrast domain difference";
       c. critic steps: maximize E[g(f_s,selected)] - E[g(f_t)] under a
          1-Lipschitz constraint (Eq. 2-3; enforced with a gradient penalty,
          which the paper doesn't specify);
       d. network step on l_c + lambda * W + eta * D_cdd.
  4. Mean AUC on the validation set every epoch; the best epoch is kept
     ("the performance of the validation set is used to determine whether
     the current training model is the best").

Paper defaults: BiFormer-B, single FC head, per-image normalization, Adam
(momentum 0.9) lr 1e-4, decay 1e-4, batch 16, 100 epochs, lambda 0.03,
eta 0.2. Every non-paper option is opt-in and listed in README.md.

Usage (paper setting, fold 0):
    python train.py \\
        --source-manifest manifests/chexpert_train.csv --source-root /data/chexpert \\
        --target-manifest manifests/cxr14_fold0_train.csv --target-root /data/cxr14 \\
        --val-manifest manifests/cxr14_fold0_val.csv \\
        --out-dir runs/wddm_fold0
"""
import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data.cache import build_ram_cache, cache_key
from data.datasets import ChestXrayDataset
from data.sampling import SAMPLING_MODES, class_aware_plan, random_plan
from data.transforms import (
    NORM_MODES,
    build_eval_tensor_transform,
    build_eval_transform,
    build_train_tensor_transform,
    build_train_transform,
)
from losses.contrast_domain import (
    CDD_MODES,
    category_groups,
    cluster_target,
    contrast_domain_discrepancy,
    source_category_masks,
)
from losses.wasserstein import select_closest_source_samples
from metrics import compute_metrics, get_predictions
from models.backbone import BiFormerFeatureExtractor
from models.classifier import HEAD_TYPES, LESION_TYPES, ClassifierHead
from models.critic import Critic, gradient_penalty
from models.parallel import DomainBalancedDataParallel


def log(msg: str) -> None:
    print(msg, flush=True)


def available_ram_bytes():
    """Bytes this process can still allocate: min(MemAvailable, cgroup limit - usage). None if unknown."""
    vals = []
    try:
        with open("/proc/meminfo") as f:
            info = {line.split(":")[0]: int(line.split()[1]) for line in f}
        vals.append(info["MemAvailable"] * 1024)
    except (OSError, KeyError, ValueError):
        pass
    for lim_p, cur_p in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                         ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            with open(lim_p) as f:
                lim = f.read().strip()
            with open(cur_p) as f:
                cur = int(f.read())
            if lim != "max" and int(lim) < 2**60:
                vals.append(int(lim) - cur)
            break
        except (OSError, ValueError):
            pass
    return min(vals) if vals else None


def mem_status() -> str:
    """Host memory as the container sees it (Linux); 'n/a' elsewhere."""
    parts = []
    try:
        with open("/proc/meminfo") as f:
            info = {line.split(":")[0]: int(line.split()[1]) for line in f}
        parts.append(f"RAM available {info['MemAvailable'] / 2**20:.1f}/{info['MemTotal'] / 2**20:.1f} GB")
    except (OSError, KeyError, ValueError):
        pass
    for p in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            with open(p) as f:
                parts.append(f"cgroup {int(f.read()) / 2**30:.1f} GB")
            break
        except (OSError, ValueError):
            pass
    try:
        import resource
        parts.append(f"main maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GB")
    except ImportError:
        pass
    return " | ".join(parts) or "n/a"


# --------------------------------------------------------------------------- #
# Reproducibility / IO helpers
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(_worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


def atomic_save(obj, path: Path) -> None:
    """Write to a temp file then rename, so an interrupted save (e.g. a spot
    instance reclaim) never leaves a truncated checkpoint behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def resolve_resume(args) -> Path:
    if not args.resume:
        return None
    p = Path(args.out_dir) / "last.pt" if args.resume == "auto" else Path(args.resume)
    if p.is_file():
        return p
    if args.resume == "auto":
        return None
    raise FileNotFoundError(f"--resume {p} does not exist")


def check_files(ds: ChestXrayDataset, name: str) -> None:
    missing = ds.missing_files(limit=10)
    if missing:
        raise FileNotFoundError(f"{name}: image files not found under {ds.root}, e.g. {missing}")


# --------------------------------------------------------------------------- #
# Clustering step (network fixed)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_features(net, loader, device, amp: bool = False, name: str = "") -> torch.Tensor:
    """`net` is a DomainBalancedDataParallel; it moves the batch to its GPU(s)."""
    net.eval()
    out, t0, n_total, next_report = [], time.time(), len(loader.dataset), 0.25
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
        for batch in loader:
            out.append(net(batch[0]).float())
            done = sum(o.size(0) for o in out)
            if name and done / n_total >= next_report:
                el = time.time() - t0
                log(f"    {name} features: {done:,}/{n_total:,} ({el:.0f}s, {done / max(el, 1e-6):.0f} img/s)")
                next_report += 0.25
    return torch.cat(out)


def category_names(mode: str):
    if mode == "multiclass":
        return list(LESION_TYPES)
    return [f"no-{n}" for n in LESION_TYPES] + list(LESION_TYPES)


def update_target_categories(extractor, args, device, src_loader, src_labels, tgt_loader, tgt_true):
    fs = extract_features(extractor, src_loader, device, args.amp, name="source")
    ft = extract_features(extractor, tgt_loader, device, args.amp, name="target")
    masks = cluster_target(
        fs, src_labels.to(device), ft,
        mode=args.cdd_mode, n_iter=args.kmeans_iters,
        keep_ratio=args.cluster_keep_ratio, min_class_size=args.cluster_min_size,
    ).cpu()

    # Monitoring only: target ground truth is NEVER used for training. It just
    # shows how trustworthy the pseudo-labels are.
    truth = source_category_masks(tgt_true, args.cdd_mode)
    parts = []
    for c, name in enumerate(category_names(args.cdd_mode)):
        n = int(masks[:, c].sum())
        prec = float(truth[masks[:, c], c].float().mean()) if n else float("nan")
        parts.append(f"{name}: n={n} prec={prec:.2f}")
    log(f"  clustering: {int(masks.any(1).sum())}/{len(masks)} target samples kept | " + " | ".join(parts)
        + "  [pseudo-label precision is a diagnostic; target labels are not used in training]")
    log(f"  memory: {mem_status()}")
    return masks


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args):
    run_start = time.time()   # --time-limit-hours counts from here (includes cache building)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    log(f"device={device} | config -> {out_dir / 'config.json'}")

    # ---- data ----------------------------------------------------------------
    train_tf, eval_tf = build_train_transform(args.norm), build_eval_transform(args.norm)
    source_ds = ChestXrayDataset(args.source_manifest, args.source_root, train_tf)
    target_ds = ChestXrayDataset(args.target_manifest, args.target_root, train_tf, return_index=True)
    val_ds = (ChestXrayDataset(args.val_manifest, args.val_root or args.target_root, eval_tf)
              if args.val_manifest else None)
    source_only = args.lam == 0 and args.eta == 0   # no-adaptation baseline: never sees target images
    tensor_tf = {"train": build_train_tensor_transform(args.norm), "eval": build_eval_tensor_transform(args.norm)}

    caches = {}

    def attach_cache(ds, name, kind):
        """Decode `ds` into RAM (shared by datasets over the same manifest). Building
        reads every image, so it also replaces check_files()."""
        key = cache_key(ds.paths, ds.root)
        if key not in caches:
            need = len(ds) * 224 * 224
            avail = available_ram_bytes()
            if avail is not None and need > avail - args.cache_headroom_gb * 2**30:
                log(f"  cache {name}: needs {need / 2**30:.1f} GB, only {avail / 2**30:.1f} GB available "
                    f"(headroom {args.cache_headroom_gb} GB) -> reading {name} from disk instead")
                caches[key] = None
            else:
                caches[key] = build_ram_cache(name, ds.paths, ds.root, workers=args.num_workers, log=log)
        if caches[key] is None:
            check_files(ds, name)
            return ds
        ds.use_cache(caches[key], tensor_tf[kind])
        n = ds.verify_cache(args.cache_verify, seed=args.seed)
        if n:
            log(f"  cache {name}: {n} random {kind} samples identical to the uncached pipeline | {mem_status()}")
        return ds

    # NIH (target/val) first: those PNGs are the expensive ones to decode.
    for ds, name in [(target_ds, "target"), (val_ds, "val"), (source_ds, "source")]:
        if ds is None or (source_only and name == "target"):
            continue
        if args.cache == "ram":
            attach_cache(ds, name, "eval" if name == "val" else "train")
        else:
            check_files(ds, name)
        prev = ", ".join(f"{k} {v * 100:.2f}%" for k, v in ds.label_prevalence().items())
        log(f"{name}: {len(ds)} images | {prev}")
    if source_only:
        log("lam = eta = 0: source-only baseline (target images are not used for training)")
    if val_ds is None:
        log("WARNING: no --val-manifest; best-model selection is disabled (only last.pt is written).")

    prevalence = source_ds.label_prevalence()
    empty = [k for k, r in prevalence.items() if r <= 0 or r >= 1]
    if empty:
        raise ValueError(f"Source labels are constant (all 0 or all 1) for {empty}; check the manifest.")

    kw = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda", worker_init_fn=_worker_init)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, **kw) if val_ds else None

    use_cdd = args.eta > 0
    groups = category_groups(args.cdd_mode, len(LESION_TYPES))
    units = [[c] for c in groups[0]] if args.cdd_mode == "multiclass" else groups
    src_cat_masks = source_category_masks(source_ds.labels, args.cdd_mode).numpy()
    if use_cdd:
        n_sub = len(source_ds) if args.cluster_source_samples <= 0 else min(args.cluster_source_samples, len(source_ds))
        sub_idx = np.sort(np.random.default_rng(args.seed).choice(len(source_ds), n_sub, replace=False)).tolist()
        src_eval_ds = ChestXrayDataset(args.source_manifest, args.source_root, eval_tf)
        tgt_eval_ds = ChestXrayDataset(args.target_manifest, args.target_root, eval_tf)
        if args.cache == "ram":   # same files as source_ds / target_ds, so the caches are reused
            attach_cache(src_eval_ds, "source", "eval")
            attach_cache(tgt_eval_ds, "target", "eval")
        cluster_src_loader = DataLoader(Subset(src_eval_ds, sub_idx), batch_size=args.eval_batch_size, **kw)
        cluster_tgt_loader = DataLoader(tgt_eval_ds, batch_size=args.eval_batch_size, **kw)
        cluster_src_labels = src_eval_ds.labels[sub_idx]

    # ---- model / optimizers --------------------------------------------------
    resume_path = resolve_resume(args)
    extractor = BiFormerFeatureExtractor(
        pretrained=args.pretrained and resume_path is None,
        weights_path=args.pretrained_weights if resume_path is None else None,
    ).to(device)
    classifier = ClassifierHead(in_dim=BiFormerFeatureExtractor.FEATURE_DIM, head=args.head).to(device)
    critic = Critic(in_dim=BiFormerFeatureExtractor.FEATURE_DIM).to(device)

    n_visible = torch.cuda.device_count() if device.type == "cuda" else 0
    n_gpu = n_visible if args.gpus == 0 else min(args.gpus, n_visible)
    if device.type == "cuda" and args.gpus > n_visible:
        log(f"WARNING: --gpus {args.gpus} requested, only {n_visible} visible")
    gpu_ids = list(range(max(n_gpu, 1))) if device.type == "cuda" else []
    net = DomainBalancedDataParallel(extractor, gpu_ids if len(gpu_ids) > 1 else None)
    if device.type == "cuda":
        log(f"backbone on {len(gpu_ids)} GPU(s): " + ", ".join(torch.cuda.get_device_name(i) for i in gpu_ids))
    log(f"memory after model setup: {mem_status()}")

    backbone_lr = args.backbone_lr if args.backbone_lr is not None else args.lr
    main_opt = torch.optim.Adam(
        [{"params": extractor.parameters(), "lr": backbone_lr},
         {"params": classifier.parameters(), "lr": args.lr}],
        betas=(0.9, 0.999), weight_decay=args.weight_decay,
    )
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(main_opt, T_max=args.epochs, eta_min=1e-7)
                 if args.scheduler == "cosine" else None)
    # No weight decay on the critic: it would fight the gradient-penalty Lipschitz constraint.
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    pos_weight = None
    if args.pos_weight:
        pos_weight = torch.tensor([(1.0 - r) / r for r in prevalence], dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_auc = 0, float("-inf")
    if resume_path is not None:
        ck = torch.load(resume_path, map_location=device, weights_only=True)
        extractor.load_state_dict(ck["extractor"])
        classifier.load_state_dict(ck["classifier"])
        critic.load_state_dict(ck["critic"])
        main_opt.load_state_dict(ck["main_opt"])
        critic_opt.load_state_dict(ck["critic_opt"])
        if scheduler is not None and ck.get("scheduler"):
            scheduler.load_state_dict(ck["scheduler"])
        if use_amp and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_epoch, best_auc = ck["epoch"] + 1, ck.get("best_auc", float("-inf"))
        log(f"resumed from {resume_path} at epoch {start_epoch} (best val AUC so far {best_auc:.4f})")

    n_batches = (len(target_ds) if args.epoch_length == "target" else len(source_ds)) // args.batch_size
    if args.max_steps:
        n_batches = min(n_batches, args.max_steps)
    log(f"{n_batches} steps per epoch (--epoch-length {args.epoch_length})")
    metrics_path = out_dir / "metrics.csv"
    target_masks = None
    last_epoch_sec = None

    for epoch in range(start_epoch, args.epochs):
        # Stop cleanly before a hard session limit (e.g. Kaggle's 12 h) so the
        # next session can pick up from last.pt with --resume auto.
        if args.time_limit_hours and last_epoch_sec is not None:
            elapsed = time.time() - run_start
            if elapsed + 1.15 * last_epoch_sec > args.time_limit_hours * 3600:
                log(f"time limit: {elapsed / 3600:.2f} h used, next epoch needs ~{last_epoch_sec / 3600:.2f} h; "
                    f"stopping before epoch {epoch}. Rerun with --resume auto to continue.")
                break
        t0 = time.time()
        rng = np.random.default_rng([args.seed, epoch])
        cdd_on = use_cdd and epoch >= args.cdd_start_epoch

        # 1. Network fixed -> update target clusters.
        cluster_sec = 0.0
        if cdd_on and (target_masks is None or (epoch - args.cdd_start_epoch) % args.cluster_every == 0):
            tc = time.time()
            target_masks = update_target_categories(
                net, args, device, cluster_src_loader, cluster_src_labels,
                cluster_tgt_loader, tgt_eval_ds.labels)
            cluster_sec = time.time() - tc

        # 2. Batch plan for this epoch.
        plan = None
        if args.sampling == "class-aware" and cdd_on and target_masks is not None:
            plan = class_aware_plan(src_cat_masks, target_masks.numpy(), units, args.batch_size,
                                    n_batches, args.cas_units, rng,
                                    min_target_per_category=args.cluster_min_size,
                                    min_per_category=args.cdd_min_samples)
            if plan is None:
                log("  no category has enough target samples; using random batches this epoch")
        if plan is None:
            plan = random_plan(len(source_ds), len(target_ds), args.batch_size, n_batches, rng)
        src_loader = DataLoader(source_ds, batch_sampler=plan[0], **kw)
        tgt_masks_dev = target_masks.to(device) if (cdd_on and target_masks is not None) else None
        if source_only:
            batches = ((sb, None) for sb in src_loader)
        else:
            batches = zip(src_loader, DataLoader(target_ds, batch_sampler=plan[1], **kw))

        # 3. Iterations.
        net.train()
        classifier.train()
        critic.train()
        sums, n_steps = defaultdict(float), 0
        win = defaultdict(float)   # running window since the last printout

        def log_window(ep, done):
            n = max(win["n"], 1)
            with torch.no_grad():
                wb = float(torch.sqrt(sum(p.detach().float().pow(2).sum() for p in extractor.parameters())))
                wh = float(torch.sqrt(sum(p.detach().float().pow(2).sum() for p in classifier.parameters())))
            el = time.time() - t_loop
            eta_min = el / max(done, 1) * (n_batches - done) / 60
            tgt = ("" if source_only else
                   f" | target batch acc {win['acc_t'] / n * 100:5.2f}% (monitor only)")
            log(f"[epoch {ep + 1}/{args.epochs} | step {done}/{n_batches} | {el / 60:.1f} min, ~{eta_min:.0f} min left]\n"
                f"   accuracy : source train {win['acc_s'] / n * 100:5.2f}%{tgt}   (mean of last {int(win['n'])} steps)\n"
                f"   losses   : l_c {win['l_c'] / n:.4f} | W {win['W'] / n:+.4f} (x lam {args.lam} = {args.lam * win['W'] / n:+.4f})"
                f" | D_cdd {win['D_cdd'] / n:+.4f} (x eta {args.eta} = {args.eta * win['D_cdd'] / n:+.4f})"
                f" | critic {win['critic'] / n:+.4f} | total {win['total'] / n:.4f}\n"
                f"   weights  : backbone ||w|| {wb:.2f} | classifier ||w|| {wh:.3f} | lr {main_opt.param_groups[-1]['lr']:.1e}")
            win.clear()

        for i in gpu_ids:
            torch.cuda.reset_peak_memory_stats(i)
        t_loop = time.time()
        for step, ((xs, ys), tb) in enumerate(batches):
            ys = ys.to(device, non_blocking=True)
            n_s = xs.size(0)

            # a. one forward pass for both domains (shared BN statistics; with
            # --gpus 2 every GPU gets half the source AND half the target
            # images). With --amp only the backbone runs in fp16; losses,
            # critic and gradient penalty stay in fp32.
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if source_only:
                    fs = net(xs)
                else:
                    xt, _yt, tidx = tb
                    fs, ft = net(xs, xt)

            if source_only:
                main_opt.zero_grad(set_to_none=True)
                logits_s = classifier(fs.float())
                l_c = bce(logits_s, ys)
                scaler.scale(l_c).backward()
                scaler.step(main_opt)
                scaler.update()
                zero = l_c.new_zeros(())
                acc_s = float(((logits_s.detach() > 0).float() == ys).float().mean())
                for k, v in (("l_c", l_c), ("W", zero), ("D_cdd", zero), ("critic", zero), ("total", l_c)):
                    sums[k] += float(v.detach())
                    win[k] += float(v.detach())
                win["acc_s"] += acc_s
                win["n"] += 1
                sums["acc_s"] += acc_s
                n_steps += 1
                if n_steps == 1:
                    t_after_first = time.time()
                if (step + 1) % args.log_every == 0 or step + 1 == n_batches:
                    log_window(epoch, step + 1)
                continue

            fs, ft = fs.float(), ft.float()
            fs_d, ft_d = fs.detach(), ft.detach()

            # b. closest source sample for each target sample
            if args.source_selection == "closest":
                sel = select_closest_source_samples(fs_d, ft_d)       # (n_t,), may repeat
                cdd_src = torch.unique(sel)
            else:
                sel = cdd_src = torch.arange(n_s, device=device)

            # c. critic steps (critic weights only)
            critic_loss = torch.zeros((), device=device)
            if args.lam > 0:
                critic.requires_grad_(True)
                for _ in range(args.n_critic):
                    critic_opt.zero_grad(set_to_none=True)
                    w_c = critic(fs_d[sel]).mean() - critic(ft_d).mean()
                    critic_loss = -w_c + args.gp_weight * gradient_penalty(critic, fs_d[sel], ft_d)
                    critic_loss.backward()
                    critic_opt.step()
                critic.requires_grad_(False)   # gradients still flow *through* it to the features

            # d. network step
            main_opt.zero_grad(set_to_none=True)
            logits_s = classifier(fs)
            l_c = bce(logits_s, ys)
            w = (critic(fs[sel]).mean() - critic(ft).mean()) if args.lam > 0 else fs.new_zeros(())
            if tgt_masks_dev is not None:
                d_cdd, n_groups = contrast_domain_discrepancy(
                    fs[cdd_src], source_category_masks(ys[cdd_src], args.cdd_mode),
                    ft, tgt_masks_dev[tidx.to(device)], groups, min_samples=args.cdd_min_samples)
            else:
                d_cdd, n_groups = fs.new_zeros(()), 0
            total = l_c + args.lam * w + args.eta * d_cdd
            scaler.scale(total).backward()
            scaler.step(main_opt)
            scaler.update()

            for k, v in (("l_c", l_c), ("W", w), ("D_cdd", d_cdd), ("critic", critic_loss), ("total", total)):
                sums[k] += float(v.detach())
            sums["cdd_groups"] += n_groups
            with torch.no_grad():
                # Accuracy = per-lesion accuracy at threshold 0.5, averaged over the 5 lesions
                # (same definition as the paper's Accuracy / eval.py). Target labels are used
                # for this printout only, never in a loss.
                acc_s = float(((logits_s.detach() > 0).float() == ys).float().mean())
                acc_t = float(((classifier(ft_d) > 0).float() == _yt.to(device)).float().mean())
            for k, v in (("l_c", l_c), ("W", w), ("D_cdd", d_cdd), ("critic", critic_loss), ("total", total)):
                win[k] += float(v.detach())
            win["acc_s"] += acc_s
            win["acc_t"] += acc_t
            win["n"] += 1
            sums["acc_s"] += acc_s
            n_steps += 1
            if n_steps == 1:
                t_after_first = time.time()   # first step includes CUDA/cuDNN warm-up
            if (step + 1) % args.log_every == 0 or step + 1 == n_batches:
                log_window(epoch, step + 1)

        t_end = time.time()
        steady_sec_per_step = ((t_end - t_after_first) / (n_steps - 1) if n_steps > 1
                               else (t_end - t_loop) / max(n_steps, 1))
        if scheduler is not None:
            scheduler.step()

        # 4. Validation + checkpoints.
        row = {"epoch": epoch, "lr": main_opt.param_groups[-1]["lr"], "steps": n_steps,
               "sec_per_step": round(steady_sec_per_step, 3),
               "peak_gpu_gb": round(max((torch.cuda.max_memory_allocated(i) for i in gpu_ids), default=0)
                                    / 1024**3, 2)}
        row.update({k: sums[k] / max(n_steps, 1) for k in ("l_c", "W", "D_cdd", "cdd_groups", "critic", "total", "acc_s")})
        val_auc = val_acc = None
        if val_loader is not None:
            probs, labels = get_predictions(net, classifier, val_loader, device)
            m = compute_metrics(probs, labels)
            val_auc = float(np.nanmean(m["AUC"].to_numpy()))
            val_acc = float(m["Accuracy"].mean())
            row["val_mean_auc"] = val_auc
            row["val_mean_acc"] = val_acc
            row.update({f"val_acc_{k}": float(v) for k, v in m["Accuracy"].items()})
            row.update({f"val_auc_{k}": float(v) for k, v in m["AUC"].items()})

        weights = {"extractor": extractor.state_dict(), "classifier": classifier.state_dict(),
                   "config": vars(args), "epoch": epoch, "val_auc": val_auc}
        if val_auc is not None and val_auc > best_auc:
            best_auc = val_auc
            atomic_save(weights, out_dir / "best.pt")
        if args.save_every and (epoch + 1) % args.save_every == 0:
            atomic_save(weights, out_dir / f"epoch{epoch:03d}.pt")
        atomic_save({**weights, "critic": critic.state_dict(), "main_opt": main_opt.state_dict(),
                     "critic_opt": critic_opt.state_dict(),
                     "scheduler": scheduler.state_dict() if scheduler is not None else None,
                     "scaler": scaler.state_dict() if use_amp else None,
                     "best_auc": best_auc}, out_dir / "last.pt")

        last_epoch_sec = time.time() - t0
        row["cluster_sec"], row["epoch_sec"] = round(cluster_sec, 1), round(last_epoch_sec, 1)
        new_file = not metrics_path.exists()
        with metrics_path.open("a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                wr.writeheader()
            wr.writerow(row)
        vs = (f"NIH val: mean ACCURACY {val_acc * 100:.2f}% | mean AUC {val_auc:.4f} (best {best_auc:.4f})"
              if val_auc is not None else "no val set")
        log(f"== epoch {epoch} done in {row['epoch_sec']:.0f}s (clustering {row['cluster_sec']:.0f}s, "
            f"{row['sec_per_step']:.2f}s/step, peak GPU {row['peak_gpu_gb']:.2f} GB) | l_c={row['l_c']:.4f} "
            f"W={row['W']:.4f} D_cdd={row['D_cdd']:.4f} | {vs}")


def parse_args():
    p = argparse.ArgumentParser(description="WDDM training (He et al., Sci Rep 2024)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    d = p.add_argument_group("data")
    d.add_argument("--source-manifest", required=True)
    d.add_argument("--source-root", required=True)
    d.add_argument("--target-manifest", required=True, help="target TRAIN split (labels unused)")
    d.add_argument("--target-root", required=True)
    d.add_argument("--val-manifest", default=None, help="labeled validation split for best-model selection")
    d.add_argument("--val-root", default=None, help="defaults to --target-root")
    d.add_argument("--norm", choices=NORM_MODES, default="per-image", help="paper: per-image")

    o = p.add_argument_group("optimization (paper defaults)")
    o.add_argument("--epochs", type=int, default=100)
    o.add_argument("--batch-size", type=int, default=16, help="per domain")
    o.add_argument("--lr", type=float, default=1e-4)
    o.add_argument("--backbone-lr", type=float, default=None, help="opt-in: separate backbone LR (default = --lr)")
    o.add_argument("--weight-decay", type=float, default=1e-4, help="paper: 'decay rate 0.0001'")
    o.add_argument("--scheduler", choices=("none", "cosine"), default="none", help="paper: none mentioned")
    o.add_argument("--lam", type=float, default=0.03, help="lambda, Wasserstein weight (paper Fig. 5)")
    o.add_argument("--eta", type=float, default=0.2, help="eta, contrast domain difference weight (paper Fig. 5)")
    o.add_argument("--head", choices=HEAD_TYPES, default="linear", help="paper: single FC layer")
    o.add_argument("--pos-weight", action="store_true", help="opt-in: class-imbalance pos_weight in BCE")
    o.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                   help="ImageNet-1k BiFormer-B init (paper doesn't say; standard practice)")
    o.add_argument("--pretrained-weights", default=None, help="local biformer_base_in1k.pth instead of download")

    w = p.add_argument_group("Wasserstein / critic")
    w.add_argument("--source-selection", choices=("closest", "all"), default="closest",
                   help="paper: closest source sample per target sample, used for W and D_cdd")
    w.add_argument("--n-critic", type=int, default=5)
    w.add_argument("--gp-weight", type=float, default=10.0)
    w.add_argument("--critic-lr", type=float, default=1e-4)

    c = p.add_argument_group("contrast domain difference / clustering")
    c.add_argument("--cdd-mode", choices=CDD_MODES, default="multiclass",
                   help="paper: N=5 lesion categories; 'binary' = per-lesion neg/pos variant")
    c.add_argument("--sampling", choices=SAMPLING_MODES, default="class-aware", help="paper: class-aware")
    c.add_argument("--cas-units", type=int, default=4,
                   help="categories (lesions in binary mode) picked per class-aware batch")
    c.add_argument("--cdd-min-samples", type=int, default=2, help="per-batch min samples per category and domain")
    c.add_argument("--cdd-start-epoch", type=int, default=0)
    c.add_argument("--cluster-every", type=int, default=1, help="epochs between target re-clustering")
    c.add_argument("--cluster-source-samples", type=int, default=10000,
                   help="source images used for class centers (0 = all)")
    c.add_argument("--cluster-keep-ratio", type=float, default=0.8,
                   help="fraction of each cluster kept (closest to center); the rest are 'fuzzy'")
    c.add_argument("--cluster-min-size", type=int, default=3, help="clusters smaller than this are dropped")
    c.add_argument("--kmeans-iters", type=int, default=20)

    r = p.add_argument_group("runtime")
    r.add_argument("--amp", action="store_true",
                   help="fp16 mixed precision for the backbone (CUDA only; big speedup on T4/P100/RTX)")
    r.add_argument("--time-limit-hours", type=float, default=0,
                   help="stop before starting an epoch that would exceed this wall time (0 = off)")
    r.add_argument("--gpus", type=int, default=1,
                   help="GPUs for the backbone (0 = all visible); >1 splits source and target evenly per GPU")
    r.add_argument("--epoch-length", choices=("source", "target"), default="source",
                   help="steps per epoch = len(source or target train set) // batch size "
                        "(the paper doesn't define an epoch for two datasets)")
    r.add_argument("--cache", choices=("none", "ram"), default="none",
                   help="ram: decode + resize every image once into RAM (uint8, ~50 KB/image; bit-identical inputs)")
    r.add_argument("--cache-headroom-gb", type=float, default=8.0,
                   help="skip caching a dataset if it would leave less than this much RAM available")
    r.add_argument("--cache-verify", type=int, default=32,
                   help="random samples per dataset checked against the uncached pipeline")
    r.add_argument("--num-workers", type=int, default=4)
    r.add_argument("--eval-batch-size", type=int, default=64)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--log-every", type=int, default=50)
    r.add_argument("--max-steps", type=int, default=0, help="cap steps per epoch (smoke tests); 0 = full epoch")
    r.add_argument("--save-every", type=int, default=0, help="also keep epochNNN.pt every N epochs; 0 = off")
    r.add_argument("--out-dir", default="checkpoints")
    r.add_argument("--resume", default=None, help="'auto' (out-dir/last.pt if present) or a checkpoint path")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
