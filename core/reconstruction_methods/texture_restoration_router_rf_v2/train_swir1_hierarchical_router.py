#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the FINAL TPSSR hierarchical router.

Architecture
------------
Stage 1:
    Spectral vs Deep-v3.1
Stage 2:
    learned hard-case gate decides whether Final Region should override Stage 1.

Final Region:
    Frozen V27 Stage-A -> V13 TopologyFirst V1

This script deliberately reuses the proven V31 feature encoder and initializes
all compatible weights from the existing V31 component router.  Only the Region
oracle is replaced with the FINAL Region branch.

No SWIR1 benchmark masks are used for training.  Oracle samples are generated
from the normal global training/validation cubes (e.g. SWIR2-5 when SWIR1 is the
held-out test dataset).
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

HERE = Path(__file__).resolve()
CORE = next(p for p in HERE.parents if (p / "config.py").exists())
METHOD_DIR = HERE.parent
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

import config
from src.evaluation_utils import load_global_training_data
from src.hierarchical_component_router import HierarchicalComponentRouter
from src.component_three_expert_router import ComponentThreeExpertRouter
from src.final_tpssr_runtime import load_models, reconstruct_mask
import router_training_support as OLD


SEED = int(getattr(config, "SEED", 1234)) + 31101

# Keep the proven V31 training-bank sizes unless the user overrides them.
TRAIN_SAMPLES = int(
    getattr(config, "V31_COMPONENT_ROUTER_CACHE_TRAIN_SAMPLES", 4000)
)
VAL_SAMPLES = int(
    getattr(config, "V31_COMPONENT_ROUTER_CACHE_VAL_SAMPLES", 1000)
)
EPOCHS = 24
BATCH_SIZE = int(getattr(config, "V31_COMPONENT_ROUTER_BATCH_SIZE", 8))
NUM_WORKERS = int(getattr(config, "V31_COMPONENT_ROUTER_NUM_WORKERS", 2))
LR = 6.0e-5
WEIGHT_DECAY = 1.0e-4
GRAD_CLIP = 1.0
EMA_DECAY = 0.999
MIN_EPOCHS = 10
PATIENCE = 7
MAX_PREVIEW_PIXELS = int(
    getattr(config, "V31_COMPONENT_ROUTER_PREVIEW_PIXELS", 32)
)
MAX_ORACLE_PIXELS = int(
    getattr(config, "V31_COMPONENT_ROUTER_ORACLE_PIXELS", 48)
)
REGION_BATCH = 8

MODEL_SUBDIR = (
    f"hierarchical_router_{str(config.TEST_DATASET_FOLDER).lower()}_dev_2gpu"
)
BEST_MODEL = "best_model.pt"


class EMA:
    def __init__(self, model, decay):
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        src = model.state_dict()
        dst = self.model.state_dict()
        for k in dst:
            if dst[k].dtype.is_floating_point:
                dst[k].mul_(self.decay).add_(
                    src[k], alpha=1.0 - self.decay
                )
            else:
                dst[k].copy_(src[k])


def parse_args():
    p = argparse.ArgumentParser(
        description="Train final hierarchical Spectral/Deep -> Region router."
    )
    p.add_argument("--gpus", type=int, default=2, choices=[1, 2])
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    return p.parse_args()


def launch_if_needed(args):
    if args.worker or args.gpus == 1 or "RANK" in os.environ:
        return False
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.gpus}",
        str(HERE),
        "--gpus",
        "1",
        "--worker",
        "--epochs",
        str(args.epochs),
    ]
    if args.force_cache:
        cmd.append("--force-cache")
    print("Launching final hierarchical-router training:")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(CORE), check=True)
    return True


def setup_runtime():
    distributed = (
        "RANK" in os.environ
        and int(os.environ.get("WORLD_SIZE", "1")) > 1
    )
    if distributed:
        lr = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(lr)
        dist.init_process_group(
            "nccl", timeout=timedelta(minutes=360)
        )
        return (
            True,
            dist.get_rank(),
            dist.get_world_size(),
            torch.device("cuda", lr),
        )
    dev = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    if dev.type == "cuda":
        torch.cuda.set_device(0)
    return False, 0, 1, dev


def cleanup(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(
            path, map_location=map_location, weights_only=False
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def old_router_paths():
    root = Path(config.MODELS_DIR) / "texture_restoration_router_rf_v2"
    tag = str(config.TEST_DATASET_FOLDER).lower()
    model_dir = root / f"component_three_expert_router_{tag}_dev_2gpu"
    return (
        model_dir / "best_model.pt",
        model_dir / "oracle_cache" / "train_component_oracle.npz",
        model_dir / "oracle_cache" / "val_component_oracle.npz",
    )


def old_cache_usable(path: Path, expected_n: int):
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            required = {
                "spectral_candidate",
                "spectral_available",
                "spectral_confidence",
                "deep_candidate",
                "true_error",
                "expert_available",
            }
            return (
                required.issubset(z.files)
                and len(z["true_error"]) == int(expected_n)
                and z["true_error"].shape[1] >= 2
            )
    except Exception:
        return False


def load_old_cache(path):
    with np.load(path, allow_pickle=False) as z:
        return {
            k: z[k].copy()
            for k in z.files
            if k != "signature"
        }


def cache_signature(
    split,
    samples,
    old_cache_path,
    tf_path,
    v27_path,
):
    payload = {
        "version": HierarchicalComponentRouter.ARCHITECTURE_VERSION,
        "heldout": str(config.TEST_DATASET_FOLDER),
        "split": str(split),
        "samples": int(samples),
        "seed": int(config.SEED),
        "old_cache": str(old_cache_path),
        "old_cache_mtime": (
            Path(old_cache_path).stat().st_mtime_ns
            if Path(old_cache_path).exists()
            else None
        ),
        "topology": str(tf_path),
        "topology_mtime": Path(tf_path).stat().st_mtime_ns,
        "v27": str(v27_path),
        "v27_mtime": Path(v27_path).stat().st_mtime_ns,
        "oracle_pixels": int(MAX_ORACLE_PIXELS),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def cache_valid(path, signature):
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            needed = {
                "spectral_candidate",
                "spectral_available",
                "spectral_confidence",
                "deep_candidate",
                "true_error",
                "expert_available",
                "stage1_oracle",
                "gate_target",
                "final_oracle",
                "signature",
            }
            return (
                needed.issubset(z.files)
                and str(z["signature"].item()) == signature
            )
    except Exception:
        return False


def load_cache(path):
    with np.load(path, allow_pickle=False) as z:
        return {
            k: z[k].copy()
            for k in z.files
            if k != "signature"
        }


def save_rank_fragment(path, rows):
    if not rows:
        np.savez_compressed(
            path,
            indices=np.empty(0, dtype=np.int64),
            region_error=np.empty(0, dtype=np.float32),
            region_available=np.empty(0, dtype=np.uint8),
        )
        return
    ids = np.asarray(sorted(rows), dtype=np.int64)
    np.savez_compressed(
        path,
        indices=ids,
        region_error=np.asarray(
            [rows[int(i)][0] for i in ids], dtype=np.float32
        ),
        region_available=np.asarray(
            [rows[int(i)][1] for i in ids], dtype=np.uint8
        ),
    )


def load_rank_fragment(path):
    rows = {}
    if not path.exists():
        return rows
    try:
        with np.load(path, allow_pickle=False) as z:
            ids = z["indices"].astype(np.int64)
            errs = z["region_error"].astype(np.float32)
            av = z["region_available"].astype(np.uint8)
            for i, e, a in zip(ids, errs, av):
                rows[int(i)] = (float(e), int(a))
    except Exception:
        return {}
    return rows


def compute_spec_deep_fallback(
    base,
    item,
    deep_model,
    deep_ck,
    device,
):
    clean = np.asarray(item["clean"], dtype=np.float32)
    mask = np.asarray(item["mask"], dtype=bool)
    comp = np.asarray(item["component"], dtype=bool)
    band = int(item["target_band"].item())

    comp_coords = np.argwhere(comp)
    preview_pixels = OLD._choose_pixels(
        comp_coords, MAX_PREVIEW_PIXELS
    )
    target_coords = np.argwhere(comp & mask[:, :, band])
    target_pixels = OLD._choose_pixels(
        target_coords, MAX_ORACLE_PIXELS
    )
    union = np.unique(
        np.vstack([preview_pixels, target_pixels]), axis=0
    )
    spec, sav, sconf, deep = OLD._pixel_previews(
        deep_model,
        deep_ck,
        clean,
        mask,
        union,
        base.band_mean,
        base.band_std,
        base.wavelengths,
        device,
    )
    lookup = {
        tuple(map(int, p)): i for i, p in enumerate(union)
    }

    ps = [
        lookup[tuple(map(int, p))]
        for p in preview_pixels
    ]
    nb = len(base.band_mean)
    spec_sum = np.zeros(nb, np.float32)
    spec_av = np.zeros(nb, np.float32)
    spec_conf = np.zeros(nb, np.float32)
    deep_sum = np.zeros(nb, np.float32)
    for b in range(nb):
        valid_pix = [
            j
            for j, p in zip(ps, preview_pixels)
            if mask[int(p[0]), int(p[1]), b]
        ]
        use = valid_pix if valid_pix else ps
        if use:
            vals = spec[use, b]
            av = sav[use, b] > 0.5
            spec_av[b] = float(np.mean(av))
            spec_conf[b] = float(np.nanmean(sconf[use, b]))
            deep_sum[b] = float(np.nanmean(deep[use, b]))
            spec_sum[b] = (
                float(np.nanmean(vals[av]))
                if np.any(av)
                else 0.0
            )

    ids = [
        lookup[tuple(map(int, p))]
        for p in target_pixels
    ]
    truth_norm = (
        clean[
            target_pixels[:, 0],
            target_pixels[:, 1],
            band,
        ]
        - base.band_mean[band]
    ) / base.band_std[band]
    svals = spec[ids, band]
    savals = sav[ids, band] > 0.5
    dvals = deep[ids, band]

    s_available = bool(
        np.all(savals) and np.all(np.isfinite(svals))
    )
    d_available = bool(np.all(np.isfinite(dvals)))
    s_rmse = (
        float(np.sqrt(np.mean((svals - truth_norm) ** 2)))
        if s_available
        else 1e6
    )
    d_rmse = (
        float(np.sqrt(np.mean((dvals - truth_norm) ** 2)))
        if d_available
        else 1e6
    )
    return (
        spec_sum,
        spec_av,
        spec_conf,
        deep_sum,
        s_rmse,
        d_rmse,
        s_available,
        d_available,
        target_pixels,
    )


@torch.inference_mode()
def final_region_error(
    tf_model,
    tf_ck,
    v27_model,
    clean,
    mask,
    comp,
    band,
    target_pixels,
    band_std,
    device,
):
    coords3 = np.column_stack(
        [
            target_pixels,
            np.full(
                len(target_pixels),
                int(band),
                dtype=np.int64,
            ),
        ]
    ).astype(np.int64)

    band_predictions, _ = reconstruct_mask(
        tf_model,
        tf_ck,
        v27_model,
        clean,
        mask,
        device,
        eval_coords=coords3,
        batch_size=REGION_BATCH,
        verify_batch=False,
        progress_desc=None,
    )
    if int(band) not in band_predictions:
        return 1e6, 0

    pred_coords, pred_values = band_predictions[int(band)]
    lookup = {
        (int(y), int(x)): float(v)
        for (y, x), v in zip(pred_coords, pred_values)
    }
    values = []
    truth = []
    for y, x in target_pixels:
        key = (int(y), int(x))
        if key not in lookup:
            return 1e6, 0
        values.append(lookup[key])
        truth.append(float(clean[int(y), int(x), int(band)]))
    values = np.asarray(values, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if not (
        np.all(np.isfinite(values))
        and np.all(np.isfinite(truth))
    ):
        return 1e6, 0
    rmse_norm = float(
        np.sqrt(
            np.mean(
                ((values - truth) / float(band_std[int(band)]))
                ** 2
            )
        )
    )
    return rmse_norm, int(np.isfinite(rmse_norm))


def build_hierarchical_cache(
    split,
    base,
    out_path,
    signature,
    old_cache_path,
    deep_model,
    deep_ck,
    tf_model,
    tf_ck,
    v27_model,
    device,
    rank,
    world,
    force,
):
    valid = cache_valid(out_path, signature) and not force
    if world > 1:
        t = torch.tensor(
            int(valid if rank == 0 else 0), device=device
        )
        dist.broadcast(t, 0)
        valid = bool(t.item())
    if valid:
        if rank == 0:
            print("Reuse hierarchical oracle cache:", out_path)
        return

    old_cache = (
        load_old_cache(old_cache_path)
        if old_cache_usable(old_cache_path, len(base))
        else None
    )
    if rank == 0:
        if old_cache is not None:
            print(
                f"{split}: reusing old V31 Spectral/Deep cache; "
                "only Final Region errors are recomputed."
            )
        else:
            print(
                f"{split}: old V31 cache unavailable; "
                "Spectral/Deep previews will also be recomputed."
            )

    frag = out_path.with_name(
        out_path.stem + f".rank{rank}.partial.npz"
    )
    partial = (
        {} if force else load_rank_fragment(frag)
    )

    assigned = np.arange(
        rank, len(base), world, dtype=np.int64
    )
    remaining = [
        int(i) for i in assigned
        if int(i) not in partial
    ]
    bar = tqdm(
        remaining,
        desc=f"GPU {rank} {split} final-region oracle",
        disable=(world > 1 and rank != 0),
        dynamic_ncols=True,
    )

    for done_i, idx in enumerate(bar, start=1):
        item = base.build(int(idx), True)
        clean = np.asarray(item["clean"], dtype=np.float32)
        mask = np.asarray(item["mask"], dtype=bool)
        comp = np.asarray(item["component"], dtype=bool)
        band = int(item["target_band"].item())

        target_coords = np.argwhere(
            comp & mask[:, :, band]
        )
        target_pixels = OLD._choose_pixels(
            target_coords, MAX_ORACLE_PIXELS
        )
        if len(target_pixels) == 0:
            partial[int(idx)] = (1e6, 0)
        else:
            rerr, rav = final_region_error(
                tf_model,
                tf_ck,
                v27_model,
                clean,
                mask,
                comp,
                band,
                target_pixels,
                base.band_std,
                device,
            )
            partial[int(idx)] = (rerr, rav)

        # Resume-safe progress file.  Cache construction is the expensive
        # stage, so an interrupted run does not lose hours of work.
        if done_i % 50 == 0:
            save_rank_fragment(frag, partial)

    save_rank_fragment(frag, partial)

    if world > 1:
        dist.barrier()

    if rank == 0:
        n = len(base)
        B = len(base.band_mean)
        sc = np.zeros((n, B), np.float32)
        sa = np.zeros((n, B), np.float32)
        sf = np.zeros((n, B), np.float32)
        dc = np.zeros((n, B), np.float32)
        te = np.full((n, 3), 1e6, np.float32)
        ea = np.zeros((n, 3), np.uint8)
        seen = np.zeros(n, bool)

        # Fill Spectral / Deep inputs and their oracle errors.
        if old_cache is not None:
            sc[:] = old_cache["spectral_candidate"]
            sa[:] = old_cache["spectral_available"]
            sf[:] = old_cache["spectral_confidence"]
            dc[:] = old_cache["deep_candidate"]
            te[:, :2] = old_cache["true_error"][:, :2]
            ea[:, :2] = old_cache["expert_available"][:, :2]
        else:
            # Fallback is intentionally done on rank 0 only. This path should
            # rarely be used because an existing V31 router implies its cache
            # normally exists.
            for idx in tqdm(
                range(n),
                desc=f"{split} fallback Spectral/Deep cache",
                dynamic_ncols=True,
            ):
                item = base.build(int(idx), True)
                (
                    spec_sum,
                    spec_av,
                    spec_conf,
                    deep_sum,
                    s_rmse,
                    d_rmse,
                    s_available,
                    d_available,
                    _target_pixels,
                ) = compute_spec_deep_fallback(
                    base,
                    item,
                    deep_model,
                    deep_ck,
                    device,
                )
                sc[idx] = spec_sum
                sa[idx] = spec_av
                sf[idx] = spec_conf
                dc[idx] = deep_sum
                te[idx, 0] = s_rmse
                te[idx, 1] = d_rmse
                ea[idx, 0] = int(s_available)
                ea[idx, 1] = int(d_available)

        # Merge new Final Region errors from both GPUs.
        for rr in range(world):
            rp = out_path.with_name(
                out_path.stem + f".rank{rr}.partial.npz"
            )
            part = load_rank_fragment(rp)
            for idx, (err, avail) in part.items():
                te[int(idx), 2] = float(err)
                ea[int(idx), 2] = int(avail)
                seen[int(idx)] = True

        if not np.all(seen):
            raise RuntimeError(
                f"{split} region oracle merge missed "
                f"{int(np.sum(~seen))} samples."
            )

        stage1_masked = np.where(
            ea[:, :2] > 0, te[:, :2], 1e6
        )
        stage1_oracle = np.argmin(
            stage1_masked, axis=1
        ).astype(np.int64)
        best_stage1 = np.min(stage1_masked, axis=1)
        gate_target = (
            (ea[:, 2] > 0)
            & np.isfinite(te[:, 2])
            & (te[:, 2] < best_stage1)
        ).astype(np.int64)
        final_oracle = np.where(
            gate_target > 0, 2, stage1_oracle
        ).astype(np.int64)

        tmp = out_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            spectral_candidate=sc,
            spectral_available=sa,
            spectral_confidence=sf,
            deep_candidate=dc,
            true_error=te,
            expert_available=ea,
            stage1_oracle=stage1_oracle,
            gate_target=gate_target,
            final_oracle=final_oracle,
            signature=np.asarray(signature),
        )
        os.replace(tmp, out_path)

        print(
            f"Built {split} hierarchical cache:",
            out_path,
        )
        print(
            "  Final oracle counts [Spectral, Deep, Region] =",
            np.bincount(
                final_oracle, minlength=3
            ).tolist(),
        )
        print(
            "  Hard-case gate positives =",
            int(gate_target.sum()),
            "/",
            int(len(gate_target)),
        )

    if world > 1:
        dist.barrier()


class CachedDataset(Dataset):
    def __init__(self, base, cache, index_map=None):
        self.base = base
        self.cache = cache
        self.map = (
            np.arange(len(base), dtype=np.int64)
            if index_map is None
            else np.asarray(index_map, dtype=np.int64)
        )

    def __len__(self):
        return len(self.map)

    def __getitem__(self, i):
        idx = int(self.map[int(i)])
        item = self.base[idx]
        for k in (
            "spectral_candidate",
            "spectral_available",
            "spectral_confidence",
            "deep_candidate",
            "true_error",
            "expert_available",
        ):
            item[k] = torch.from_numpy(
                self.cache[k][idx].astype(
                    np.float32, copy=False
                )
            )
        item["stage1_oracle"] = torch.tensor(
            int(self.cache["stage1_oracle"][idx]),
            dtype=torch.long,
        )
        item["gate_target"] = torch.tensor(
            int(self.cache["gate_target"][idx]),
            dtype=torch.float32,
        )
        item["final_oracle"] = torch.tensor(
            int(self.cache["final_oracle"][idx]),
            dtype=torch.long,
        )
        return item


def balanced_map(final_oracle, seed):
    # Preserve the old router's successful strategy: balance the three FINAL
    # outcomes during training, while validation remains natural/unbalanced.
    rng = np.random.default_rng(seed)
    groups = [
        np.flatnonzero(final_oracle == k)
        for k in range(3)
    ]
    present = [g for g in groups if len(g)]
    target = max([len(g) for g in present] or [len(final_oracle)])
    pieces = []
    for g in groups:
        if not len(g):
            continue
        if len(g) < target:
            g = np.concatenate(
                [
                    g,
                    rng.choice(
                        g,
                        size=target - len(g),
                        replace=True,
                    ),
                ]
            )
        pieces.append(g)
    out = (
        np.concatenate(pieces)
        if pieces
        else np.arange(len(final_oracle))
    )
    rng.shuffle(out)
    return out.astype(np.int64)


def move_batch(batch, device):
    keys = (
        "router_input",
        "geometry",
        "component_band_fraction",
        "component_band_count",
        "component_features",
        "spectral_candidate",
        "spectral_available",
        "spectral_confidence",
        "deep_candidate",
        "target_band",
        "true_error",
        "expert_available",
        "stage1_oracle",
        "gate_target",
        "final_oracle",
    )
    return {
        k: batch[k].to(device, non_blocking=True)
        for k in keys
    }


def loss_fn(aux, batch, stage1_class_weights, gate_pos_weight):
    bands = batch["target_band"].long()
    rows = torch.arange(len(bands), device=bands.device)

    true = batch["true_error"].float()
    avail = batch["expert_available"] > 0.5

    pred_err = aux["predicted_candidate_error"][
        rows, bands, :
    ]
    stage1_logits = aux["stage1_logits"][
        rows, bands, :
    ]
    stage1_prob = aux["stage1_probabilities"][
        rows, bands, :
    ]
    gate_logit = aux["region_gate_logit"][
        rows, bands
    ]
    gate_prob = aux["region_gate_probability"][
        rows, bands
    ]

    # 1) Stage 1: Spectral vs Deep only.
    stage1_target = batch["stage1_oracle"].long()
    stage1_ce = F.cross_entropy(
        stage1_logits,
        stage1_target,
        weight=stage1_class_weights,
    )

    # 2) Hard-case gate: should Final Region override Stage 1?
    gate_target = batch["gate_target"].float()
    best_stage1_true = torch.min(
        true[:, 0].masked_fill(~avail[:, 0], 1e6),
        true[:, 1].masked_fill(~avail[:, 1], 1e6),
    )
    region_true = true[:, 2].masked_fill(
        ~avail[:, 2], 1e6
    )
    contrast = (
        torch.abs(best_stage1_true - region_true)
        / torch.maximum(
            best_stage1_true,
            region_true,
        ).clamp_min(1e-4)
    ).clamp(0, 1)
    gate_weight = 1.0 + 2.0 * contrast.detach()
    gate_bce_raw = F.binary_cross_entropy_with_logits(
        gate_logit,
        gate_target,
        pos_weight=gate_pos_weight,
        reduction="none",
    )
    gate_bce = (gate_bce_raw * gate_weight).mean()

    # 3) Expected-error calibration for all three experts.
    eps = float(
        getattr(config, "SPARSE_ROUTER_ERROR_FLOOR", 1e-4)
    )
    log_pred = torch.log(pred_err.clamp_min(eps))
    log_true = torch.log(true.detach().clamp_min(eps))
    unc = (
        F.smooth_l1_loss(
            log_pred,
            log_true,
            reduction="none",
            beta=0.4,
        )
        * avail
    ).sum() / avail.sum().clamp_min(1)

    # 4) Hierarchical expected regret.
    final_probs = torch.stack(
        [
            (1.0 - gate_prob) * stage1_prob[:, 0],
            (1.0 - gate_prob) * stage1_prob[:, 1],
            gate_prob,
        ],
        dim=1,
    )
    masked_true = true.masked_fill(~avail, 1e6)
    best = masked_true.min(dim=1).values.detach()
    regret = (
        final_probs
        * (masked_true.detach() - best[:, None]).clamp_min(0)
    ).sum(dim=1).mean()

    total = (
        0.80 * stage1_ce
        + 0.85 * gate_bce
        + 0.30 * unc
        + 0.70 * regret
    )
    return total


@torch.no_grad()
def collect_validation(
    model,
    loader,
    device,
    stage1_class_weights,
    gate_pos_weight,
):
    model.eval()
    total_loss = 0.0
    n = 0
    stage1_correct = 0

    gate_probs = []
    gate_targets = []
    stage1_routes = []
    true_errors = []
    expert_available = []

    for batch in loader:
        rb = move_batch(batch, device)
        aux = model(
            rb["router_input"],
            rb["spectral_candidate"],
            rb["spectral_available"],
            rb["spectral_confidence"],
            rb["geometry"],
            rb["deep_candidate"],
            rb["component_band_fraction"],
            rb["component_band_count"],
            rb["component_features"],
        )
        loss = loss_fn(
            aux,
            rb,
            stage1_class_weights,
            gate_pos_weight,
        )
        bs = len(rb["target_band"])
        total_loss += float(loss.item()) * bs
        n += bs

        rows = torch.arange(bs, device=device)
        bands = rb["target_band"].long()
        sroute = aux["stage1_hard_route_index"][
            rows, bands
        ]
        gp = aux["region_gate_probability"][
            rows, bands
        ]

        stage1_correct += int(
            (
                sroute
                == rb["stage1_oracle"].long()
            ).sum().item()
        )
        gate_probs.append(gp.float().cpu().numpy())
        gate_targets.append(
            rb["gate_target"].float().cpu().numpy()
        )
        stage1_routes.append(
            sroute.long().cpu().numpy()
        )
        true_errors.append(
            rb["true_error"].float().cpu().numpy()
        )
        expert_available.append(
            rb["expert_available"].float().cpu().numpy()
        )

    return {
        "loss": total_loss / max(n, 1),
        "stage1_accuracy": stage1_correct / max(n, 1),
        "gate_prob": np.concatenate(gate_probs),
        "gate_target": np.concatenate(gate_targets).astype(np.int64),
        "stage1_route": np.concatenate(stage1_routes).astype(np.int64),
        "true_error": np.concatenate(true_errors),
        "expert_available": np.concatenate(expert_available) > 0.5,
    }


def threshold_stats(val, threshold):
    gp = val["gate_prob"]
    gt = val["gate_target"]
    sroute = val["stage1_route"]
    err = np.asarray(val["true_error"], dtype=np.float64)
    avail = np.asarray(val["expert_available"], dtype=bool)

    route = sroute.copy()
    use_region = gp >= float(threshold)
    route[use_region] = 2

    masked = np.where(avail, err, 1e6)
    chosen = masked[np.arange(len(route)), route]
    best = masked.min(axis=1)
    regret = np.maximum(chosen - best, 0.0)

    pred_gate = use_region.astype(np.int64)
    tp = int(np.sum((pred_gate == 1) & (gt == 1)))
    fp = int(np.sum((pred_gate == 1) & (gt == 0)))
    fn = int(np.sum((pred_gate == 0) & (gt == 1)))
    tn = int(np.sum((pred_gate == 0) & (gt == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / max(len(gt), 1)

    objective = float(
        np.mean(chosen) + 0.20 * np.mean(regret)
    )
    return {
        "threshold": float(threshold),
        "objective": objective,
        "mean_selected_error": float(np.mean(chosen)),
        "mean_regret": float(np.mean(regret)),
        "gate_accuracy": float(accuracy),
        "gate_precision": float(precision),
        "gate_recall": float(recall),
        "region_fraction": float(np.mean(use_region)),
        "final_route_accuracy": float(
            np.mean(
                route
                == np.argmin(masked, axis=1)
            )
        ),
    }


def calibrate_threshold(val):
    candidates = np.linspace(0.05, 0.95, 37)
    stats = [
        threshold_stats(val, t)
        for t in candidates
    ]
    # Accuracy-first calibration: minimize actual validation reconstruction
    # error/regret. Ties prefer the higher threshold (more conservative Region
    # invocation).
    stats.sort(
        key=lambda s: (
            s["objective"],
            -s["threshold"],
        )
    )
    return stats[0]


def reduce_mean(value, device, distributed):
    t = torch.tensor(
        float(value), device=device, dtype=torch.float64
    )
    if distributed:
        dist.all_reduce(t)
        t /= dist.get_world_size()
    return float(t.item())


def init_from_old_router(model):
    old_path, _, _ = old_router_paths()
    if not old_path.exists():
        return None, 0, 0
    ck = torch_load(old_path, "cpu")
    if ck.get("architecture") != ComponentThreeExpertRouter.ARCHITECTURE_VERSION:
        return old_path, 0, len(model.state_dict())

    src = ck["model_state_dict"]
    dst = model.state_dict()
    compatible = {
        k: v
        for k, v in src.items()
        if k in dst and tuple(v.shape) == tuple(dst[k].shape)
    }
    missing, unexpected = model.load_state_dict(
        compatible, strict=False
    )
    return old_path, len(compatible), len(missing)


def main_worker(args):
    distributed, rank, world, device = setup_runtime()
    main = rank == 0
    try:
        # Frozen experts used only to build oracle labels.
        deep_model, deep_ck, deep_path = OLD.load_deep(device)
        (
            tf_model,
            tf_ck,
            v27_model,
            _v27_ck,
            tf_path,
            v27_path,
        ) = load_models(device)

        band_mean = np.asarray(
            deep_ck["band_mean"], np.float32
        )
        band_std = np.maximum(
            np.asarray(deep_ck["band_std"], np.float32),
            1e-6,
        )
        wavelengths = np.asarray(
            deep_ck["wavelengths_nm"], np.float32
        )
        n_bands = len(band_mean)

        train_cubes, val_cubes, w = load_global_training_data()
        w = np.asarray(w, np.float32)

        region_patch = int(tf_ck.get("patch_size", 65))
        train_base = OLD.ComponentRouterSamples(
            train_cubes,
            w,
            band_mean,
            band_std,
            TRAIN_SAMPLES,
            SEED + 101,
            deep_ck,
            region_patch,
        )
        val_base = OLD.ComponentRouterSamples(
            val_cubes,
            w,
            band_mean,
            band_std,
            VAL_SAMPLES,
            SEED + 202,
            deep_ck,
            region_patch,
        )

        out = (
            Path(config.MODELS_DIR)
            / "texture_restoration_router_rf_v2"
            / MODEL_SUBDIR
        )
        cache_dir = out / "oracle_cache"
        if main:
            cache_dir.mkdir(parents=True, exist_ok=True)
        if distributed:
            dist.barrier()

        old_router_path, old_train_cache, old_val_cache = old_router_paths()

        train_path = cache_dir / "train_hierarchical_oracle.npz"
        val_path = cache_dir / "val_hierarchical_oracle.npz"

        train_sig = cache_signature(
            "train",
            TRAIN_SAMPLES,
            old_train_cache,
            tf_path,
            v27_path,
        )
        val_sig = cache_signature(
            "val",
            VAL_SAMPLES,
            old_val_cache,
            tf_path,
            v27_path,
        )

        build_hierarchical_cache(
            "train",
            train_base,
            train_path,
            train_sig,
            old_train_cache,
            deep_model,
            deep_ck,
            tf_model,
            tf_ck,
            v27_model,
            device,
            rank,
            world,
            args.force_cache,
        )
        build_hierarchical_cache(
            "val",
            val_base,
            val_path,
            val_sig,
            old_val_cache,
            deep_model,
            deep_ck,
            tf_model,
            tf_ck,
            v27_model,
            device,
            rank,
            world,
            args.force_cache,
        )

        tc = load_cache(train_path)
        vc = load_cache(val_path)

        train_map = balanced_map(
            tc["final_oracle"], SEED + 303
        )
        train_ds = CachedDataset(
            train_base, tc, train_map
        )
        val_ds = CachedDataset(val_base, vc)

        sampler = (
            DistributedSampler(
                train_ds,
                num_replicas=world,
                rank=rank,
                shuffle=True,
                seed=SEED + 404,
            )
            if distributed
            else None
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
            persistent_workers=NUM_WORKERS > 0,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        # Stage-1 class weights from natural training cache.
        s_counts = np.bincount(
            tc["stage1_oracle"].astype(np.int64),
            minlength=2,
        ).astype(np.float32)
        s_weights = np.ones(2, np.float32)
        present = s_counts > 0
        if present.any():
            s_weights[present] = np.sqrt(
                s_counts[present].sum()
                / (present.sum() * s_counts[present])
            )
            s_weights = np.clip(
                s_weights, 0.65, 2.5
            )
        stage1_class_weights = torch.from_numpy(
            s_weights
        ).to(device)

        g = tc["gate_target"].astype(np.int64)
        pos = max(int(np.sum(g == 1)), 1)
        neg = max(int(np.sum(g == 0)), 1)
        gate_pos_weight = torch.tensor(
            float(np.clip(neg / pos, 0.7, 4.0)),
            device=device,
            dtype=torch.float32,
        )

        raw = HierarchicalComponentRouter(
            n_bands,
            w,
            d_model=int(
                getattr(config, "THREE_ROUTER_MODEL_DIM", 192)
            ),
            n_heads=int(
                getattr(config, "THREE_ROUTER_ATTENTION_HEADS", 8)
            ),
            n_layers=int(
                getattr(config, "THREE_ROUTER_TRANSFORMER_LAYERS", 6)
            ),
            ff_mult=int(
                getattr(config, "THREE_ROUTER_FF_MULTIPLIER", 4)
            ),
            dropout=float(
                getattr(config, "THREE_ROUTER_DROPOUT", 0.05)
            ),
        ).to(device)

        init_path, copied, missing_count = init_from_old_router(raw)

        ema = EMA(raw, EMA_DECAY)
        model = (
            DDP(
                raw,
                device_ids=[device.index],
                broadcast_buffers=False,
            )
            if distributed
            else raw
        )

        opt = torch.optim.AdamW(
            raw.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(device.type == "cuda"),
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max(int(args.epochs), 1),
            eta_min=LR * 0.08,
        )

        best = float("inf")
        best_epoch = 0
        patience = 0
        history = []

        if main:
            print("=" * 98)
            print("FINAL TPSSR HIERARCHICAL ROUTER")
            print("Stage 1: Spectral vs Deep-v3.1")
            print(
                "Stage 2: hard-case gate -> "
                "Frozen V27 Stage-A -> V13 TopologyFirst V1"
            )
            print(
                "No scenario/severity labels are router inputs."
            )
            print(
                "Initialized from old V31 router:",
                init_path,
            )
            print(
                f"Compatible tensors copied: {copied}; "
                f"new/missing tensors: {missing_count}"
            )
            print(
                "Training final-oracle counts:",
                np.bincount(
                    tc["final_oracle"].astype(np.int64),
                    minlength=3,
                ).tolist(),
            )
            print(
                "Validation final-oracle counts:",
                np.bincount(
                    vc["final_oracle"].astype(np.int64),
                    minlength=3,
                ).tolist(),
            )
            print(
                "Training hard cases:",
                int(tc["gate_target"].sum()),
                "/",
                len(tc["gate_target"]),
            )
            print("=" * 98)

        for epoch in range(
            1, int(args.epochs) + 1
        ):
            if sampler is not None:
                sampler.set_epoch(epoch)

            model.train()
            running = 0.0
            seen = 0
            bar = tqdm(
                train_loader,
                desc=(
                    f"GPU {rank} HierRouter "
                    f"{epoch:02d}/{int(args.epochs)}"
                ),
                disable=not main,
                dynamic_ncols=True,
            )
            for batch in bar:
                rb = move_batch(batch, device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    "cuda",
                    enabled=(device.type == "cuda"),
                ):
                    aux = model(
                        rb["router_input"],
                        rb["spectral_candidate"],
                        rb["spectral_available"],
                        rb["spectral_confidence"],
                        rb["geometry"],
                        rb["deep_candidate"],
                        rb["component_band_fraction"],
                        rb["component_band_count"],
                        rb["component_features"],
                    )
                    loss = loss_fn(
                        aux,
                        rb,
                        stage1_class_weights,
                        gate_pos_weight,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    raw.parameters(), GRAD_CLIP
                )
                scaler.step(opt)
                scaler.update()
                ema.update(raw)

                bs = len(rb["target_band"])
                running += float(loss.item()) * bs
                seen += bs
                if main:
                    bar.set_postfix(
                        loss=f"{running / max(seen, 1):.4f}"
                    )

            sched.step()

            train_loss = reduce_mean(
                running / max(seen, 1),
                device,
                distributed,
            )

            # Every rank evaluates the same validation set; average scalar
            # metrics are therefore already equivalent.  Threshold calibration
            # is deterministic and uses validation only.
            val = collect_validation(
                ema.model,
                val_loader,
                device,
                stage1_class_weights,
                gate_pos_weight,
            )
            best_thr = calibrate_threshold(val)
            objective = float(best_thr["objective"])

            if main:
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": float(val["loss"]),
                    "stage1_accuracy": float(
                        val["stage1_accuracy"]
                    ),
                    **{
                        f"val_{k}": v
                        for k, v in best_thr.items()
                    },
                    "lr": float(opt.param_groups[0]["lr"]),
                }
                history.append(row)

                improved = objective < best
                if improved:
                    best = objective
                    best_epoch = epoch
                    patience = 0
                    payload = {
                        "architecture": (
                            HierarchicalComponentRouter.ARCHITECTURE_VERSION
                        ),
                        "model_state_dict": (
                            ema.model.state_dict()
                        ),
                        "epoch": epoch,
                        "val_objective": objective,
                        "val_stats": {
                            "stage1_accuracy": float(
                                val["stage1_accuracy"]
                            ),
                            **best_thr,
                        },
                        "gate_threshold": float(
                            best_thr["threshold"]
                        ),
                        "n_bands": n_bands,
                        "band_mean": band_mean,
                        "band_std": band_std,
                        "wavelengths_nm": w,
                        "candidate_names": (
                            HierarchicalComponentRouter.CANDIDATE_NAMES
                        ),
                        "initial_router_checkpoint": (
                            str(init_path)
                            if init_path is not None
                            else None
                        ),
                        "deep_v31_checkpoint": str(deep_path),
                        "topology_checkpoint": str(tf_path),
                        "v27_stagea_checkpoint": str(v27_path),
                        "settings": {
                            "d_model": int(
                                getattr(
                                    config,
                                    "THREE_ROUTER_MODEL_DIM",
                                    192,
                                )
                            ),
                            "n_heads": int(
                                getattr(
                                    config,
                                    "THREE_ROUTER_ATTENTION_HEADS",
                                    8,
                                )
                            ),
                            "n_layers": int(
                                getattr(
                                    config,
                                    "THREE_ROUTER_TRANSFORMER_LAYERS",
                                    6,
                                )
                            ),
                            "ff_mult": int(
                                getattr(
                                    config,
                                    "THREE_ROUTER_FF_MULTIPLIER",
                                    4,
                                )
                            ),
                            "dropout": float(
                                getattr(
                                    config,
                                    "THREE_ROUTER_DROPOUT",
                                    0.05,
                                )
                            ),
                            "router_patch_size": int(
                                config.SPARSE_ROUTER_PATCH_SIZE
                            ),
                            "routing": (
                                "stage1_spectral_vs_deep_then_"
                                "binary_region_override"
                            ),
                        },
                    }
                    torch.save(
                        payload, out / BEST_MODEL
                    )
                else:
                    patience += 1

                with (
                    out / "training_history.csv"
                ).open(
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as f:
                    wr = csv.DictWriter(
                        f,
                        fieldnames=list(
                            history[0].keys()
                        ),
                    )
                    wr.writeheader()
                    wr.writerows(history)

                summary = {
                    "best_epoch": best_epoch,
                    "best_val_objective": best,
                    "latest": row,
                    "train_final_oracle_counts": (
                        np.bincount(
                            tc["final_oracle"].astype(
                                np.int64
                            ),
                            minlength=3,
                        ).tolist()
                    ),
                    "val_final_oracle_counts": (
                        np.bincount(
                            vc["final_oracle"].astype(
                                np.int64
                            ),
                            minlength=3,
                        ).tolist()
                    ),
                    "train_gate_positive": int(
                        tc["gate_target"].sum()
                    ),
                    "val_gate_positive": int(
                        vc["gate_target"].sum()
                    ),
                }
                (
                    out / "training_summary.json"
                ).write_text(
                    json.dumps(
                        summary,
                        indent=2,
                        allow_nan=True,
                    ),
                    encoding="utf-8",
                )

                print(
                    f"Epoch {epoch:02d}: "
                    f"obj={objective:.5f} "
                    f"stage1_acc={val['stage1_accuracy']:.3f} "
                    f"gate_acc={best_thr['gate_accuracy']:.3f} "
                    f"P/R={best_thr['gate_precision']:.3f}/"
                    f"{best_thr['gate_recall']:.3f} "
                    f"thr={best_thr['threshold']:.2f} "
                    f"final_acc={best_thr['final_route_accuracy']:.3f} "
                    f"region={best_thr['region_fraction']:.3f} "
                    f"regret={best_thr['mean_regret']:.4f}"
                    + (" NEW BEST" if improved else "")
                )
                stop = (
                    epoch >= MIN_EPOCHS
                    and patience >= PATIENCE
                )
            else:
                stop = False

            if distributed:
                t = torch.tensor(
                    int(stop), device=device
                )
                dist.broadcast(t, 0)
                stop = bool(t.item())
            if stop:
                break

        if main:
            print("")
            print("Best hierarchical router:")
            print(out / BEST_MODEL)
            print("Best epoch:", best_epoch)
            print("Best validation objective:", best)
            print(
                "Next step: run the 200-mask hierarchical router audit "
                "before integrating it into Final TPSSR."
            )

    finally:
        cleanup(distributed)


if __name__ == "__main__":
    args = parse_args()
    if launch_if_needed(args):
        raise SystemExit(0)
    main_worker(args)
