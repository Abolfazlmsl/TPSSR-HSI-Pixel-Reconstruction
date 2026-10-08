#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train TPSSR New V13 with frozen Original-V27 Stage-A guidance.

The training is intentionally a SINGLE stage. Original V27 is loaded and frozen;
only the new filler is optimized. The filler explicitly separates radiometry/
low-frequency reconstruction from native-resolution high-frequency texture.

Use the project-root launcher instead of calling this file directly:
    python run_new_v13.py --dataset SWIR1 --gpus 1 --no-resume
or
    python run_new_v13.py --dataset SWIR1 --gpus 2 --no-resume
"""

from __future__ import annotations

from pathlib import Path
import copy
import csv
import importlib.util
import os
import random
import sys
from datetime import timedelta

HERE = Path(__file__).resolve()
CORE = next(p for p in HERE.parents if (p / "config.py").exists())
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import config
from src.evaluation_utils import load_global_training_data
from src.new_v13_structure_guided_filler import NewV13StructureGuidedFiller

METHOD_DIR = CORE / "reconstruction_methods" / "texture_restoration_router_rf_v2"
METHOD = "texture_restoration_router_rf_v2"


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


V27 = import_file("new_v13_v27_original_arch", METHOD_DIR / "train_swir1_v27_learned_correspondence.py")

# Quick-development defaults. All can be overridden by run_new_v13.py.
EPOCHS = int(os.environ.get("NEWV13_EPOCHS", "12"))
MIN_EPOCHS = int(os.environ.get("NEWV13_MIN_EPOCHS", "6"))
PATIENCE = int(os.environ.get("NEWV13_PATIENCE", "4"))
TRAIN_SAMPLES = int(os.environ.get("NEWV13_TRAIN_SAMPLES", "1800"))
VAL_SAMPLES = int(os.environ.get("NEWV13_VAL_SAMPLES", "450"))
BATCH_SIZE = int(os.environ.get("NEWV13_BATCH_SIZE", "1"))
NUM_WORKERS = int(os.environ.get("NEWV13_NUM_WORKERS", "2"))
LEARNING_RATE = float(os.environ.get("NEWV13_LR", "1.5e-4"))
WEIGHT_DECAY = float(os.environ.get("NEWV13_WEIGHT_DECAY", "1.0e-5"))
GRAD_CLIP = float(os.environ.get("NEWV13_GRAD_CLIP", "2.0"))
EMA_DECAY = float(os.environ.get("NEWV13_EMA_DECAY", "0.995"))
NO_RESUME = os.environ.get("NEWV13_NO_RESUME", "0") == "1"

# Extra losses dedicated to LF/HF separation and radiometry preservation.
W_LOW = 0.40
W_HIGH = 0.50
W_HIGH_MAG = 0.22
W_LOCAL_MEAN = 0.28
W_LOCAL_STD = 0.18
W_AMP = 0.015

SEED = int(getattr(config, "SEED", 1234)) + 131300


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.model = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        src = model.module if isinstance(model, DDP) else model
        src_state = src.state_dict()
        ema_state = self.model.state_dict()
        for key, value in ema_state.items():
            source = src_state[key].detach()
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                value.copy_(source)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_runtime():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for New V13 training.")

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                device_id=device,
                timeout=timedelta(minutes=30),
            )
        except TypeError:
            dist.init_process_group(
                backend="nccl", init_method="env://", timeout=timedelta(minutes=30)
            )
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    return distributed, rank, local_rank, world_size, device


def cleanup(distributed: bool):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def v27_checkpoint_path() -> Path:
    tag = str(config.TEST_DATASET_FOLDER).lower()
    root = Path(config.MODELS_DIR) / METHOD
    candidates = [
        root / f"region_v27_original_{tag}_dev_2gpu" / "best_model.pt",
        root / f"region_v27_learned_correspondence_{tag}_dev_2gpu" / "best_model.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Original V27 checkpoint not found. Expected one of:\n" + "\n".join(str(p) for p in candidates)
    )


def build_frozen_structure(v27_ck: dict, device: torch.device):
    settings = v27_ck["settings"]
    structure = V27.StructureCompletionNet(
        n_bands=int(v27_ck["n_bands"]),
        spectral_width=int(settings["spectral_width"]),
        base_width=int(settings["base_width"]),
        attention_heads=int(settings["attention_heads"]),
    ).to(device)

    prefix = "structure_net."
    state = {
        key[len(prefix):]: value
        for key, value in v27_ck["model_state_dict"].items()
        if key.startswith(prefix)
    }
    if not state:
        raise RuntimeError("V27 checkpoint does not contain structure_net.* weights")
    structure.load_state_dict(state, strict=True)
    structure.eval()
    for parameter in structure.parameters():
        parameter.requires_grad_(False)
    return structure, int(settings["base_width"])


def structure_forward(structure, batch):
    return structure(
        corrupted_cube=batch["cube"],
        observed_target=batch["observed_target"],
        mask=batch["mask"],
        spatial_valid=batch["spatial_valid"],
        valid_fraction=batch["valid_fraction"],
        distance=batch["distance"],
        band_position=batch["band_position"],
    )


def filler_forward(filler, batch, structure_output):
    return filler(
        corrupted_cube=batch["cube"],
        observed_target=batch["observed_target"],
        mask=batch["mask"],
        spatial_valid=batch["spatial_valid"],
        valid_fraction=batch["valid_fraction"],
        distance=batch["distance"],
        band_position=batch["band_position"],
        structure=structure_output,
    )


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def local_stats(x: torch.Tensor, kernel: int):
    pad = kernel // 2
    mean = F.avg_pool2d(x, kernel, stride=1, padding=pad)
    second = F.avg_pool2d(x.square(), kernel, stride=1, padding=pad)
    std = torch.sqrt((second - mean.square()).clamp_min(0.0) + 1e-6)
    return mean, std


def new_v13_loss(output: dict, batch: dict):
    # Reuse the proven V27 structural/radiometric terms, but prevent V27's
    # "coarse" term from forcing our LF branch to contain high-frequency detail.
    base_output = dict(output)
    base_output["coarse_prediction"] = output["prediction"]
    base_loss, base_parts = V27.reconstruction_loss(base_output, batch)

    target = batch["target"]
    mask = batch["mask"]
    target_low = F.avg_pool2d(target, 11, stride=1, padding=5)
    target_high = target - target_low

    low = masked_mean(
        F.smooth_l1_loss(output["low_prediction"], target_low, reduction="none", beta=0.30),
        mask,
    )
    high = masked_mean(
        F.smooth_l1_loss(output["high_frequency"], target_high, reduction="none", beta=0.20),
        mask,
    )
    high_mag = masked_mean(
        (output["high_frequency"].abs() - target_high.abs()).abs(),
        mask,
    )

    pred_mean7, pred_std7 = local_stats(output["prediction"], 7)
    targ_mean7, targ_std7 = local_stats(target, 7)
    pred_mean15, pred_std15 = local_stats(output["prediction"], 15)
    targ_mean15, targ_std15 = local_stats(target, 15)
    local_mean = 0.5 * (
        masked_mean((pred_mean7 - targ_mean7).abs(), mask)
        + masked_mean((pred_mean15 - targ_mean15).abs(), mask)
    )
    local_std = 0.5 * (
        masked_mean((pred_std7 - targ_std7).abs(), mask)
        + masked_mean((pred_std15 - targ_std15).abs(), mask)
    )

    # Small regularizer only prevents the internal texture amplitude from
    # saturating everywhere. It does not choose between alternative outputs.
    amp = output["texture_amplitude"]
    amp_reg = ((amp - 0.5).square() * mask).sum() / mask.sum().clamp_min(1.0)

    total = (
        base_loss
        + W_LOW * low
        + W_HIGH * high
        + W_HIGH_MAG * high_mag
        + W_LOCAL_MEAN * local_mean
        + W_LOCAL_STD * local_std
        + W_AMP * amp_reg
    )

    parts = {
        **base_parts,
        "low": low.detach(),
        "high": high.detach(),
        "high_mag": high_mag.detach(),
        "local_mean": local_mean.detach(),
        "local_std": local_std.detach(),
        "amp_reg": amp_reg.detach(),
    }
    return total, parts


def move_batch(batch: dict, device: torch.device):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


@torch.no_grad()
def evaluate(filler, structure, loader, device):
    filler.eval()
    structure.eval()
    total = 0.0
    count = 0
    sums = {"pixel": 0.0, "ssim": 0.0, "low": 0.0, "high": 0.0, "local_mean": 0.0, "local_std": 0.0}
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", enabled=True):
            s = structure_forward(structure, batch)
            output = filler_forward(filler, batch, s)
            loss, parts = new_v13_loss(output, batch)
        total += float(loss.item())
        count += 1
        for key in sums:
            sums[key] += float(parts[key].item())
    result = {"loss": total / max(count, 1)}
    result.update({key: value / max(count, 1) for key, value in sums.items()})
    return result


def save_checkpoint(
    path: Path,
    raw_filler,
    ema_filler,
    optimizer,
    scheduler,
    epoch: int,
    best_val: float,
    v27_ck: dict,
    v27_path: Path,
    history: list[dict],
):
    payload = {
        "architecture": NewV13StructureGuidedFiller.ARCHITECTURE_VERSION,
        "epoch": int(epoch),
        "best_val": float(best_val),
        "patch_size": int(v27_ck.get("patch_size", 65)),
        "n_bands": int(v27_ck["n_bands"]),
        "band_mean": np.asarray(v27_ck["band_mean"], dtype=np.float32),
        "band_std": np.asarray(v27_ck["band_std"], dtype=np.float32),
        "wavelengths_nm": np.asarray(v27_ck["wavelengths_nm"], dtype=np.float32),
        "source_v27_checkpoint": str(v27_path),
        "model_state_dict": ema_filler.state_dict(),
        "train_model_state_dict": raw_filler.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "settings": {
            "structure_feature_channels": int(raw_filler.structure_feature_channels),
            "spectral_width": int(raw_filler.spectral_width),
            "base_width": int(raw_filler.base_width),
            "mid_width": int(raw_filler.mid_width),
            "bottleneck_width": int(raw_filler.bottleneck_width),
            "hf_residual_scale": float(raw_filler.hf_residual_scale),
            "low_delta_scale": float(raw_filler.low_delta_scale),
            "epochs": int(EPOCHS),
            "train_samples": int(TRAIN_SAMPLES),
            "val_samples": int(VAL_SAMPLES),
            "v27_frozen": True,
            "native_resolution_texture_path": True,
            "lf_hf_decomposition": True,
        },
    }
    torch.save(payload, path)


def main():
    distributed, rank, local_rank, world_size, device = setup_runtime()
    main_process = rank == 0
    try:
        set_seed(SEED + rank)
        train_cubes, val_cubes, wavelengths = load_global_training_data()

        source_v27 = v27_checkpoint_path()
        v27_ck = torch_load(source_v27, "cpu")
        if int(v27_ck["n_bands"]) != int(train_cubes[0].shape[-1]):
            raise RuntimeError("V27 spectral dimension differs from current training fold")
        if not np.allclose(np.asarray(v27_ck["wavelengths_nm"]), np.asarray(wavelengths), atol=1e-3, rtol=0.0):
            raise RuntimeError("V27 wavelength grid differs from current training fold")

        structure, structure_feature_channels = build_frozen_structure(v27_ck, device)

        band_mean = np.asarray(v27_ck["band_mean"], dtype=np.float32)
        band_std = np.asarray(v27_ck["band_std"], dtype=np.float32)
        train_ds = V27.V27LearnedCorrespondenceDataset(
            train_cubes, band_mean, band_std, TRAIN_SAMPLES, SEED + 101, True
        )
        val_ds = V27.V27LearnedCorrespondenceDataset(
            val_cubes, band_mean, band_std, VAL_SAMPLES, SEED + 202, False
        ) if main_process else None

        sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED
        ) if distributed else None
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
        ) if main_process else None

        base_filler = NewV13StructureGuidedFiller(
            n_bands=int(v27_ck["n_bands"]),
            structure_feature_channels=structure_feature_channels,
        ).to(device)
        ema = EMA(base_filler, EMA_DECAY)
        filler = DDP(base_filler, device_ids=[local_rank], broadcast_buffers=False) if distributed else base_filler
        optimizer = torch.optim.AdamW(
            filler.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(EPOCHS, 1), eta_min=LEARNING_RATE * 0.08
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        tag = str(config.TEST_DATASET_FOLDER).lower()
        outdir = Path(config.MODELS_DIR) / METHOD / f"new_v13_v27_guided_{tag}"
        if main_process:
            outdir.mkdir(parents=True, exist_ok=True)
        if distributed:
            dist.barrier()

        last_path = outdir / "last_model.pt"
        best_path = outdir / "best_model.pt"
        history_path = outdir / "training_history.csv"

        start_epoch = 1
        best_val = float("inf")
        no_improve = 0
        history: list[dict] = []

        if NO_RESUME and main_process:
            for path in (last_path, best_path, history_path):
                if path.exists():
                    path.unlink()
        if distributed:
            dist.barrier()

        if (not NO_RESUME) and last_path.exists():
            resume = torch_load(last_path, "cpu")
            base_filler.load_state_dict(resume.get("train_model_state_dict", resume["model_state_dict"]), strict=True)
            ema.model.load_state_dict(resume["model_state_dict"], strict=True)
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            scheduler.load_state_dict(resume["scheduler_state_dict"])
            start_epoch = int(resume["epoch"]) + 1
            best_val = float(resume.get("best_val", float("inf")))
            history = list(resume.get("history", []))
            if main_process:
                print(f"Resuming New V13 from epoch {start_epoch}")

        if main_process:
            trainable = sum(p.numel() for p in base_filler.parameters() if p.requires_grad)
            frozen = sum(p.numel() for p in structure.parameters())
            print("NEW V13 — V27-guided texture-preserving filler")
            print("Source V27:", source_v27)
            print(f"Frozen V27 Stage-A: {frozen/1e6:.3f} M parameters")
            print(f"Trainable New V13: {trainable/1e6:.3f} M parameters")
            print("Single training stage; V27 remains frozen.")
            print("Texture path: native resolution, no pooling/downsampling/interpolation.")
            print(f"Epochs={EPOCHS} train_samples={TRAIN_SAMPLES} val_samples={VAL_SAMPLES} world_size={world_size}")

        for epoch in range(start_epoch, EPOCHS + 1):
            train_ds.set_epoch(epoch)
            if sampler is not None:
                sampler.set_epoch(epoch)
            filler.train()
            structure.eval()
            running = 0.0
            seen = 0.0
            progress = tqdm(
                train_loader,
                desc=f"New V13 epoch {epoch}/{EPOCHS}",
                disable=not main_process,
                dynamic_ncols=True,
            )
            last_parts = None
            for batch in progress:
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                    structure_output = structure_forward(structure, batch)
                # Explicit detach guarantees no accidental gradient enters V27.
                structure_output = {key: value.detach() for key, value in structure_output.items()}
                with torch.amp.autocast("cuda", enabled=True):
                    output = filler_forward(filler, batch, structure_output)
                    loss, parts = new_v13_loss(output, batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(filler.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                ema.update(filler)

                running += float(loss.item())
                seen += 1.0
                last_parts = parts
                if main_process:
                    progress.set_postfix(
                        loss=f"{running/max(seen,1):.4f}",
                        pix=f"{parts['pixel'].item():.3f}",
                        ssim=f"{parts['ssim'].item():.3f}",
                        lf=f"{parts['low'].item():.3f}",
                        hf=f"{parts['high'].item():.3f}",
                    )

            if distributed:
                packed = torch.tensor([running, seen], dtype=torch.float64, device=device)
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                train_loss = float(packed[0].item() / max(packed[1].item(), 1.0))
                dist.barrier()
            else:
                train_loss = running / max(seen, 1.0)

            epoch_lr = float(optimizer.param_groups[0]["lr"])
            scheduler.step()

            stop = False
            if main_process:
                val = evaluate(ema.model.to(device), structure, val_loader, device)
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val["loss"],
                    "val_pixel": val["pixel"],
                    "val_ssim_loss": val["ssim"],
                    "val_low": val["low"],
                    "val_high": val["high"],
                    "val_local_mean": val["local_mean"],
                    "val_local_std": val["local_std"],
                    "lr": epoch_lr,
                }
                history.append(row)

                improved = val["loss"] < best_val - 1e-6
                if improved:
                    best_val = val["loss"]
                    no_improve = 0
                else:
                    no_improve += 1

                raw = filler.module if isinstance(filler, DDP) else filler
                save_checkpoint(
                    last_path, raw, ema.model, optimizer, scheduler, epoch, best_val,
                    v27_ck, source_v27, history,
                )
                if improved:
                    save_checkpoint(
                        best_path, raw, ema.model, optimizer, scheduler, epoch, best_val,
                        v27_ck, source_v27, history,
                    )

                with history_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                    writer.writeheader()
                    writer.writerows(history)

                print(
                    f"Epoch {epoch}: train={train_loss:.6f} val={val['loss']:.6f} "
                    f"pixel={val['pixel']:.4f} ssimLoss={val['ssim']:.4f} "
                    f"LF={val['low']:.4f} HF={val['high']:.4f}"
                )
                stop = epoch >= MIN_EPOCHS and no_improve >= PATIENCE

            flag = torch.tensor([int(stop)], dtype=torch.int32, device=device)
            if distributed:
                dist.broadcast(flag, src=0)
            if bool(flag.item()):
                if main_process:
                    print(f"Early stopping at epoch {epoch}")
                break

        if main_process:
            print("New V13 training complete:", best_path)
    finally:
        cleanup(distributed)


if __name__ == "__main__":
    main()
