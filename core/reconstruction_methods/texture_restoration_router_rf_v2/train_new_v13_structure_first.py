#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train TPSSR V13 Structure-First from scratch.

The frozen Original-V27 Stage-A supplies guidance only. V13 first predicts
native-resolution geometry, then radiometry, then texture/detail. No previous
V13 checkpoint is warm-started.
"""
from __future__ import annotations

from pathlib import Path
import copy
import csv
import importlib.util
import os
import sys

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
from src.new_v13_structure_first_filler import StructureFirstV13Filler

METHOD_DIR = CORE / "reconstruction_methods" / "texture_restoration_router_rf_v2"
METHOD = "texture_restoration_router_rf_v2"


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


BASE = import_file("structure_first_base", METHOD_DIR / "train_new_v13_structure_guided.py")
V27 = BASE.V27

EPOCHS = int(os.environ.get("V13SF_EPOCHS", "20"))
MIN_EPOCHS = int(os.environ.get("V13SF_MIN_EPOCHS", "12"))
PATIENCE = int(os.environ.get("V13SF_PATIENCE", "5"))
TRAIN_SAMPLES = int(os.environ.get("V13SF_TRAIN_SAMPLES", "3600"))
VAL_SAMPLES = int(os.environ.get("V13SF_VAL_SAMPLES", "800"))
BATCH_SIZE = int(os.environ.get("V13SF_BATCH_SIZE", "1"))
NUM_WORKERS = int(os.environ.get("V13SF_NUM_WORKERS", "2"))
LEARNING_RATE = float(os.environ.get("V13SF_LR", "1.0e-4"))
WEIGHT_DECAY = float(os.environ.get("V13SF_WEIGHT_DECAY", "1.0e-5"))
GRAD_CLIP = float(os.environ.get("V13SF_GRAD_CLIP", "2.0"))
EMA_DECAY = float(os.environ.get("V13SF_EMA_DECAY", "0.995"))
NO_RESUME = os.environ.get("V13SF_NO_RESUME", "0") == "1"

# Geometry-first losses. These intentionally put more pressure on the explicit
# structure representation than the prior E2E/Corr variants.
W_PIXEL = 0.70
W_SUPPORT_BCE = 0.42
W_SUPPORT_DICE = 0.38
W_CORE_BCE = 0.52
W_CORE_DICE = 0.58
W_DARK = 0.38
W_BRIGHT = 0.38
W_TANGENT = 0.34
W_THICKNESS = 0.18
W_CONTRAST = 0.54
W_RADIOMETRY = 0.48
W_CENTER_VALUE = 0.80
W_FINAL_POLARITY = 0.72
W_TRANSPORT = 0.18
W_DONOR_RANK = 0.08
W_HF = 0.28
W_GUIDE_ORIENT = 0.16

SEED = int(getattr(config, "SEED", 1234)) + 131410


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.model = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        src = model.module if isinstance(model, DDP) else model
        ss = src.state_dict(); es = self.model.state_dict()
        for k, v in es.items():
            s = ss[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(s, alpha=1.0 - self.decay)
            else:
                v.copy_(s)


def masked_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    ww = w.to(x.dtype)
    return (x * ww).sum() / ww.sum().clamp_min(1.0)


def local_stats(x: torch.Tensor, kernel: int = 11):
    p = kernel // 2
    mean = F.avg_pool2d(x, kernel, 1, p)
    second = F.avg_pool2d(x.square(), kernel, 1, p)
    std = torch.sqrt((second - mean.square()).clamp_min(0.0) + 1e-6)
    return mean, std


def inner_mask(mask: torch.Tensor) -> torch.Tensor:
    eroded = 1.0 - F.max_pool2d(1.0 - mask, 3, 1, 1)
    return (eroded * mask).clamp(0.0, 1.0)


def high_pass(x: torch.Tensor, k: int) -> torch.Tensor:
    return x - F.avg_pool2d(x, k, 1, k // 2)


def structure_targets(batch: dict) -> dict[str, torch.Tensor]:
    target = batch["target"]
    mean, std = local_stats(target, 11)
    contrast = batch.get("contrast_target", target - mean)
    denom = 0.16 * std + 0.035
    dark = torch.sigmoid(((-contrast) - 0.52 * std) / denom)
    bright = torch.sigmoid((contrast - 0.52 * std) / denom)
    edge = batch.get("edge_target")
    if edge is None:
        gx, gy = V27.sobel_xy(target)
        mag = torch.sqrt(gx.square() + gy.square() + 1e-8)
        lm = F.avg_pool2d(mag, 9, 1, 4)
        edge = torch.clamp(mag / (2.2 * lm + 0.04), 0.0, 1.0)
    else:
        edge = edge.clamp(0.0, 1.0)

    polarity = torch.maximum(dark, bright)
    support = torch.maximum(edge, 0.88 * polarity).clamp(0.0, 1.0)

    # Native ridge/core target: local maxima of structure/polarity are the
    # explicit one/few-pixel skeleton the decoder must preserve through holes.
    score = torch.maximum(edge, polarity).clamp(0.0, 1.0)
    mx = F.max_pool2d(score, 3, 1, 1)
    ridge = score * torch.exp(-((mx - score).clamp_min(0.0) / 0.12).square())
    core = (ridge * (0.35 + 0.65 * polarity)).clamp(0.0, 1.0)

    hard_support = (support.detach() > 0.45).to(target.dtype)
    thickness = F.avg_pool2d(hard_support, 5, 1, 2).clamp(0.0, 1.0)

    orient = batch.get("orientation_target")
    if orient is None:
        gx, gy = V27.sobel_xy(target)
        den = gx.square() + gy.square() + 1e-6
        grad_c2 = (gx.square() - gy.square()) / den
        grad_s2 = (2.0 * gx * gy) / den
        orient = torch.cat([grad_c2, grad_s2], 1)
    # Dataset/V27 orientation is gradient-normal orientation. The line tangent
    # is the negated doubled-angle vector.
    tangent = -F.normalize(orient, dim=1, eps=1e-6)

    return {
        "mean": mean, "std": std, "contrast": contrast,
        "dark": dark, "bright": bright, "polarity": polarity,
        "edge": edge, "support": support, "core": core,
        "thickness": thickness, "tangent": tangent,
    }


def weighted_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return masked_mean(raw * (1.0 + 1.75 * target.detach()), weight)


def soft_dice(prob: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    w = weight.to(prob.dtype)
    p = prob * w; t = target * w
    inter = (p * t).sum()
    den = p.square().sum() + t.square().sum() + 1e-6
    return 1.0 - (2.0 * inter + 1e-6) / den


def final_polarity_loss(pred: torch.Tensor, t: dict, mask: torch.Tensor) -> torch.Tensor:
    # Directly punish washing a real dark/bright core into a gray average.
    # PyTorch deliberately disallows probability-space BCE inside autocast.
    # This loss is defined from reconstructed intensities (not logits), so run
    # the numerically-sensitive probability calculation explicitly in FP32.
    with torch.autocast(device_type=pred.device.type, enabled=False):
        pred_f = pred.float()
        mean_f = t["mean"].float()
        std_f = t["std"].float()
        dark_t = t["dark"].float()
        bright_t = t["bright"].float()
        core_f = t["core"].float()
        polarity_f = t["polarity"].float()
        mask_f = mask.float()

        pred_contrast = pred_f - mean_f
        denom = 0.16 * std_f + 0.035
        pred_dark = torch.sigmoid(((-pred_contrast) - 0.52 * std_f) / denom)
        pred_bright = torch.sigmoid((pred_contrast - 0.52 * std_f) / denom)
        inner = inner_mask(mask_f).float()
        w = mask_f * (0.15 + 2.8 * core_f + 3.2 * inner * polarity_f)
        ld = F.binary_cross_entropy(pred_dark.clamp(1e-5, 1.0 - 1e-5), dark_t, reduction="none")
        lb = F.binary_cross_entropy(pred_bright.clamp(1e-5, 1.0 - 1e-5), bright_t, reduction="none")
        return 0.5 * (masked_mean(ld, w) + masked_mean(lb, w))


def structure_first_loss(output: dict, batch: dict, v27_structure: dict):
    base_out = {"prediction": output["prediction"], "coarse_prediction": output["prediction"]}
    base_loss, base_parts = V27.reconstruction_loss(base_out, batch)

    target = batch["target"]; mask = batch["mask"]
    t = structure_targets(batch)
    inner = inner_mask(mask)

    pix_w = mask * (1.0 + 1.5 * t["support"] + 2.6 * t["core"] + 2.8 * inner * t["polarity"])
    pixel = masked_mean(F.smooth_l1_loss(output["prediction"], target, reduction="none", beta=0.10), pix_w)

    support_bce = weighted_bce(output["support_logit"], t["support"], mask)
    support_dice = soft_dice(output["support_prob"], t["support"], mask)
    core_bce = weighted_bce(output["core_logit"], t["core"], mask)
    core_dice = soft_dice(output["core_prob"], t["core"], mask)
    dark_loss = weighted_bce(output["dark_logit"], t["dark"], mask)
    bright_loss = weighted_bce(output["bright_logit"], t["bright"], mask)

    dot = (output["tangent"] * t["tangent"]).sum(1, keepdim=True).clamp(-1.0, 1.0)
    tangent = masked_mean(1.0 - dot, mask * (0.15 + 1.85 * t["support"]))
    thickness = masked_mean((output["thickness"] - t["thickness"]).abs(), mask * (0.25 + t["support"]))

    contrast = masked_mean(
        F.smooth_l1_loss(output["signed_contrast"], t["contrast"], reduction="none", beta=0.08),
        mask * (0.20 + 2.2 * t["support"] + 2.5 * inner * t["polarity"]),
    )
    radiometry = masked_mean(
        F.smooth_l1_loss(output["radiometry"], target, reduction="none", beta=0.10),
        mask * (0.25 + 1.6 * t["support"] + 2.6 * inner * t["polarity"]),
    )
    center_value = masked_mean(
        F.smooth_l1_loss(output["prediction"], target, reduction="none", beta=0.07),
        inner * (0.10 + 3.5 * t["polarity"] + 2.0 * t["core"]),
    )
    polarity_final = final_polarity_loss(output["prediction"], t, mask)

    transport = masked_mean(
        F.smooth_l1_loss(output["transported"], target, reduction="none", beta=0.10),
        mask * (0.10 + 0.9 * t["support"] + 1.2 * t["core"]),
    )
    donor_rank = output["donor_rank_loss"]

    grad_mask = F.max_pool2d(mask, 3, 1, 1)
    hf = 0.5 * (
        masked_mean((high_pass(output["prediction"], 3) - high_pass(target, 3)).abs(), grad_mask)
        + masked_mean((high_pass(output["prediction"], 7) - high_pass(target, 7)).abs(), grad_mask)
    )

    # Keep final image directionally consistent with frozen V27 guidance, but
    # the explicit structure decoder is supervised from GT independently.
    gx, gy = V27.sobel_xy(output["prediction"])
    den = gx.square() + gy.square() + 1e-6
    cos2 = (gx.square() - gy.square()) / den
    sin2 = (2.0 * gx * gy) / den
    guide = v27_structure["orientation"].detach()
    guide_dot = (cos2 * guide[:, 0:1] + sin2 * guide[:, 1:2]).clamp(-1.0, 1.0)
    guide_orient = masked_mean(1.0 - guide_dot, mask * (0.20 + v27_structure["edge_prob"].detach()))

    total = (
        base_loss
        + W_PIXEL * pixel
        + W_SUPPORT_BCE * support_bce + W_SUPPORT_DICE * support_dice
        + W_CORE_BCE * core_bce + W_CORE_DICE * core_dice
        + W_DARK * dark_loss + W_BRIGHT * bright_loss
        + W_TANGENT * tangent + W_THICKNESS * thickness
        + W_CONTRAST * contrast + W_RADIOMETRY * radiometry
        + W_CENTER_VALUE * center_value + W_FINAL_POLARITY * polarity_final
        + W_TRANSPORT * transport + W_DONOR_RANK * donor_rank
        + W_HF * hf + W_GUIDE_ORIENT * guide_orient
    )

    ctype = batch.get("center_type")
    if isinstance(ctype, (list, tuple)) and ctype:
        s = str(ctype[0]).lower()
        if "crossing" in s or "diagonal" in s or "strong_edge" in s or "edge" in s:
            total = 1.15 * total

    parts = {
        **base_parts,
        "pixel_sf": pixel.detach(), "support_bce": support_bce.detach(), "support_dice": support_dice.detach(),
        "core_bce": core_bce.detach(), "core_dice": core_dice.detach(), "dark": dark_loss.detach(),
        "bright": bright_loss.detach(), "tangent": tangent.detach(), "thickness": thickness.detach(),
        "contrast": contrast.detach(), "radiometry": radiometry.detach(), "center_value": center_value.detach(),
        "final_polarity": polarity_final.detach(), "transport": transport.detach(), "donor_rank": donor_rank.detach(),
        "donor_hit": output["donor_oracle_hit"].detach(), "hf_sf": hf.detach(), "guide_orient_sf": guide_orient.detach(),
        "mean_support": masked_mean(output["support_prob"].detach(), mask),
        "mean_core": masked_mean(output["core_prob"].detach(), mask),
        "mean_conf": masked_mean(output["transport_confidence"].detach(), mask),
    }
    return total, parts


def filler_forward(model, batch, structure, supervised: bool = True):
    target = batch["target"].detach() if supervised else None
    weight = None
    if supervised:
        t = structure_targets(batch)
        weight = (batch["mask"] * (0.15 + 1.4 * t["support"] + 2.5 * t["core"])).detach()
    return model(
        corrupted_cube=batch["cube"], observed_target=batch["observed_target"], mask=batch["mask"],
        spatial_valid=batch["spatial_valid"], valid_fraction=batch["valid_fraction"],
        distance=batch["distance"], band_position=batch["band_position"], structure=structure,
        supervision_target=target, supervision_weight=weight,
    )


@torch.no_grad()
def evaluate(model, structure_net, loader, device):
    model.eval(); structure_net.eval()
    keys = [
        "pixel", "ssim", "pixel_sf", "support_bce", "support_dice", "core_bce", "core_dice",
        "dark", "bright", "tangent", "thickness", "contrast", "radiometry", "center_value",
        "final_polarity", "transport", "donor_rank", "donor_hit", "hf_sf", "guide_orient_sf",
        "mean_support", "mean_core", "mean_conf",
    ]
    sums = {k: 0.0 for k in keys}; total = 0.0; count = 0
    for batch in loader:
        batch = BASE.move_batch(batch, device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            s = BASE.structure_forward(structure_net, batch)
            out = filler_forward(model, batch, s, supervised=True)
            loss, parts = structure_first_loss(out, batch, s)
        total += float(loss.item()); count += 1
        for k in keys:
            sums[k] += float(parts[k].item())
    result = {"loss": total / max(count, 1)}
    result.update({k: v / max(count, 1) for k, v in sums.items()})
    return result


def save_checkpoint(path, raw, ema, opt, sched, epoch, best_val, v27_ck, v27_path, history):
    torch.save({
        "architecture": StructureFirstV13Filler.ARCHITECTURE_VERSION,
        "epoch": int(epoch), "best_val": float(best_val),
        "patch_size": int(v27_ck.get("patch_size", 65)),
        "n_bands": int(v27_ck["n_bands"]),
        "band_mean": np.asarray(v27_ck["band_mean"], np.float32),
        "band_std": np.asarray(v27_ck["band_std"], np.float32),
        "wavelengths_nm": np.asarray(v27_ck["wavelengths_nm"], np.float32),
        "source_v27_checkpoint": str(v27_path),
        "uses_v27_final_reconstruction": False,
        "warm_started_from_previous_v13": False,
        "model_state_dict": ema.state_dict(),
        "train_model_state_dict": raw.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "history": history,
        "settings": {
            "structure_feature_channels": int(raw.structure_feature_channels),
            "spectral_width": int(raw.spectral_width), "base_width": int(raw.base_width),
            "transport_embed_ch": int(raw.transport_embed_ch), "transport_heads": int(raw.transport_heads),
            "transport_topk": int(raw.transport_topk), "transport_radius": int(raw.transport_radius),
            "transport_temperature": float(raw.transport_temperature),
            "radiometry_delta_scale": float(raw.radiometry_delta_scale), "detail_scale": float(raw.detail_scale),
            "epochs": int(EPOCHS), "train_samples": int(TRAIN_SAMPLES), "val_samples": int(VAL_SAMPLES),
            "v27_stage_a_frozen": True, "v27_final_output_used": False,
            "v16_used": False, "bridge_used": False, "selector_used": False,
            "structure_decoder_native_resolution": True, "structure_first": True,
        },
    }, path)


def main():
    distributed, rank, local_rank, world_size, device = BASE.setup_runtime()
    main_process = rank == 0
    try:
        BASE.set_seed(SEED + rank)
        train_cubes, val_cubes, wavelengths = load_global_training_data()
        v27_path = BASE.v27_checkpoint_path(); v27_ck = BASE.torch_load(v27_path, "cpu")
        if int(v27_ck["n_bands"]) != int(train_cubes[0].shape[-1]):
            raise RuntimeError("V27 spectral dimension differs from current fold")
        if not np.allclose(np.asarray(v27_ck["wavelengths_nm"]), np.asarray(wavelengths), atol=1e-3, rtol=0):
            raise RuntimeError("V27 wavelength grid differs from current fold")

        structure_net, structure_ch = BASE.build_frozen_structure(v27_ck, device)
        mean = np.asarray(v27_ck["band_mean"], np.float32); std = np.asarray(v27_ck["band_std"], np.float32)
        train_ds = V27.V27LearnedCorrespondenceDataset(train_cubes, mean, std, TRAIN_SAMPLES, SEED + 101, True)
        val_ds = V27.V27LearnedCorrespondenceDataset(val_cubes, mean, std, VAL_SAMPLES, SEED + 202, False) if main_process else None
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED) if distributed else None
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=sampler is None, sampler=sampler,
                                  num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True) if main_process else None

        raw = StructureFirstV13Filler(int(v27_ck["n_bands"]), structure_ch).to(device)
        ema = EMA(raw, EMA_DECAY)
        model = DDP(raw, device_ids=[local_rank], broadcast_buffers=False) if distributed else raw
        opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(EPOCHS, 1), eta_min=LEARNING_RATE * 0.08)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

        tag = str(config.TEST_DATASET_FOLDER).lower()
        outdir = Path(config.MODELS_DIR) / METHOD / f"new_v13_structure_first_{tag}"
        if main_process: outdir.mkdir(parents=True, exist_ok=True)
        if distributed: dist.barrier()
        last_path, best_path = outdir / "last_model.pt", outdir / "best_model.pt"
        hist_path = outdir / "training_history.csv"

        start, best, no_improve, history = 1, float("inf"), 0, []
        if NO_RESUME and main_process:
            for p in (last_path, best_path, hist_path):
                if p.exists(): p.unlink()
        if distributed: dist.barrier()
        if (not NO_RESUME) and last_path.exists():
            ck = BASE.torch_load(last_path, "cpu")
            raw.load_state_dict(ck.get("train_model_state_dict", ck["model_state_dict"]), strict=True)
            ema.model.load_state_dict(ck["model_state_dict"], strict=True)
            opt.load_state_dict(ck["optimizer_state_dict"]); sched.load_state_dict(ck["scheduler_state_dict"])
            start = int(ck["epoch"]) + 1; best = float(ck.get("best_val", best)); history = list(ck.get("history", []))

        if main_process:
            trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
            frozen = sum(p.numel() for p in structure_net.parameters())
            print("V13 STRUCTURE-FIRST — NATIVE GEOMETRY -> RADIOMETRY -> TEXTURE")
            print(f"Frozen V27 Stage-A: {frozen/1e6:.3f} M | Trainable Structure-First V13: {trainable/1e6:.3f} M")
            print("Training FROM SCRATCH: no V3.2/E2E/Corr/Sparse warm-start.")
            print("No V27 final image, no V16, no bridge/trace/selector/post-fill.")
            print(f"Epochs={EPOCHS} train_samples={TRAIN_SAMPLES} val_samples={VAL_SAMPLES} world_size={world_size}")

        for epoch in range(start, EPOCHS + 1):
            train_ds.set_epoch(epoch)
            if sampler is not None: sampler.set_epoch(epoch)
            model.train(); structure_net.eval(); running = seen = 0.0
            bar = tqdm(train_loader, desc=f"StructureFirst epoch {epoch}/{EPOCHS}", disable=not main_process, dynamic_ncols=True)
            for batch in bar:
                batch = BASE.move_batch(batch, device); opt.zero_grad(set_to_none=True)
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    s = BASE.structure_forward(structure_net, batch)
                s = {k: v.detach() for k, v in s.items()}
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    out = filler_forward(model, batch, s, supervised=True)
                    loss, parts = structure_first_loss(out, batch, s)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update(); ema.update(model)
                running += float(loss.item()); seen += 1.0
                if main_process:
                    bar.set_postfix(loss=f"{running/max(seen,1):.4f}", core=f"{parts['core_dice'].item():.3f}",
                                    pol=f"{parts['final_polarity'].item():.3f}", hit=f"{parts['donor_hit'].item():.2f}")

            if distributed:
                pack = torch.tensor([running, seen], dtype=torch.float64, device=device); dist.all_reduce(pack)
                train_loss = float(pack[0].item() / max(pack[1].item(), 1.0)); dist.barrier()
            else:
                train_loss = running / max(seen, 1.0)
            lr = float(opt.param_groups[0]["lr"]); sched.step(); stop = False

            if main_process:
                val = evaluate(ema.model.to(device), structure_net, val_loader, device)
                row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"], "lr": lr,
                       **{f"val_{k}": val[k] for k in val if k != "loss"}}
                history.append(row); improved = val["loss"] < best - 1e-6
                if improved: best, no_improve = val["loss"], 0
                else: no_improve += 1
                current = model.module if isinstance(model, DDP) else model
                save_checkpoint(last_path, current, ema.model, opt, sched, epoch, best, v27_ck, v27_path, history)
                if improved: save_checkpoint(best_path, current, ema.model, opt, sched, epoch, best, v27_ck, v27_path, history)
                with hist_path.open("w", newline="", encoding="utf-8") as f:
                    wr = csv.DictWriter(f, fieldnames=list(row.keys())); wr.writeheader(); wr.writerows(history)
                print(f"Epoch {epoch}: train={train_loss:.6f} val={val['loss']:.6f} pixel={val['pixel']:.4f} coreDiceLoss={val['core_dice']:.4f} polarity={val['final_polarity']:.4f} donorHit={val['donor_hit']:.3f}")
                stop = epoch >= MIN_EPOCHS and no_improve >= PATIENCE
            flag = torch.tensor([int(stop)], dtype=torch.int32, device=device)
            if distributed: dist.broadcast(flag, src=0)
            if bool(flag.item()):
                if main_process: print(f"Early stopping at epoch {epoch}")
                break

        if main_process:
            print("Structure-First V13 training complete:", best_path)
    finally:
        BASE.cleanup(distributed)


if __name__ == "__main__":
    main()
