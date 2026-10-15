#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the TPSSR topology-completion model.

Topology-First makes a skeleton/centerline the primary hidden representation.
Support is derived from skeleton + width, polarity is a single signed variable,
and radiometry is synthesized only after topology exists.
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
from src.topology_completion import TopologyCompletionNetwork

METHOD_DIR = CORE / "reconstruction_methods" / "tpssr"
METHOD = "tpssr"


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


SF = import_file("topology_sf_training_helpers", METHOD_DIR / "train_structure_first.py")
BASE = SF.BASE
Geometry = SF.Geometry

EPOCHS = int(os.environ.get("TPSSR_TOPOLOGY_EPOCHS", "22"))
MIN_EPOCHS = int(os.environ.get("TPSSR_TOPOLOGY_MIN_EPOCHS", "14"))
PATIENCE = int(os.environ.get("TPSSR_TOPOLOGY_PATIENCE", "5"))
TRAIN_SAMPLES = int(os.environ.get("TPSSR_TOPOLOGY_TRAIN_SAMPLES", "4200"))
VAL_SAMPLES = int(os.environ.get("TPSSR_TOPOLOGY_VAL_SAMPLES", "900"))
BATCH_SIZE = int(os.environ.get("TPSSR_TOPOLOGY_BATCH_SIZE", "1"))
NUM_WORKERS = int(os.environ.get("TPSSR_TOPOLOGY_NUM_WORKERS", "2"))
LEARNING_RATE = float(os.environ.get("TPSSR_TOPOLOGY_LR", "9.0e-5"))
WEIGHT_DECAY = float(os.environ.get("TPSSR_TOPOLOGY_WEIGHT_DECAY", "1.0e-5"))
GRAD_CLIP = float(os.environ.get("TPSSR_TOPOLOGY_GRAD_CLIP", "2.0"))
EMA_DECAY = float(os.environ.get("TPSSR_TOPOLOGY_EMA_DECAY", "0.995"))
NO_RESUME = os.environ.get("TPSSR_TOPOLOGY_NO_RESUME", "0") == "1"

# Topology supervision deliberately dominates the representation stage.
W_PIXEL = 0.62
W_SKEL_BCE = 0.72
W_SKEL_DICE = 0.78
W_CLDICE = 1.05
W_CONTINUITY = 0.90
W_SUPPORT_DICE = 0.30
W_POLARITY = 0.72
W_TANGENT = 0.30
W_WIDTH = 0.22
W_CONTRAST = 0.48
W_RADIOMETRY = 0.48
W_CENTER_VALUE = 0.82
W_FINAL_POLARITY = 0.92
W_TRANSPORT = 0.14
W_DONOR_RANK = 0.06
W_HF = 0.26
W_GUIDE_ORIENT = 0.12

SEED = int(getattr(config, "SEED", 1234)) + 131510


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


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, 3, stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iterations: int = 7) -> torch.Tensor:
    x = img.clamp(0.0, 1.0)
    opened = soft_open(x)
    skel = F.relu(x - opened)
    for _ in range(iterations):
        x = soft_erode(x)
        opened = soft_open(x)
        delta = F.relu(x - opened)
        skel = skel + F.relu(delta - skel * delta)
    return skel.clamp(0.0, 1.0)


def erosion_depth(binary_support: torch.Tensor, steps: int = 4) -> torch.Tensor:
    x = binary_support
    depth = torch.zeros_like(x)
    for _ in range(steps):
        depth = depth + x
        x = (1.0 - F.max_pool2d(1.0 - x, 3, 1, 1)).clamp(0.0, 1.0)
    return (depth / float(steps)).clamp(0.0, 1.0)


def topology_targets(batch: dict) -> dict[str, torch.Tensor]:
    target = batch["target"]
    mean, std = local_stats(target, 11)
    contrast = batch.get("contrast_target", target - mean)

    # Single signed polarity target. Negative means dark, positive bright.
    pol_denom = 0.18 * std + 0.028
    signed_polarity = torch.tanh(contrast / pol_denom)
    pol_strength = torch.sigmoid((contrast.abs() - 0.30 * std) / (0.10 * std + 0.020))

    edge = batch.get("edge_target")
    gx, gy = Geometry.sobel_xy(target)
    if edge is None:
        mag = torch.sqrt(gx.square() + gy.square() + 1e-8)
        lm = F.avg_pool2d(mag, 9, 1, 4)
        edge = torch.clamp(mag / (2.0 * lm + 0.035), 0.0, 1.0)
    else:
        edge = edge.clamp(0.0, 1.0)

    # Thin-line evidence that is not diluted by an 11x11 local statistic.
    hf3 = (target - F.avg_pool2d(target, 3, 1, 1)).abs()
    hf_strength = torch.sigmoid((hf3 - 0.12 * std) / (0.07 * std + 0.012))
    support = torch.maximum(edge, torch.maximum(0.92 * pol_strength, 0.78 * hf_strength)).clamp(0.0, 1.0)

    # Native one/few-pixel centerline target: combine local structural maxima
    # with a differentiable skeleton of polarity evidence.
    score = torch.maximum(edge, torch.maximum(pol_strength, hf_strength)).clamp(0.0, 1.0)
    mx = F.max_pool2d(score, 3, 1, 1)
    nms_ridge = score * torch.exp(-((mx - score).clamp_min(0.0) / 0.075).square())
    polar_skeleton = soft_skeletonize(pol_strength * (0.35 + 0.65 * support), 7)
    skeleton = torch.maximum(nms_ridge * (0.30 + 0.70 * pol_strength), polar_skeleton).clamp(0.0, 1.0)

    hard_support = (support.detach() > 0.48).to(target.dtype)
    width = erosion_depth(hard_support, 4)

    orient = batch.get("orientation_target")
    if orient is None:
        den = gx.square() + gy.square() + 1e-6
        grad_c2 = (gx.square() - gy.square()) / den
        grad_s2 = (2.0 * gx * gy) / den
        orient = torch.cat([grad_c2, grad_s2], 1)
    tangent = -F.normalize(orient, dim=1, eps=1e-6)

    return {
        "mean": mean, "std": std, "contrast": contrast,
        "signed_polarity": signed_polarity, "polarity_strength": pol_strength,
        "edge": edge, "support": support, "skeleton": skeleton,
        "width": width, "tangent": tangent,
    }


def weighted_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return masked_mean(raw * (1.0 + 2.0 * target.detach()), weight)


def soft_dice(prob: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    w = weight.to(prob.dtype)
    p = prob * w; t = target * w
    inter = (p * t).sum()
    den = p.square().sum() + t.square().sum() + 1e-6
    return 1.0 - (2.0 * inter + 1e-6) / den


def soft_cldice(pred_skel: torch.Tensor, pred_support: torch.Tensor,
                target_skel: torch.Tensor, target_support: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
    w = mask.to(pred_skel.dtype)
    ps = pred_skel * w; pp = pred_support * w
    ts = target_skel * w; tp = target_support * w
    tprec = (ps * tp).sum() / ps.sum().clamp_min(1e-6)
    tsens = (ts * pp).sum() / ts.sum().clamp_min(1e-6)
    return 1.0 - (2.0 * tprec * tsens + 1e-6) / (tprec + tsens + 1e-6)


def final_signed_polarity_loss(pred: torch.Tensor, t: dict, mask: torch.Tensor) -> torch.Tensor:
    # Directly punish a dark/bright skeleton becoming gray or flipping sign.
    # Explicit FP32 avoids AMP probability-space numerical issues.
    with torch.autocast(device_type=pred.device.type, enabled=False):
        pred_f = pred.float(); mean_f = t["mean"].float(); std_f = t["std"].float()
        signed_t = t["signed_polarity"].float(); strength = t["polarity_strength"].float()
        skel = t["skeleton"].float(); mask_f = mask.float(); inner = inner_mask(mask_f)
        pred_contrast = pred_f - mean_f
        pred_signed = torch.tanh(pred_contrast / (0.18 * std_f + 0.028))
        w = mask_f * (0.10 + 2.4 * skel + 3.2 * inner * strength)
        reg = F.smooth_l1_loss(pred_signed, signed_t, reduction="none", beta=0.12)
        margin = 0.08 * std_f + 0.012
        sign_hinge = F.relu(margin * strength - signed_t * pred_contrast)
        return masked_mean(reg + 0.75 * sign_hinge, w)


def topology_first_loss(output: dict, batch: dict, geometry_structure: dict):
    base_out = {"prediction": output["prediction"], "coarse_prediction": output["prediction"]}
    base_loss, base_parts = Geometry.reconstruction_loss(base_out, batch)

    target = batch["target"]; mask = batch["mask"]
    t = topology_targets(batch); inner = inner_mask(mask)

    pix_w = mask * (1.0 + 1.6 * t["support"] + 3.2 * t["skeleton"] + 3.0 * inner * t["polarity_strength"])
    pixel = masked_mean(F.smooth_l1_loss(output["prediction"], target, reduction="none", beta=0.09), pix_w)

    skel_bce = weighted_bce(output["skeleton_logit"], t["skeleton"], mask)
    skel_dice = soft_dice(output["skeleton_prob"], t["skeleton"], mask)
    cldice = soft_cldice(output["skeleton_prob"], output["support_prob"], t["skeleton"], t["support"], mask)
    continuity = masked_mean(
        -torch.log(output["skeleton_prob"].clamp_min(1e-5)),
        inner * t["skeleton"] * (0.25 + 2.75 * t["polarity_strength"]),
    )
    support_dice = soft_dice(output["support_prob"], t["support"], mask)

    polarity = masked_mean(
        F.smooth_l1_loss(output["polarity"], t["signed_polarity"], reduction="none", beta=0.12),
        mask * (0.15 + 2.1 * t["support"] + 3.0 * t["skeleton"]),
    )
    dot = (output["tangent"] * t["tangent"]).sum(1, keepdim=True).clamp(-1.0, 1.0)
    tangent = masked_mean(1.0 - dot, mask * (0.10 + 2.2 * t["support"] + 2.0 * t["skeleton"]))
    width = masked_mean((output["width_expected"] - t["width"]).abs(), mask * (0.15 + 2.0 * t["skeleton"]))

    contrast = masked_mean(
        F.smooth_l1_loss(output["signed_contrast"], t["contrast"], reduction="none", beta=0.07),
        mask * (0.15 + 2.1 * t["support"] + 3.0 * inner * t["polarity_strength"]),
    )
    radiometry = masked_mean(
        F.smooth_l1_loss(output["radiometry"], target, reduction="none", beta=0.09),
        mask * (0.20 + 1.8 * t["support"] + 3.0 * inner * t["polarity_strength"]),
    )
    center_value = masked_mean(
        F.smooth_l1_loss(output["prediction"], target, reduction="none", beta=0.06),
        inner * (0.10 + 4.0 * t["polarity_strength"] + 2.8 * t["skeleton"]),
    )
    final_polarity = final_signed_polarity_loss(output["prediction"], t, mask)

    transport = masked_mean(
        F.smooth_l1_loss(output["transported"], target, reduction="none", beta=0.09),
        mask * (0.10 + 0.9 * t["support"] + 1.3 * t["skeleton"]),
    )
    donor_rank = output["donor_rank_loss"]

    grad_mask = F.max_pool2d(mask, 3, 1, 1)
    hf = 0.5 * (
        masked_mean((high_pass(output["prediction"], 3) - high_pass(target, 3)).abs(), grad_mask)
        + masked_mean((high_pass(output["prediction"], 7) - high_pass(target, 7)).abs(), grad_mask)
    )

    gx, gy = Geometry.sobel_xy(output["prediction"])
    den = gx.square() + gy.square() + 1e-6
    cos2 = (gx.square() - gy.square()) / den
    sin2 = (2.0 * gx * gy) / den
    guide = geometry_structure["orientation"].detach()
    guide_dot = (cos2 * guide[:, 0:1] + sin2 * guide[:, 1:2]).clamp(-1.0, 1.0)
    guide_orient = masked_mean(1.0 - guide_dot, mask * (0.15 + geometry_structure["edge_prob"].detach()))

    total = (
        base_loss + W_PIXEL * pixel
        + W_SKEL_BCE * skel_bce + W_SKEL_DICE * skel_dice + W_CLDICE * cldice + W_CONTINUITY * continuity
        + W_SUPPORT_DICE * support_dice + W_POLARITY * polarity + W_TANGENT * tangent + W_WIDTH * width
        + W_CONTRAST * contrast + W_RADIOMETRY * radiometry + W_CENTER_VALUE * center_value
        + W_FINAL_POLARITY * final_polarity + W_TRANSPORT * transport + W_DONOR_RANK * donor_rank
        + W_HF * hf + W_GUIDE_ORIENT * guide_orient
    )

    ctype = batch.get("center_type")
    if isinstance(ctype, (list, tuple)) and ctype:
        s = str(ctype[0]).lower()
        if "crossing" in s or "diagonal" in s or "strong_edge" in s or "edge" in s:
            total = 1.18 * total

    parts = {
        **base_parts,
        "pixel_tf": pixel.detach(), "skel_bce": skel_bce.detach(), "skel_dice": skel_dice.detach(),
        "cldice": cldice.detach(), "continuity": continuity.detach(), "support_dice": support_dice.detach(),
        "polarity": polarity.detach(), "tangent": tangent.detach(), "width": width.detach(),
        "contrast": contrast.detach(), "radiometry": radiometry.detach(), "center_value": center_value.detach(),
        "final_polarity": final_polarity.detach(), "transport": transport.detach(), "donor_rank": donor_rank.detach(),
        "donor_hit": output["donor_oracle_hit"].detach(), "hf_tf": hf.detach(), "guide_orient_tf": guide_orient.detach(),
        "mean_skeleton": masked_mean(output["skeleton_prob"].detach(), mask),
        "mean_support": masked_mean(output["support_prob"].detach(), mask),
        "mean_abs_polarity": masked_mean(output["polarity"].detach().abs(), mask),
        "mean_conf": masked_mean(output["transport_confidence"].detach(), mask),
    }
    return total, parts


def filler_forward(model, batch, structure, supervised: bool = True):
    target = batch["target"].detach() if supervised else None
    weight = None
    if supervised:
        t = topology_targets(batch)
        weight = (batch["mask"] * (0.10 + 1.2 * t["support"] + 2.8 * t["skeleton"])).detach()
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
        "pixel", "ssim", "pixel_tf", "skel_bce", "skel_dice", "cldice", "continuity", "support_dice",
        "polarity", "tangent", "width", "contrast", "radiometry", "center_value", "final_polarity",
        "transport", "donor_rank", "donor_hit", "hf_tf", "guide_orient_tf",
        "mean_skeleton", "mean_support", "mean_abs_polarity", "mean_conf",
    ]
    sums = {k: 0.0 for k in keys}; total = 0.0; count = 0
    for batch in loader:
        batch = BASE.move_batch(batch, device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            s = BASE.structure_forward(structure_net, batch)
            out = filler_forward(model, batch, s, supervised=True)
            loss, parts = topology_first_loss(out, batch, s)
        total += float(loss.item()); count += 1
        for k in keys:
            sums[k] += float(parts[k].item())
    result = {"loss": total / max(count, 1)}
    result.update({k: v / max(count, 1) for k, v in sums.items()})
    return result


def save_checkpoint(path, raw, ema, opt, sched, epoch, best_val, geometry_ck, geometry_path, history):
    torch.save({
        "architecture": TopologyCompletionNetwork.ARCHITECTURE_VERSION,
        "epoch": int(epoch), "best_val": float(best_val),
        "patch_size": int(geometry_ck.get("patch_size", 65)),
        "n_bands": int(geometry_ck["n_bands"]),
        "band_mean": np.asarray(geometry_ck["band_mean"], np.float32),
        "band_std": np.asarray(geometry_ck["band_std"], np.float32),
        "wavelengths_nm": np.asarray(geometry_ck["wavelengths_nm"], np.float32),
        "source_geometry_checkpoint": str(geometry_path),
        "uses_geometry_final_reconstruction": False,
        "warm_started_from_previous_topology": False,
        "model_state_dict": ema.state_dict(),
        "train_model_state_dict": raw.state_dict(),
        "optimizer_state_dict": opt.state_dict(), "scheduler_state_dict": sched.state_dict(),
        "history": history,
        "settings": {
            "structure_feature_channels": int(raw.structure_feature_channels),
            "spectral_width": int(raw.spectral_width), "base_width": int(raw.base_width),
            "transport_embed_ch": int(raw.transport_embed_ch), "transport_heads": int(raw.transport_heads),
            "transport_topk": int(raw.transport_topk), "transport_radius": int(raw.transport_radius),
            "transport_temperature": float(raw.transport_temperature),
            "radiometry_delta_scale": float(raw.radiometry_delta_scale), "detail_scale": float(raw.detail_scale),
            "epochs": int(EPOCHS), "train_samples": int(TRAIN_SAMPLES), "val_samples": int(VAL_SAMPLES),
            "geometry_frozen": True, "geometry_final_output_used": False, "detail_refinement_used": False,
            "bridge_used": False, "selector_used": False, "topology_decoder_native_resolution": True,
            "topology_first": True, "support_derived_from_skeleton_width": True,
            "single_signed_polarity": True, "cldice_supervision": True,
        },
    }, path)


def main():
    distributed, rank, local_rank, world_size, device = BASE.setup_runtime()
    main_process = rank == 0
    try:
        BASE.set_seed(SEED + rank)
        train_cubes, val_cubes, wavelengths = load_global_training_data()
        geometry_path = BASE.geometry_checkpoint_path(); geometry_ck = BASE.torch_load(geometry_path, "cpu")
        if int(geometry_ck["n_bands"]) != int(train_cubes[0].shape[-1]):
            raise RuntimeError("Geometry spectral dimension differs from current fold")
        if not np.allclose(np.asarray(geometry_ck["wavelengths_nm"]), np.asarray(wavelengths), atol=1e-3, rtol=0):
            raise RuntimeError("Geometry wavelength grid differs from current fold")

        structure_net, structure_ch = BASE.build_frozen_structure(geometry_ck, device)
        mean = np.asarray(geometry_ck["band_mean"], np.float32); std = np.asarray(geometry_ck["band_std"], np.float32)
        train_ds = Geometry.GeometryLearnedCorrespondenceDataset(train_cubes, mean, std, TRAIN_SAMPLES, SEED + 101, True)
        val_ds = Geometry.GeometryLearnedCorrespondenceDataset(val_cubes, mean, std, VAL_SAMPLES, SEED + 202, False) if main_process else None
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED) if distributed else None
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=sampler is None, sampler=sampler,
                                  num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True) if main_process else None

        raw = TopologyCompletionNetwork(int(geometry_ck["n_bands"]), structure_ch).to(device)
        ema = EMA(raw, EMA_DECAY)
        model = DDP(raw, device_ids=[local_rank], broadcast_buffers=False) if distributed else raw
        opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(EPOCHS, 1), eta_min=LEARNING_RATE * 0.08)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

        tag = str(config.TEST_DATASET_FOLDER).lower()
        outdir = Path(config.MODELS_DIR) / METHOD / f"topology_{tag}"
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
            print("Topology TOPOLOGY-FIRST — SKELETON -> SIGNED POLARITY -> RADIOMETRY -> TEXTURE")
            print(f"Frozen Geometry Stage-A: {frozen/1e6:.3f} M | Trainable Topology-First Topology: {trainable/1e6:.3f} M")
            print("Training FROM SCRATCH. Support is derived from skeleton+width; dark/bright are one signed polarity.")
            print("No Geometry final image, no Refinement, no bridge/trace/selector/post-fill.")
            print(f"Epochs={EPOCHS} train_samples={TRAIN_SAMPLES} val_samples={VAL_SAMPLES} world_size={world_size}")

        for epoch in range(start, EPOCHS + 1):
            train_ds.set_epoch(epoch)
            if sampler is not None: sampler.set_epoch(epoch)
            model.train(); structure_net.eval(); running = seen = 0.0
            bar = tqdm(train_loader, desc=f"TopologyFirst epoch {epoch}/{EPOCHS}", disable=not main_process, dynamic_ncols=True)
            for batch in bar:
                batch = BASE.move_batch(batch, device); opt.zero_grad(set_to_none=True)
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    s = BASE.structure_forward(structure_net, batch)
                s = {k: v.detach() for k, v in s.items()}
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    out = filler_forward(model, batch, s, supervised=True)
                    loss, parts = topology_first_loss(out, batch, s)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update(); ema.update(model)
                running += float(loss.item()); seen += 1.0
                if main_process:
                    bar.set_postfix(loss=f"{running/max(seen,1):.4f}", skel=f"{parts['skel_dice'].item():.3f}",
                                    cl=f"{parts['cldice'].item():.3f}", pol=f"{parts['final_polarity'].item():.3f}")

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
                save_checkpoint(last_path, current, ema.model, opt, sched, epoch, best, geometry_ck, geometry_path, history)
                if improved: save_checkpoint(best_path, current, ema.model, opt, sched, epoch, best, geometry_ck, geometry_path, history)
                with hist_path.open("w", newline="", encoding="utf-8") as f:
                    wr = csv.DictWriter(f, fieldnames=list(row.keys())); wr.writeheader(); wr.writerows(history)
                print(f"Epoch {epoch}: train={train_loss:.6f} val={val['loss']:.6f} pixel={val['pixel']:.4f} skelDiceLoss={val['skel_dice']:.4f} clDiceLoss={val['cldice']:.4f} polarity={val['final_polarity']:.4f}")
                stop = epoch >= MIN_EPOCHS and no_improve >= PATIENCE
            flag = torch.tensor([int(stop)], dtype=torch.int32, device=device)
            if distributed: dist.broadcast(flag, src=0)
            if bool(flag.item()):
                if main_process: print(f"Early stopping at epoch {epoch}")
                break

        if main_process:
            print("Topology-First Topology training complete:", best_path)
    finally:
        BASE.cleanup(distributed)


if __name__ == "__main__":
    main()
