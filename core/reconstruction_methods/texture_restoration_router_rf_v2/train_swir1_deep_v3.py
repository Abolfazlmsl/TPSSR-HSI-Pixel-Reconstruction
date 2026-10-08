#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TPSSR — SWIR1 DEEP EXPERT V3
=============================

ONE-FILE launcher + 2-GPU DDP trainer.

Run from the HSI_Saturation project root:

    python reconstruction_methods/texture_restoration_router_rf_v2/train_swir1_deep_v3.py \
        --gpus 2 --prepare

Why V3?
-------
The direct SWIR1 Deep-v2 diagnostic showed:

1) the 15x15 pixel-by-pixel Deep-v2 was too local for large S5 corruption;
2) the auxiliary gradient head could predict gradients without forcing the
   reconstructed image itself to contain the correct gradients;
3) the gated spectral anchor remained useful conceptually, but the residual
   branch was not accurate enough;
4) for whole-spectrum pixels, V2 increased texture energy but worsened RMSE,
   indicating that texture was being generated without enough fidelity.

V3 changes the Deep expert itself:

    41x41 spatial-spectral context
            ↓
    joint central 9x9 tile reconstruction
            ↓
    spectral-anchor residual branch when anchor exists
    direct context fallback when the whole spectrum is missing

Losses are applied DIRECTLY to the reconstructed tile:
    - pixel reconstruction
    - spatial gradient consistency
    - local SSIM / structure
    - spectral first-derivative consistency
    - conservative spectral-anchor safety / gate supervision

No auxiliary gradient prediction head is used.

Data policy
-----------
SWIR1 is the held-out DEVELOPMENT TEST image for this fold.
Training uses only load_global_training_data(), which under the SWIR1 LOIO
configuration should use SWIR2, SWIR3, SWIR4, SWIR5 for train/validation.
"""

from pathlib import Path
import sys
import os
import csv
import json
import math
import random
import argparse
import subprocess

HERE = Path(__file__).resolve()
PROJECT_ROOT = next(
    parent for parent in HERE.parents
    if (parent / "config.py").exists()
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import config
from src.evaluation_utils import load_global_training_data
from src.ml_common import band_stats


# =============================================================================
# DEVELOPMENT SETTINGS
# =============================================================================

# Larger context than V2 so difficult S5 regions can see intact surroundings.
CONTEXT_SIZE = 41

# Joint output tile. The network reconstructs all corrupted values in this
# central tile together rather than predicting one center spectrum at a time.
OUTPUT_TILE = 9

# Global synthetic sample counts. DistributedSampler divides these between GPUs.
TRAIN_SAMPLES_PER_EPOCH = 5000
VAL_SAMPLES = 1000
EPOCHS = 14

# Per GPU. Two GPUs => effective batch size 16.
BATCH_SIZE_PER_GPU = 8
NUM_WORKERS_PER_GPU = 4

LEARNING_RATE = 2.0e-4
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 2.0

# Compact but full-context model.
MODEL_WIDTH = 80
ANCHOR_WIDTH = 32

# Dilations give a receptive field larger than the 41x41 input context.
DILATIONS = (1, 2, 4, 8, 4, 2, 1)

# Synthetic training mixture — concentrate on the two hard families.
P_WHOLE_SPECTRUM = 0.45
P_SPATIO_SPECTRAL_BLOCK = 0.45
P_CONSECUTIVE_SPECTRAL = 0.10

# Direct reconstruction losses.
W_ANCHORED_PIXEL = 1.00
W_NO_ANCHOR_PIXEL = 1.50
W_SPATIAL_GRADIENT = 0.25
W_LOCAL_SSIM = 0.12
W_SPECTRAL_D1 = 0.08

# Conservative spectral-anchor logic.
W_GATE_SUPERVISION = 0.08
W_ANCHOR_SAFETY = 0.15
W_GATE_PRIOR = 0.02

# In normalized reflectance units.
GATE_ERROR_CENTER = 0.20
GATE_ERROR_TEMPERATURE = 0.06
ANCHOR_SAFETY_SCALE = 0.15

MODEL_SUBDIR = f"deep_v3_{str(config.TEST_DATASET_FOLDER).lower()}_dev_2gpu"
BEST_MODEL = "best_model.pt"
LAST_MODEL = "last_model.pt"
HISTORY_CSV = "training_history.csv"
SUMMARY_TXT = "TRAINING_SUMMARY.txt"

SEED = int(getattr(config, "SEED", 1234)) + 18001


# =============================================================================
# DDP
# =============================================================================

def setup_ddp():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this 2-GPU DDP trainer.")

    if (
        "RANK" not in os.environ
        or "LOCAL_RANK" not in os.environ
        or "WORLD_SIZE" not in os.environ
    ):
        raise RuntimeError(
            "DDP worker context is missing. Launch this file with plain Python:\n\n"
            "python reconstruction_methods/texture_restoration_router_rf_v2/"
            "train_swir1_deep_v3.py --gpus 2 --prepare"
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if world_size != 2:
        raise RuntimeError(
            f"Current V3 development setup expects exactly 2 GPUs; "
            f"WORLD_SIZE={world_size}."
        )

    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"Two GPUs requested, but only {torch.cuda.device_count()} "
            "CUDA device(s) are visible."
        )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Newer PyTorch versions accept device_id and avoid rank/device warnings.
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=device,
        )
    except TypeError:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )

    return rank, local_rank, world_size, device


def ddp_barrier(local_rank):
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier(device_ids=[int(local_rank)])
        except TypeError:
            dist.barrier()


def cleanup_ddp(local_rank):
    if dist.is_available() and dist.is_initialized():
        ddp_barrier(local_rank)
        dist.destroy_process_group()


def all_reduce_sum(value, device):
    tensor = torch.tensor(
        float(value),
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def normalize_patch(patch, band_mean, band_std):
    patch = np.asarray(patch, dtype=np.float32)
    finite = np.isfinite(patch)

    safe = patch.copy()

    for b in range(patch.shape[-1]):
        safe[..., b] = np.where(
            finite[..., b],
            safe[..., b],
            band_mean[b],
        )

    normalized = (
        safe
        - band_mean[None, None, :]
    ) / band_std[None, None, :]

    return normalized.astype(np.float32), finite


def spectral_anchor_1d(values, missing, wavelengths):
    """
    Spectral interpolation at one spatial pixel.

    values/missing are in normalized spectral space.

    If the whole spectrum is missing, availability is all zero.
    """
    values = np.asarray(values, dtype=np.float32)
    missing = np.asarray(missing, dtype=bool)
    wl = np.asarray(wavelengths, dtype=np.float32)

    n_bands = len(values)

    anchor = np.zeros(
        n_bands,
        dtype=np.float32,
    )

    available = np.zeros(
        n_bands,
        dtype=np.float32,
    )

    observed = (
        ~missing
        & np.isfinite(values)
    )

    obs_idx = np.flatnonzero(observed)

    if len(obs_idx) == 0:
        return anchor, available

    anchor[observed] = values[observed]
    available[observed] = 1.0

    for b in np.flatnonzero(missing):
        left = obs_idx[obs_idx < b]
        right = obs_idx[obs_idx > b]

        if len(left) > 0 and len(right) > 0:
            l = int(left[-1])
            r = int(right[0])

            denom = float(
                wl[r] - wl[l]
            )

            if abs(denom) < 1e-12:
                value = 0.5 * (
                    values[l]
                    + values[r]
                )
            else:
                t = float(
                    (wl[b] - wl[l])
                    / denom
                )

                value = (
                    (1.0 - t) * values[l]
                    + t * values[r]
                )

            anchor[b] = np.float32(value)
            available[b] = 1.0

        elif len(left) > 0:
            anchor[b] = values[int(left[-1])]
            available[b] = 1.0

        elif len(right) > 0:
            anchor[b] = values[int(right[0])]
            available[b] = 1.0

    return anchor, available


def anchor_map_for_tile(
    clean_tile,
    mask_tile,
    wavelengths,
):
    """
    Build per-pixel spectral anchors for the whole output tile.

    clean_tile/mask_tile: [T,T,B]
    returns anchor/available [T,T,B]
    """
    t_h, t_w, n_bands = clean_tile.shape

    anchor = np.zeros(
        (t_h, t_w, n_bands),
        dtype=np.float32,
    )

    available = np.zeros(
        (t_h, t_w, n_bands),
        dtype=np.float32,
    )

    for yy in range(t_h):
        for xx in range(t_w):
            a, av = spectral_anchor_1d(
                clean_tile[yy, xx],
                mask_tile[yy, xx],
                wavelengths,
            )

            anchor[yy, xx] = a
            available[yy, xx] = av

    return anchor, available


# =============================================================================
# SYNTHETIC CORRUPTION
# =============================================================================

def odd_choice(rng, values, maximum):
    valid = [
        int(v)
        for v in values
        if int(v) <= int(maximum)
    ]

    if not valid:
        return 1

    return int(rng.choice(valid))


def add_spatial_block(
    mask,
    cy,
    cx,
    block,
    spectral_slice,
):
    h, w, _b = mask.shape
    radius = block // 2

    cy = int(
        np.clip(
            cy,
            radius,
            h - radius - 1,
        )
    )

    cx = int(
        np.clip(
            cx,
            radius,
            w - radius - 1,
        )
    )

    y0 = cy - radius
    y1 = cy + radius + 1

    x0 = cx - radius
    x1 = cx + radius + 1

    mask[
        y0:y1,
        x0:x1,
        spectral_slice,
    ] = True


def make_corruption_mask(
    rng,
    context_size,
    output_tile,
    n_bands,
):
    """
    Hard-case synthetic masks for V3.

    The primary corruption always intersects the central output tile.
    Additional blocks make the context less trivial and resemble the real
    benchmark's multiple-component cases.
    """
    h = w = int(context_size)
    center = h // 2
    half_tile = output_tile // 2

    mask = np.zeros(
        (h, w, n_bands),
        dtype=bool,
    )

    u = float(rng.random())

    # The primary center is jittered but kept inside the output tile.
    primary_cy = int(
        center
        + rng.integers(
            -half_tile,
            half_tile + 1,
        )
    )

    primary_cx = int(
        center
        + rng.integers(
            -half_tile,
            half_tile + 1,
        )
    )

    if u < P_WHOLE_SPECTRUM:
        scenario = "whole_spectrum_pixels"

        primary_block = odd_choice(
            rng,
            [3, 5, 7, 9, 11, 13, 15],
            min(h - 2, 15),
        )

        add_spatial_block(
            mask,
            primary_cy,
            primary_cx,
            primary_block,
            slice(None),
        )

        # 0-3 additional whole-spectrum components.
        n_extra = int(
            rng.integers(
                0,
                4,
            )
        )

        for _ in range(n_extra):
            block = odd_choice(
                rng,
                [3, 5, 7, 9],
                min(h - 2, 9),
            )

            cy = int(
                rng.integers(
                    block // 2,
                    h - block // 2,
                )
            )

            cx = int(
                rng.integers(
                    block // 2,
                    w - block // 2,
                )
            )

            add_spatial_block(
                mask,
                cy,
                cx,
                block,
                slice(None),
            )

    elif u < (
        P_WHOLE_SPECTRUM
        + P_SPATIO_SPECTRAL_BLOCK
    ):
        scenario = "spatio_spectral_block"

        primary_block = odd_choice(
            rng,
            [7, 9, 11, 13, 15, 17, 19, 21],
            min(h - 2, 21),
        )

        frac = float(
            rng.uniform(
                0.10,
                0.50,
            )
        )

        length = int(
            np.clip(
                round(frac * n_bands),
                3,
                n_bands,
            )
        )

        start = int(
            rng.integers(
                0,
                max(
                    1,
                    n_bands - length + 1,
                ),
            )
        )

        add_spatial_block(
            mask,
            primary_cy,
            primary_cx,
            primary_block,
            slice(
                start,
                start + length,
            ),
        )

        # Sometimes add a second spectral-spatial component.
        if float(rng.random()) < 0.35:
            block2 = odd_choice(
                rng,
                [5, 7, 9, 11],
                min(h - 2, 11),
            )

            frac2 = float(
                rng.uniform(
                    0.08,
                    0.30,
                )
            )

            length2 = int(
                np.clip(
                    round(frac2 * n_bands),
                    3,
                    n_bands,
                )
            )

            start2 = int(
                rng.integers(
                    0,
                    max(
                        1,
                        n_bands - length2 + 1,
                    ),
                )
            )

            cy2 = int(
                rng.integers(
                    block2 // 2,
                    h - block2 // 2,
                )
            )

            cx2 = int(
                rng.integers(
                    block2 // 2,
                    w - block2 // 2,
                )
            )

            add_spatial_block(
                mask,
                cy2,
                cx2,
                block2,
                slice(
                    start2,
                    start2 + length2,
                ),
            )

    else:
        scenario = "consecutive_spectral_bands"

        frac = float(
            rng.uniform(
                0.05,
                0.30,
            )
        )

        length = int(
            np.clip(
                round(frac * n_bands),
                2,
                n_bands,
            )
        )

        start = int(
            rng.integers(
                0,
                max(
                    1,
                    n_bands - length + 1,
                ),
            )
        )

        block = odd_choice(
            rng,
            [1, 3, 5],
            min(h - 2, 5),
        )

        add_spatial_block(
            mask,
            primary_cy,
            primary_cx,
            block,
            slice(
                start,
                start + length,
            ),
        )

    # Ensure the output tile actually contains target corruption.
    t0 = center - half_tile
    t1 = center + half_tile + 1

    if not np.any(
        mask[
            t0:t1,
            t0:t1,
            :
        ]
    ):
        b = int(
            rng.integers(
                0,
                n_bands,
            )
        )

        mask[
            center,
            center,
            b,
        ] = True

    return mask, scenario


# =============================================================================
# DATASET
# =============================================================================

class DeepV3SyntheticDataset(Dataset):
    def __init__(
        self,
        cubes,
        wavelengths,
        band_mean,
        band_std,
        samples,
        context_size,
        output_tile,
        seed,
        training=True,
    ):
        self.cubes = [
            np.asarray(cube)
            for cube in cubes
        ]

        self.wavelengths = np.asarray(
            wavelengths,
            dtype=np.float32,
        )

        self.band_mean = np.asarray(
            band_mean,
            dtype=np.float32,
        )

        self.band_std = np.asarray(
            band_std,
            dtype=np.float32,
        )

        self.samples = int(samples)
        self.context_size = int(context_size)
        self.output_tile = int(output_tile)

        if self.context_size % 2 == 0:
            raise ValueError(
                "CONTEXT_SIZE must be odd."
            )

        if self.output_tile % 2 == 0:
            raise ValueError(
                "OUTPUT_TILE must be odd."
            )

        if self.output_tile >= self.context_size:
            raise ValueError(
                "OUTPUT_TILE must be smaller than CONTEXT_SIZE."
            )

        self.radius = (
            self.context_size // 2
        )

        self.tile_half = (
            self.output_tile // 2
        )

        self.seed = int(seed)
        self.training = bool(training)
        self.epoch = 0

        n_bands = int(
            self.cubes[0].shape[-1]
        )

        for cube in self.cubes:
            if cube.ndim != 3:
                raise ValueError(
                    f"Expected HxWxB cube, got {cube.shape}"
                )

            if cube.shape[-1] != n_bands:
                raise ValueError(
                    "All cubes must have the same band count."
                )

        self.n_bands = n_bands

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.samples

    def _rng(self, index):
        epoch_term = (
            self.epoch
            if self.training
            else 0
        )

        return np.random.default_rng(
            self.seed
            + int(index) * 1009
            + epoch_term * 1000003
        )

    def _sample_clean_patch(self, rng):
        last_patch = None

        for _ in range(40):
            cube = self.cubes[
                int(
                    rng.integers(
                        0,
                        len(self.cubes),
                    )
                )
            ]

            h, w, _b = cube.shape
            r = self.radius

            if (
                h <= 2 * r + 2
                or w <= 2 * r + 2
            ):
                raise RuntimeError(
                    f"Cube {cube.shape} is too small for "
                    f"CONTEXT_SIZE={self.context_size}"
                )

            y = int(
                rng.integers(
                    r,
                    h - r,
                )
            )

            x = int(
                rng.integers(
                    r,
                    w - r,
                )
            )

            patch = np.asarray(
                cube[
                    y - r:y + r + 1,
                    x - r:x + r + 1,
                    :
                ],
                dtype=np.float32,
            )

            last_patch = patch

            c = r
            th = self.tile_half

            tile = patch[
                c - th:c + th + 1,
                c - th:c + th + 1,
                :
            ]

            # Require useful GT in the central output tile.
            if np.mean(
                np.isfinite(tile)
            ) >= 0.98:
                return patch

        return last_patch

    def __getitem__(self, index):
        rng = self._rng(index)

        clean_patch = self._sample_clean_patch(
            rng
        )

        norm_clean, finite = normalize_patch(
            clean_patch,
            self.band_mean,
            self.band_std,
        )

        corruption, scenario = make_corruption_mask(
            rng,
            self.context_size,
            self.output_tile,
            self.n_bands,
        )

        # Never construct a synthetic target at invalid original values.
        corruption &= finite

        c = self.radius
        th = self.tile_half

        y0 = c - th
        y1 = c + th + 1
        x0 = c - th
        x1 = c + th + 1

        tile_mask = corruption[
            y0:y1,
            x0:x1,
            :
        ].copy()

        # Rare fallback if original invalid values removed all synthetic targets.
        if not np.any(tile_mask):
            valid = np.argwhere(
                finite[
                    y0:y1,
                    x0:x1,
                    :
                ]
            )

            if len(valid) > 0:
                yy, xx, bb = valid[
                    int(
                        rng.integers(
                            0,
                            len(valid),
                        )
                    )
                ]

                corruption[
                    y0 + int(yy),
                    x0 + int(xx),
                    int(bb),
                ] = True

                tile_mask = corruption[
                    y0:y1,
                    x0:x1,
                    :
                ].copy()

        corrupted = norm_clean.copy()
        corrupted[corruption] = 0.0

        clean_tile = norm_clean[
            y0:y1,
            x0:x1,
            :
        ].astype(np.float32)

        valid_tile = finite[
            y0:y1,
            x0:x1,
            :
        ].astype(np.float32)

        anchor, anchor_available = anchor_map_for_tile(
            clean_tile,
            tile_mask,
            self.wavelengths,
        )

        # PyTorch layout B,H,W.
        input_cube = np.moveaxis(
            corrupted,
            -1,
            0,
        ).astype(np.float32)

        input_mask = np.moveaxis(
            corruption.astype(np.float32),
            -1,
            0,
        )

        target = np.moveaxis(
            clean_tile,
            -1,
            0,
        ).astype(np.float32)

        target_mask = np.moveaxis(
            tile_mask.astype(np.float32),
            -1,
            0,
        )

        target_valid = np.moveaxis(
            valid_tile,
            -1,
            0,
        ).astype(np.float32)

        anchor = np.moveaxis(
            anchor,
            -1,
            0,
        ).astype(np.float32)

        anchor_available = np.moveaxis(
            anchor_available,
            -1,
            0,
        ).astype(np.float32)

        scenario_id = {
            "whole_spectrum_pixels": 0,
            "spatio_spectral_block": 1,
            "consecutive_spectral_bands": 2,
        }[scenario]

        return {
            "cube": torch.from_numpy(input_cube),
            "mask": torch.from_numpy(input_mask),
            "target": torch.from_numpy(target),
            "target_mask": torch.from_numpy(target_mask),
            "target_valid": torch.from_numpy(target_valid),
            "anchor": torch.from_numpy(anchor),
            "anchor_available": torch.from_numpy(anchor_available),
            "scenario_id": torch.tensor(
                scenario_id,
                dtype=torch.long,
            ),
        }


# =============================================================================
# MODEL
# =============================================================================

class DilatedResidualBlock(nn.Module):
    def __init__(
        self,
        width,
        dilation,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(
            8,
            width,
        )

        self.conv1 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            padding=int(dilation),
            dilation=int(dilation),
        )

        self.norm2 = nn.GroupNorm(
            8,
            width,
        )

        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            padding=int(dilation),
            dilation=int(dilation),
        )

    def forward(self, x):
        residual = x

        x = self.conv1(
            F.gelu(
                self.norm1(x)
            )
        )

        x = self.conv2(
            F.gelu(
                self.norm2(x)
            )
        )

        return residual + 0.5 * x


class DeepExpertV3(nn.Module):
    ARCHITECTURE_VERSION = (
        "deep_expert_v3_joint_tile_gated_anchor_direct_structure_v1"
    )

    def __init__(
        self,
        n_bands,
        context_size=41,
        output_tile=9,
        width=80,
        anchor_width=32,
        dilations=DILATIONS,
    ):
        super().__init__()

        self.n_bands = int(n_bands)
        self.context_size = int(context_size)
        self.output_tile = int(output_tile)
        self.width = int(width)

        # Corrupted normalized HSI + per-band mask + collapsed missing fraction.
        in_channels = (
            2 * self.n_bands
            + 1
        )

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                width,
                kernel_size=1,
            ),
            nn.GroupNorm(
                8,
                width,
            ),
            nn.GELU(),
        )

        self.context_blocks = nn.Sequential(
            *[
                DilatedResidualBlock(
                    width,
                    d,
                )
                for d in dilations
            ]
        )

        self.context_norm = nn.GroupNorm(
            8,
            width,
        )

        # Anchor information is compressed only on the central output tile.
        self.anchor_encoder = nn.Sequential(
            nn.Conv2d(
                2 * self.n_bands,
                anchor_width,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                anchor_width,
                anchor_width,
                kernel_size=1,
            ),
            nn.GELU(),
        )

        head_in = (
            width
            + anchor_width
        )

        self.head_fusion = nn.Sequential(
            nn.Conv2d(
                head_in,
                width,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                width,
            ),
            nn.GELU(),
            nn.Conv2d(
                width,
                width,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                width,
            ),
            nn.GELU(),
        )

        # Per-band, per-pixel heads over the whole 9x9 target tile.
        self.residual_head = nn.Conv2d(
            width,
            self.n_bands,
            kernel_size=1,
        )

        self.gate_head = nn.Conv2d(
            width,
            self.n_bands,
            kernel_size=1,
        )

        self.fallback_head = nn.Conv2d(
            width,
            self.n_bands,
            kernel_size=1,
        )

        # Conservative start: initially favor the spectral anchor.
        nn.init.constant_(
            self.gate_head.bias,
            -2.5,
        )

    def _center_crop(
        self,
        tensor,
        size,
    ):
        h = tensor.shape[-2]
        w = tensor.shape[-1]

        y0 = (
            h - int(size)
        ) // 2

        x0 = (
            w - int(size)
        ) // 2

        return tensor[
            :,
            :,
            y0:y0 + int(size),
            x0:x0 + int(size),
        ]

    def forward(
        self,
        cube,
        mask,
        anchor,
        anchor_available,
    ):
        missing_fraction = mask.mean(
            dim=1,
            keepdim=True,
        )

        x = torch.cat(
            [
                cube,
                mask,
                missing_fraction,
            ],
            dim=1,
        )

        x = self.stem(x)
        x = self.context_blocks(x)
        x = F.gelu(
            self.context_norm(x)
        )

        tile_features = self._center_crop(
            x,
            self.output_tile,
        )

        anchor_features = self.anchor_encoder(
            torch.cat(
                [
                    anchor,
                    anchor_available,
                ],
                dim=1,
            )
        )

        fused = self.head_fusion(
            torch.cat(
                [
                    tile_features,
                    anchor_features,
                ],
                dim=1,
            )
        )

        residual = self.residual_head(
            fused
        )

        gate_logits = self.gate_head(
            fused
        )

        gate = torch.sigmoid(
            gate_logits
        )

        fallback = self.fallback_head(
            fused
        )

        anchored = (
            anchor
            + gate * residual
        )

        prediction = torch.where(
            anchor_available > 0.5,
            anchored,
            fallback,
        )

        return {
            "prediction": prediction,
            "residual": residual,
            "gate_logits": gate_logits,
            "gate": gate,
            "fallback": fallback,
        }


# =============================================================================
# DIRECT STRUCTURE LOSSES
# =============================================================================

def spatial_gradient_loss(
    full_pred,
    target,
    use,
    valid,
):
    # Horizontal
    pdx = (
        full_pred[:, :, :, 1:]
        - full_pred[:, :, :, :-1]
    )

    tdx = (
        target[:, :, :, 1:]
        - target[:, :, :, :-1]
    )

    x_mask = (
        (
            use[:, :, :, 1:]
            | use[:, :, :, :-1]
        )
        & valid[:, :, :, 1:]
        & valid[:, :, :, :-1]
    )

    loss_x = masked_mean(
        F.smooth_l1_loss(
            pdx,
            tdx,
            reduction="none",
            beta=0.20,
        ),
        x_mask,
    )

    # Vertical
    pdy = (
        full_pred[:, :, 1:, :]
        - full_pred[:, :, :-1, :]
    )

    tdy = (
        target[:, :, 1:, :]
        - target[:, :, :-1, :]
    )

    y_mask = (
        (
            use[:, :, 1:, :]
            | use[:, :, :-1, :]
        )
        & valid[:, :, 1:, :]
        & valid[:, :, :-1, :]
    )

    loss_y = masked_mean(
        F.smooth_l1_loss(
            pdy,
            tdy,
            reduction="none",
            beta=0.20,
        ),
        y_mask,
    )

    return 0.5 * (
        loss_x + loss_y
    )


def spectral_d1_loss(
    full_pred,
    target,
    use,
    valid,
):
    pred_d1 = (
        full_pred[:, 1:, :, :]
        - full_pred[:, :-1, :, :]
    )

    target_d1 = (
        target[:, 1:, :, :]
        - target[:, :-1, :, :]
    )

    d1_mask = (
        (
            use[:, 1:, :, :]
            | use[:, :-1, :, :]
        )
        & valid[:, 1:, :, :]
        & valid[:, :-1, :, :]
    )

    return masked_mean(
        F.smooth_l1_loss(
            pred_d1,
            target_d1,
            reduction="none",
            beta=0.20,
        ),
        d1_mask,
    )


def local_ssim_loss(
    full_pred,
    target,
    use,
):
    """
    Direct local SSIM on reconstructed tile.

    Observed positions in full_pred are replaced by GT before this call, so
    only synthetic corruption changes the local structure.

    Computed per spectral band with a 3x3 local window.
    """
    n, b, h, w = full_pred.shape

    pred_2d = full_pred.reshape(
        n * b,
        1,
        h,
        w,
    )

    target_2d = target.reshape(
        n * b,
        1,
        h,
        w,
    )

    mu_x = F.avg_pool2d(
        pred_2d,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    mu_y = F.avg_pool2d(
        target_2d,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    sigma_x = (
        F.avg_pool2d(
            pred_2d * pred_2d,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        - mu_x * mu_x
    ).clamp_min(0.0)

    sigma_y = (
        F.avg_pool2d(
            target_2d * target_2d,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        - mu_y * mu_y
    ).clamp_min(0.0)

    sigma_xy = (
        F.avg_pool2d(
            pred_2d * target_2d,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        - mu_x * mu_y
    )

    # Appropriate for normalized reflectance rather than [0,1] images.
    c1 = 0.01
    c2 = 0.09

    ssim = (
        (
            2.0 * mu_x * mu_y
            + c1
        )
        * (
            2.0 * sigma_xy
            + c2
        )
        / (
            (
                mu_x * mu_x
                + mu_y * mu_y
                + c1
            )
            * (
                sigma_x
                + sigma_y
                + c2
            )
            + 1e-8
        )
    )

    ssim = ssim.reshape(
        n,
        b,
        h,
        w,
    )

    # Windows near a missing pixel matter. Dilate the synthetic target mask.
    missing_window = F.max_pool2d(
        use.float().reshape(
            n * b,
            1,
            h,
            w,
        ),
        kernel_size=3,
        stride=1,
        padding=1,
    ).reshape(
        n,
        b,
        h,
        w,
    ) > 0.5

    return masked_mean(
        1.0 - ssim,
        missing_window,
    )


# =============================================================================
# TOTAL LOSS
# =============================================================================

def deep_v3_loss(
    output,
    batch,
):
    pred = output["prediction"]
    target = batch["target"]

    valid = (
        batch["target_valid"]
        > 0.5
    )

    missing = (
        batch["target_mask"]
        > 0.5
    )

    use = (
        missing
        & valid
    )

    anchor_available = (
        batch["anchor_available"]
        > 0.5
    )

    anchored_use = (
        use
        & anchor_available
    )

    no_anchor_use = (
        use
        & ~anchor_available
    )

    pixel_error = F.smooth_l1_loss(
        pred,
        target,
        reduction="none",
        beta=0.20,
    )

    anchored_pixel = masked_mean(
        pixel_error,
        anchored_use,
    )

    no_anchor_pixel = masked_mean(
        pixel_error,
        no_anchor_use,
    )

    # Build a complete tile where observed values are exact GT.
    # Structural losses therefore only react to reconstruction differences.
    full_pred = torch.where(
        use,
        pred,
        target,
    )

    spatial_grad = spatial_gradient_loss(
        full_pred,
        target,
        use,
        valid,
    )

    ssim = local_ssim_loss(
        full_pred,
        target,
        use,
    )

    spectral_d1 = spectral_d1_loss(
        full_pred,
        target,
        use,
        valid,
    )

    # Conservative anchor supervision.
    anchor_error = (
        batch["anchor"]
        - target
    ).abs().detach()

    gate_target = torch.sigmoid(
        (
            anchor_error
            - GATE_ERROR_CENTER
        )
        / max(
            GATE_ERROR_TEMPERATURE,
            1e-5,
        )
    )

    gate_supervision = masked_mean(
        F.binary_cross_entropy_with_logits(
            output["gate_logits"],
            gate_target,
            reduction="none",
        ),
        anchored_use,
    )

    good_anchor_weight = torch.exp(
        -anchor_error
        / ANCHOR_SAFETY_SCALE
    )

    anchor_safety = masked_mean(
        (
            pred
            - batch["anchor"]
        ).abs()
        * good_anchor_weight,
        anchored_use,
    )

    gate_prior = masked_mean(
        output["gate"],
        anchored_use,
    )

    total = (
        W_ANCHORED_PIXEL
        * anchored_pixel

        + W_NO_ANCHOR_PIXEL
        * no_anchor_pixel

        + W_SPATIAL_GRADIENT
        * spatial_grad

        + W_LOCAL_SSIM
        * ssim

        + W_SPECTRAL_D1
        * spectral_d1

        + W_GATE_SUPERVISION
        * gate_supervision

        + W_ANCHOR_SAFETY
        * anchor_safety

        + W_GATE_PRIOR
        * gate_prior
    )

    with torch.no_grad():
        anchored_abs = masked_mean(
            (
                pred
                - target
            ).abs(),
            anchored_use,
        )

        no_anchor_abs = masked_mean(
            (
                pred
                - target
            ).abs(),
            no_anchor_use,
        )

        gate_mean = masked_mean(
            output["gate"],
            anchored_use,
        )

    return total, {
        "anchored_pixel": anchored_pixel.detach(),
        "no_anchor_pixel": no_anchor_pixel.detach(),
        "spatial_grad": spatial_grad.detach(),
        "ssim": ssim.detach(),
        "spectral_d1": spectral_d1.detach(),
        "gate_supervision": gate_supervision.detach(),
        "anchor_safety": anchor_safety.detach(),
        "gate_prior": gate_prior.detach(),
        "anchored_abs": anchored_abs.detach(),
        "no_anchor_abs": no_anchor_abs.detach(),
        "gate_mean": gate_mean.detach(),
    }


# =============================================================================
# TRAINING HELPERS
# =============================================================================

def move_batch(
    batch,
    device,
):
    out = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(
                device,
                non_blocking=True,
            )
        else:
            out[key] = value

    return out


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    loss_sum_local = 0.0
    loss_count_local = 0

    anchored_sq_local = 0.0
    anchored_n_local = 0

    no_anchor_sq_local = 0.0
    no_anchor_n_local = 0

    gate_sum_local = 0.0
    gate_n_local = 0

    spatial_grad_sum_local = 0.0
    ssim_sum_local = 0.0
    metric_steps_local = 0

    for batch in loader:
        batch = move_batch(
            batch,
            device,
        )

        with torch.amp.autocast(
            "cuda",
            enabled=True,
        ):
            output = model(
                batch["cube"],
                batch["mask"],
                batch["anchor"],
                batch["anchor_available"],
            )

            loss, parts = deep_v3_loss(
                output,
                batch,
            )

        pred = output[
            "prediction"
        ].float()

        target = batch[
            "target"
        ].float()

        use = (
            (
                batch["target_mask"]
                > 0.5
            )
            & (
                batch["target_valid"]
                > 0.5
            )
        )

        anchor_available = (
            batch["anchor_available"]
            > 0.5
        )

        anchored = (
            use
            & anchor_available
        )

        no_anchor = (
            use
            & ~anchor_available
        )

        err2 = (
            pred
            - target
        ).square()

        if torch.any(anchored):
            anchored_sq_local += float(
                err2[
                    anchored
                ].sum().item()
            )

            anchored_n_local += int(
                anchored.sum().item()
            )

            gate_sum_local += float(
                output["gate"][
                    anchored
                ].sum().item()
            )

            gate_n_local += int(
                anchored.sum().item()
            )

        if torch.any(no_anchor):
            no_anchor_sq_local += float(
                err2[
                    no_anchor
                ].sum().item()
            )

            no_anchor_n_local += int(
                no_anchor.sum().item()
            )

        loss_sum_local += float(
            loss.item()
        )

        loss_count_local += 1

        spatial_grad_sum_local += float(
            parts[
                "spatial_grad"
            ].item()
        )

        ssim_sum_local += float(
            parts[
                "ssim"
            ].item()
        )

        metric_steps_local += 1

    loss_sum = all_reduce_sum(
        loss_sum_local,
        device,
    )

    loss_count = all_reduce_sum(
        loss_count_local,
        device,
    )

    anchored_sq = all_reduce_sum(
        anchored_sq_local,
        device,
    )

    anchored_n = all_reduce_sum(
        anchored_n_local,
        device,
    )

    no_anchor_sq = all_reduce_sum(
        no_anchor_sq_local,
        device,
    )

    no_anchor_n = all_reduce_sum(
        no_anchor_n_local,
        device,
    )

    gate_sum = all_reduce_sum(
        gate_sum_local,
        device,
    )

    gate_n = all_reduce_sum(
        gate_n_local,
        device,
    )

    grad_sum = all_reduce_sum(
        spatial_grad_sum_local,
        device,
    )

    ssim_sum = all_reduce_sum(
        ssim_sum_local,
        device,
    )

    metric_steps = all_reduce_sum(
        metric_steps_local,
        device,
    )

    return {
        "loss": (
            loss_sum
            / max(
                1.0,
                loss_count,
            )
        ),
        "anchored_rmse": math.sqrt(
            anchored_sq
            / max(
                1.0,
                anchored_n,
            )
        ),
        "no_anchor_rmse": math.sqrt(
            no_anchor_sq
            / max(
                1.0,
                no_anchor_n,
            )
        ),
        "gate_mean": (
            gate_sum
            / max(
                1.0,
                gate_n,
            )
        ),
        "spatial_grad_loss": (
            grad_sum
            / max(
                1.0,
                metric_steps,
            )
        ),
        "ssim_loss": (
            ssim_sum
            / max(
                1.0,
                metric_steps,
            )
        ),
        "anchored_n": int(
            anchored_n
        ),
        "no_anchor_n": int(
            no_anchor_n
        ),
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_val,
    n_bands,
    wavelengths,
    band_mean,
    band_std,
    world_size,
):
    base_model = (
        model.module
        if isinstance(
            model,
            DDP,
        )
        else model
    )

    payload = {
        "architecture": (
            DeepExpertV3.ARCHITECTURE_VERSION
        ),
        "model_state_dict": (
            base_model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "epoch": int(epoch),
        "best_val_loss": float(best_val),
        "n_bands": int(n_bands),
        "context_size": int(CONTEXT_SIZE),
        "output_tile": int(OUTPUT_TILE),
        "world_size": int(world_size),
        "batch_size_per_gpu": int(
            BATCH_SIZE_PER_GPU
        ),
        "global_batch_size": int(
            BATCH_SIZE_PER_GPU
            * world_size
        ),
        "wavelengths_nm": np.asarray(
            wavelengths,
            dtype=np.float32,
        ),
        "band_mean": np.asarray(
            band_mean,
            dtype=np.float32,
        ),
        "band_std": np.asarray(
            band_std,
            dtype=np.float32,
        ),
        "settings": {
            "width": int(
                MODEL_WIDTH
            ),
            "anchor_width": int(
                ANCHOR_WIDTH
            ),
            "dilations": list(
                DILATIONS
            ),
        },
    }

    torch.save(
        payload,
        path,
    )


# =============================================================================
# DDP WORKER
# =============================================================================

def worker_main():
    rank = None
    local_rank = None

    try:
        (
            rank,
            local_rank,
            world_size,
            device,
        ) = setup_ddp()

        main_process = (
            rank == 0
        )

        set_seed(SEED)

        if main_process:
            print("")
            print("=" * 104)
            print("TPSSR — TRAIN DEEP EXPERT V3 | SWIR1 | 2-GPU DDP")
            print("=" * 104)
            print("Architecture:")
            print(
                f"  context={CONTEXT_SIZE}x{CONTEXT_SIZE} | "
                f"joint output={OUTPUT_TILE}x{OUTPUT_TILE}"
            )
            print(
                "  direct pixel + spatial-gradient + SSIM + spectral losses"
            )
            print(
                "  gated spectral anchor when available; "
                "direct context fallback otherwise"
            )
            print("")
            print(
                "World size:",
                world_size,
            )
            print(
                "Batch/GPU:",
                BATCH_SIZE_PER_GPU,
            )
            print(
                "Effective batch:",
                BATCH_SIZE_PER_GPU
                * world_size,
            )
            print("")

        train_cubes, val_cubes, wavelengths = (
            load_global_training_data()
        )

        wavelengths = np.asarray(
            wavelengths,
            dtype=np.float32,
        )

        if len(train_cubes) == 0:
            raise RuntimeError(
                "No training cubes were returned."
            )

        if len(val_cubes) == 0:
            raise RuntimeError(
                "No validation cubes were returned."
            )

        n_bands = int(
            np.asarray(
                train_cubes[0]
            ).shape[-1]
        )

        for cube in list(
            train_cubes
        ) + list(
            val_cubes
        ):
            if (
                np.asarray(
                    cube
                ).shape[-1]
                != n_bands
            ):
                raise RuntimeError(
                    "Training/validation band counts do not match."
                )

        band_mean, band_std = band_stats(
            train_cubes
        )

        band_mean = np.asarray(
            band_mean,
            dtype=np.float32,
        )

        band_std = np.asarray(
            band_std,
            dtype=np.float32,
        )

        band_std = np.maximum(
            band_std,
            1e-6,
        )

        model_dir = (
            config.MODELS_DIR
            / "texture_restoration_router_rf_v2"
            / MODEL_SUBDIR
        )

        if main_process:
            model_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.savez(
                model_dir
                / "normalization.npz",
                band_mean=band_mean,
                band_std=band_std,
                wavelengths_nm=wavelengths,
            )

            print(
                "Training cubes:",
                len(train_cubes),
            )
            print(
                "Validation cubes:",
                len(val_cubes),
            )
            print(
                "Bands:",
                n_bands,
            )
            print(
                "Train samples/epoch:",
                TRAIN_SAMPLES_PER_EPOCH,
            )
            print(
                "Validation samples:",
                VAL_SAMPLES,
            )
            print(
                "Epochs:",
                EPOCHS,
            )
            print(
                "Model folder:",
                model_dir,
            )
            print("")

        ddp_barrier(
            local_rank
        )

        train_ds = DeepV3SyntheticDataset(
            cubes=train_cubes,
            wavelengths=wavelengths,
            band_mean=band_mean,
            band_std=band_std,
            samples=TRAIN_SAMPLES_PER_EPOCH,
            context_size=CONTEXT_SIZE,
            output_tile=OUTPUT_TILE,
            seed=SEED + 101,
            training=True,
        )

        val_ds = DeepV3SyntheticDataset(
            cubes=val_cubes,
            wavelengths=wavelengths,
            band_mean=band_mean,
            band_std=band_std,
            samples=VAL_SAMPLES,
            context_size=CONTEXT_SIZE,
            output_tile=OUTPUT_TILE,
            seed=SEED + 202,
            training=False,
        )

        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=SEED + 303,
            drop_last=False,
        )

        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE_PER_GPU,
            sampler=train_sampler,
            shuffle=False,
            num_workers=NUM_WORKERS_PER_GPU,
            pin_memory=True,
            persistent_workers=(
                NUM_WORKERS_PER_GPU
                > 0
            ),
            drop_last=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE_PER_GPU,
            sampler=val_sampler,
            shuffle=False,
            num_workers=NUM_WORKERS_PER_GPU,
            pin_memory=True,
            persistent_workers=(
                NUM_WORKERS_PER_GPU
                > 0
            ),
            drop_last=False,
        )

        model = DeepExpertV3(
            n_bands=n_bands,
            context_size=CONTEXT_SIZE,
            output_tile=OUTPUT_TILE,
            width=MODEL_WIDTH,
            anchor_width=ANCHOR_WIDTH,
            dilations=DILATIONS,
        ).to(device)

        model = DDP(
            model,
            device_ids=[
                local_rank
            ],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(
                    1,
                    EPOCHS,
                ),
                eta_min=(
                    LEARNING_RATE
                    * 0.10
                ),
            )
        )

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )

        param_count = sum(
            p.numel()
            for p in model.module.parameters()
        )

        if main_process:
            print(
                f"Deep-v3 parameters: "
                f"{param_count / 1e6:.3f} M"
            )
            print("")

        best_val = float("inf")
        history = []

        for epoch in range(
            1,
            EPOCHS + 1,
        ):
            train_ds.set_epoch(
                epoch
            )

            train_sampler.set_epoch(
                epoch
            )

            model.train()

            local = {
                "loss": 0.0,
                "anchored_pixel": 0.0,
                "no_anchor_pixel": 0.0,
                "spatial_grad": 0.0,
                "ssim": 0.0,
                "spectral_d1": 0.0,
                "gate": 0.0,
                "steps": 0.0,
            }

            progress = tqdm(
                train_loader,
                desc=(
                    f"Epoch "
                    f"{epoch:02d}/{EPOCHS}"
                ),
                dynamic_ncols=True,
                disable=not main_process,
            )

            for batch in progress:
                batch = move_batch(
                    batch,
                    device,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                with torch.amp.autocast(
                    "cuda",
                    enabled=True,
                ):
                    output = model(
                        batch["cube"],
                        batch["mask"],
                        batch["anchor"],
                        batch["anchor_available"],
                    )

                    loss, parts = deep_v3_loss(
                        output,
                        batch,
                    )

                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                local["loss"] += float(
                    loss.item()
                )

                local[
                    "anchored_pixel"
                ] += float(
                    parts[
                        "anchored_pixel"
                    ].item()
                )

                local[
                    "no_anchor_pixel"
                ] += float(
                    parts[
                        "no_anchor_pixel"
                    ].item()
                )

                local[
                    "spatial_grad"
                ] += float(
                    parts[
                        "spatial_grad"
                    ].item()
                )

                local[
                    "ssim"
                ] += float(
                    parts[
                        "ssim"
                    ].item()
                )

                local[
                    "spectral_d1"
                ] += float(
                    parts[
                        "spectral_d1"
                    ].item()
                )

                local[
                    "gate"
                ] += float(
                    parts[
                        "gate_mean"
                    ].item()
                )

                local["steps"] += 1.0

                if main_process:
                    progress.set_postfix(
                        loss=(
                            f"{loss.item():.4f}"
                        ),
                        grad=(
                            f"{parts['spatial_grad'].item():.3f}"
                        ),
                        ssim=(
                            f"{parts['ssim'].item():.3f}"
                        ),
                        gate=(
                            f"{parts['gate_mean'].item():.3f}"
                        ),
                    )

            global_steps = all_reduce_sum(
                local["steps"],
                device,
            )

            train_means = {}

            for key in (
                "loss",
                "anchored_pixel",
                "no_anchor_pixel",
                "spatial_grad",
                "ssim",
                "spectral_d1",
                "gate",
            ):
                train_means[key] = (
                    all_reduce_sum(
                        local[key],
                        device,
                    )
                    / max(
                        1.0,
                        global_steps,
                    )
                )

            val = evaluate(
                model,
                val_loader,
                device,
            )

            scheduler.step()

            is_new_best = (
                val["loss"]
                < best_val
            )

            if is_new_best:
                best_val = float(
                    val["loss"]
                )

            if main_process:
                row = {
                    "epoch": int(epoch),
                    "lr": float(
                        optimizer.param_groups[
                            0
                        ]["lr"]
                    ),
                    "world_size": int(
                        world_size
                    ),
                    "batch_size_per_gpu": int(
                        BATCH_SIZE_PER_GPU
                    ),
                    "global_batch_size": int(
                        BATCH_SIZE_PER_GPU
                        * world_size
                    ),
                    "train_loss": train_means[
                        "loss"
                    ],
                    "train_anchored_pixel": train_means[
                        "anchored_pixel"
                    ],
                    "train_no_anchor_pixel": train_means[
                        "no_anchor_pixel"
                    ],
                    "train_spatial_grad": train_means[
                        "spatial_grad"
                    ],
                    "train_ssim_loss": train_means[
                        "ssim"
                    ],
                    "train_spectral_d1": train_means[
                        "spectral_d1"
                    ],
                    "train_gate_mean": train_means[
                        "gate"
                    ],
                    "val_loss": val[
                        "loss"
                    ],
                    "val_anchored_rmse_norm": val[
                        "anchored_rmse"
                    ],
                    "val_no_anchor_rmse_norm": val[
                        "no_anchor_rmse"
                    ],
                    "val_gate_mean": val[
                        "gate_mean"
                    ],
                    "val_spatial_grad_loss": val[
                        "spatial_grad_loss"
                    ],
                    "val_ssim_loss": val[
                        "ssim_loss"
                    ],
                    "val_anchored_n": val[
                        "anchored_n"
                    ],
                    "val_no_anchor_n": val[
                        "no_anchor_n"
                    ],
                }

                history.append(
                    row
                )

                print("")
                print(
                    f"Epoch {epoch:02d}: "
                    f"train={train_means['loss']:.5f} | "
                    f"val={val['loss']:.5f} | "
                    f"anchor RMSE(norm)={val['anchored_rmse']:.4f} | "
                    f"no-anchor RMSE(norm)={val['no_anchor_rmse']:.4f} | "
                    f"grad={val['spatial_grad_loss']:.4f} | "
                    f"SSIMloss={val['ssim_loss']:.4f} | "
                    f"gate={val['gate_mean']:.3f}"
                )

                save_checkpoint(
                    model_dir
                    / LAST_MODEL,
                    model,
                    optimizer,
                    epoch,
                    best_val,
                    n_bands,
                    wavelengths,
                    band_mean,
                    band_std,
                    world_size,
                )

                if is_new_best:
                    save_checkpoint(
                        model_dir
                        / BEST_MODEL,
                        model,
                        optimizer,
                        epoch,
                        best_val,
                        n_bands,
                        wavelengths,
                        band_mean,
                        band_std,
                        world_size,
                    )

                    print(
                        "  -> NEW BEST MODEL"
                    )

                with (
                    model_dir
                    / HISTORY_CSV
                ).open(
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=list(
                            history[0].keys()
                        ),
                    )

                    writer.writeheader()
                    writer.writerows(
                        history
                    )

                print("")

            ddp_barrier(
                local_rank
            )

        if main_process:
            best_row = min(
                history,
                key=lambda r: r[
                    "val_loss"
                ],
            )

            summary = {
                "architecture": (
                    DeepExpertV3.ARCHITECTURE_VERSION
                ),
                "purpose": (
                    "SWIR1 development fold; joint patch Deep Expert V3"
                ),
                "world_size": int(
                    world_size
                ),
                "batch_size_per_gpu": int(
                    BATCH_SIZE_PER_GPU
                ),
                "global_batch_size": int(
                    BATCH_SIZE_PER_GPU
                    * world_size
                ),
                "training_cubes": int(
                    len(train_cubes)
                ),
                "validation_cubes": int(
                    len(val_cubes)
                ),
                "n_bands": int(
                    n_bands
                ),
                "context_size": int(
                    CONTEXT_SIZE
                ),
                "output_tile": int(
                    OUTPUT_TILE
                ),
                "parameters": int(
                    param_count
                ),
                "epochs": int(
                    EPOCHS
                ),
                "best_epoch": int(
                    best_row["epoch"]
                ),
                "best_val_loss": float(
                    best_row["val_loss"]
                ),
                "best_val_anchored_rmse_norm": float(
                    best_row[
                        "val_anchored_rmse_norm"
                    ]
                ),
                "best_val_no_anchor_rmse_norm": float(
                    best_row[
                        "val_no_anchor_rmse_norm"
                    ]
                ),
                "best_val_spatial_grad_loss": float(
                    best_row[
                        "val_spatial_grad_loss"
                    ]
                ),
                "best_val_ssim_loss": float(
                    best_row[
                        "val_ssim_loss"
                    ]
                ),
                "best_val_gate_mean": float(
                    best_row[
                        "val_gate_mean"
                    ]
                ),
                "model_path": str(
                    model_dir
                    / BEST_MODEL
                ),
            }

            (
                model_dir
                / SUMMARY_TXT
            ).write_text(
                "\n".join(
                    [
                        "TPSSR — DEEP EXPERT V3 TRAINING SUMMARY",
                        "=" * 84,
                        "",
                        "2-GPU DISTRIBUTED TRAINING",
                        f"  World size: {world_size}",
                        f"  Batch/GPU: {BATCH_SIZE_PER_GPU}",
                        (
                            "  Effective global batch: "
                            f"{BATCH_SIZE_PER_GPU * world_size}"
                        ),
                        "",
                        "SWIR1 DEVELOPMENT / LOIO POLICY",
                        "  Held-out test: SWIR1",
                        (
                            "  Training/validation data came only from "
                            "load_global_training_data()."
                        ),
                        "",
                        (
                            "Architecture: "
                            f"{summary['architecture']}"
                        ),
                        (
                            "Context: "
                            f"{CONTEXT_SIZE}x{CONTEXT_SIZE}"
                        ),
                        (
                            "Joint output tile: "
                            f"{OUTPUT_TILE}x{OUTPUT_TILE}"
                        ),
                        (
                            "Parameters: "
                            f"{param_count}"
                        ),
                        "",
                        "DIRECT RECONSTRUCTION LOSSES",
                        "  pixel fidelity",
                        "  spatial gradient consistency",
                        "  local SSIM / structure",
                        "  spectral derivative consistency",
                        "  conservative gated spectral anchor",
                        "",
                        (
                            "Best epoch: "
                            f"{summary['best_epoch']}"
                        ),
                        (
                            "Best val loss: "
                            f"{summary['best_val_loss']:.8f}"
                        ),
                        (
                            "Best anchored RMSE (normalized): "
                            f"{summary['best_val_anchored_rmse_norm']:.6f}"
                        ),
                        (
                            "Best no-anchor RMSE (normalized): "
                            f"{summary['best_val_no_anchor_rmse_norm']:.6f}"
                        ),
                        (
                            "Best spatial-gradient loss: "
                            f"{summary['best_val_spatial_grad_loss']:.6f}"
                        ),
                        (
                            "Best SSIM loss: "
                            f"{summary['best_val_ssim_loss']:.6f}"
                        ),
                        (
                            "Best mean learned gate: "
                            f"{summary['best_val_gate_mean']:.6f}"
                        ),
                        "",
                        (
                            "Best model: "
                            f"{summary['model_path']}"
                        ),
                        "",
                        "NEXT STEP",
                        (
                            "  Run a direct held-out SWIR1 comparison:"
                        ),
                        (
                            "  Spectral vs Current Deep vs Deep-v3."
                        ),
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with (
                model_dir
                / "training_summary.json"
            ).open(
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    summary,
                    f,
                    indent=2,
                )

            print("=" * 104)
            print(
                "DEEP-V3 TRAINING COMPLETE"
            )
            print("=" * 104)
            print(
                "Best model:"
            )
            print(
                model_dir
                / BEST_MODEL
            )
            print("")
            print(
                "Send me:"
            )
            print(
                "  1. TRAINING_SUMMARY.txt"
            )
            print(
                "  2. training_history.csv"
            )

        ddp_barrier(
            local_rank
        )

    finally:
        if (
            rank is not None
            and dist.is_available()
            and dist.is_initialized()
        ):
            cleanup_ddp(
                local_rank
            )


# =============================================================================
# ONE-FILE LAUNCHER
# =============================================================================

def launcher_main():
    # torch.distributed.run relaunches the same file with WORLD_SIZE > 1.
    if int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    ) > 1:
        worker_main()
        return

    parser = argparse.ArgumentParser(
        description=(
            "Prepare the SWIR1 LOIO fold and train Deep Expert V3 "
            "with two-GPU DDP from this single file."
        )
    )

    parser.add_argument(
        "--gpus",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--prepare",
        action="store_true",
    )

    args = parser.parse_args()

    if args.gpus != 2:
        raise ValueError(
            "Current V3 development setup expects --gpus 2."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"Two GPUs requested but only "
            f"{torch.cuda.device_count()} are visible."
        )

    print("")
    print("=" * 104)
    print(
        "TPSSR — SWIR1 DEEP EXPERT V3 | ONE-FILE 2-GPU TRAINING"
    )
    print("=" * 104)
    print(
        "Experiment:",
        config.EXPERIMENT_NAME,
    )
    print(
        "Train datasets:",
        ", ".join(
            config.TRAIN_DATASET_FOLDERS
        ),
    )
    print(
        "Held-out test:",
        config.TEST_DATASET_FOLDER,
    )
    print(
        "GPUs:",
        args.gpus,
    )
    print("=" * 104)


    expected_train = {
        str(x).upper()
        for x in getattr(config, "ALL_SENSOR_DATASETS", [])
        if str(x).upper() != str(config.TEST_DATASET_FOLDER).upper()
    }
    actual_train = {str(x).upper() for x in config.TRAIN_DATASET_FOLDERS}
    if len(expected_train) != 4 or actual_train != expected_train:
        raise RuntimeError(
            f"Invalid LOIO fold: held-out={config.TEST_DATASET_FOLDER}, "
            f"train={config.TRAIN_DATASET_FOLDERS}"
        )

    if args.prepare:
        prepare_script = (
            PROJECT_ROOT
            / "data_preparation"
            / "prepare_dataset.py"
        )

        if not prepare_script.exists():
            raise FileNotFoundError(
                f"prepare_dataset.py not found:\n{prepare_script}"
            )

        print("")
        print(
            "[Preparation] Building current SWIR1 LOIO fold..."
        )

        proc = subprocess.run(
            [
                sys.executable,
                str(prepare_script),
            ],
            cwd=str(
                PROJECT_ROOT
            ),
            env=os.environ.copy(),
        )

        if proc.returncode != 0:
            raise SystemExit(
                proc.returncode
            )

    print("")
    print(
        "[Deep-v3] Starting 2-GPU DDP training..."
    )

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={int(args.gpus)}",
        str(HERE),
    ]

    print("")
    print(
        "RUNNING:",
        " ".join(command),
    )
    print("")

    proc = subprocess.run(
        command,
        cwd=str(
            PROJECT_ROOT
        ),
        env=os.environ.copy(),
    )

    if proc.returncode != 0:
        raise SystemExit(
            proc.returncode
        )

    print("")
    print("=" * 104)
    print(
        "SWIR1 DEEP EXPERT V3 TRAINING FINISHED"
    )
    print("=" * 104)


if __name__ == "__main__":
    launcher_main()
