#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train the geometry-guidance network used in TPSSR spatial completion.

The model learns correspondence and structural cues from measured context.
Target reflectance is used for supervised loss only, never as an inference input.
Training uses the configured training acquisitions and two CUDA devices.
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
from datetime import timedelta

HERE = Path(__file__).resolve()

PROJECT_ROOT = next(
    parent
    for parent in HERE.parents
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

from scipy.ndimage import distance_transform_edt, binary_dilation, gaussian_filter

import config
from src.evaluation_utils import load_global_training_data


# =============================================================================
# SETTINGS
# =============================================================================

PATCH_SIZE = 65

# Axis-aligned whole-spectrum holes, matching the real synthetic benchmark
# family. Structure-aware placement makes many of them cut across real lines.
MASK_SHAPES = (
    (5, 15),
    (9, 15),
    (15, 5),
    (15, 9),
    (15, 15),
    (11, 11),
    (9, 9),
)

# Sampling policy.
P_STRUCTURE_CENTER = 0.92
P_CROSSING_CENTER = 0.78
P_DIAGONAL_CENTER = 0.70

TRAIN_SAMPLES_PER_EPOCH = 2200
VAL_SAMPLES = 550

STRUCTURE_EPOCHS = 6
JOINT_EPOCHS = 10

BATCH_SIZE_PER_GPU = 1
NUM_WORKERS_PER_GPU = 2

STRUCTURE_LR = 1.8e-4
JOINT_STRUCTURE_LR = 6.0e-5
JOINT_RECON_LR = 1.5e-4

WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 2.0

SPECTRAL_WIDTH = 32
BASE_WIDTH = 56
ATTENTION_HEADS = 4

# Phase-A structure losses.
W_STRUCT_EDGE_BCE = 1.00
W_STRUCT_EDGE_DICE = 0.65
W_STRUCT_ORIENTATION = 0.90
W_STRUCT_ENDPOINT = 0.45

# Phase-B reconstruction losses.
W_PIXEL = 1.00
W_EDGE_PIXEL = 0.40
W_GRADIENT = 0.32
W_RECON_ORIENTATION = 0.38
W_DIAGONAL = 0.28
W_SSIM = 0.18
W_BOUNDARY = 0.30
W_LAPLACIAN = 0.06

# Geometry-specific anti-blur / radiometric-structure terms.
W_LOCAL_CONTRAST = 0.42
W_POLARITY = 0.34
W_PROFILE = 0.30
W_COARSE = 0.12

# Stage-A signed contrast supervision.
W_STRUCT_CONTRAST = 0.55

# Detail residual is intentionally limited so the final refiner cannot
# overwrite the learned structure completion with an unconstrained texture map.
DETAIL_RESIDUAL_SCALE = 0.45

# Keep the intermediate representation meaningful during joint training.
W_STRUCTURE_DURING_JOINT = 0.50

MODEL_SUBDIR = f"geometry_{str(config.TEST_DATASET_FOLDER).lower()}"

BEST_STRUCTURE_MODEL = "best_structure_model.pt"
BEST_MODEL = "best_model.pt"
LAST_MODEL = "last_model.pt"

STRUCTURE_HISTORY_CSV = "structure_history.csv"
JOINT_HISTORY_CSV = "training_history.csv"
SUMMARY_TXT = "TRAINING_SUMMARY.txt"

# Normalization is now derived from the configured TRAIN acquisitions only.
# The model architecture, losses and checkpoint schema are unchanged.
TRAIN_ONLY_NORM = Path(config.MODELS_DIR) / "tpssr" / (
    f"geometry_train_stats_{str(config.TEST_DATASET_FOLDER).lower()}.npz"
)

SEED = int(getattr(config, "SEED", 1234)) + 27001


# =============================================================================
# REPRODUCIBILITY / DDP
# =============================================================================

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_ddp():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        if name not in os.environ:
            raise RuntimeError(
                "Missing DDP worker environment. "
                "Launch this file with plain Python and --gpus 2."
            )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if world_size != 2:
        raise RuntimeError(
            f"Geometry is configured for exactly two GPUs; WORLD_SIZE={world_size}."
        )

    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"Two GPUs requested, but only {torch.cuda.device_count()} are visible."
        )

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
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=30),
        )

    return rank, local_rank, world_size, device


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def ddp_barrier(local_rank):
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        dist.barrier(device_ids=[int(local_rank)])
    except TypeError:
        dist.barrier()


def all_reduce_sum(value, device):
    tensor = torch.tensor(
        float(value),
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


# =============================================================================
# NORMALIZATION METADATA
# =============================================================================

def load_normalization_metadata():
    """Read per-band statistics computed exclusively on training acquisitions.

    Run the project-root ``python train.py --preset swir_8x2`` first; it writes
    this small NPZ metadata file before invoking Geometry. No the previous experimental expert model or
    checkpoint is required. Both DDP ranks only READ the precomputed NPZ.
    """
    path = Path(os.environ.get("TPSSR_GEOMETRY_TRAIN_STATS", str(TRAIN_ONLY_NORM)))
    if not path.is_file():
        raise FileNotFoundError(
            "Geometry TRAIN-only normalization metadata not found: " + str(path) +
            ". Run the project-root train.py to create it; the previous experimental expert is not required."
        )
    with np.load(path, allow_pickle=False) as stored:
        required = ("n_bands", "band_mean", "band_std", "wavelengths_nm", "train_datasets")
        missing = [name for name in required if name not in stored]
        if missing:
            raise RuntimeError("Missing TRAIN normalization fields: " + ", ".join(missing))
        names = tuple(str(x) for x in stored["train_datasets"].tolist())
        if names != tuple(config.TRAIN_DATASET_FOLDERS):
            raise RuntimeError(f"Geometry normalization TRAIN SPLIT mismatch: {names}")
        metadata = {
            "n_bands": int(stored["n_bands"]),
            "band_mean": np.asarray(stored["band_mean"], dtype=np.float32).copy(),
            "band_std": np.asarray(stored["band_std"], dtype=np.float32).copy(),
            "wavelengths_nm": np.asarray(stored["wavelengths_nm"], dtype=np.float32).copy(),
        }
    b = metadata["n_bands"]
    if any(metadata[key].shape != (b,) for key in ("band_mean", "band_std", "wavelengths_nm")):
        raise RuntimeError("Geometry normalization dimension mismatch")
    if not np.isfinite(metadata["band_mean"]).all() or not np.isfinite(metadata["band_std"]).all() or (metadata["band_std"] <= 0).any():
        raise RuntimeError("Geometry normalization statistics are invalid")
    return metadata, path


# =============================================================================
# NUMPY STRUCTURE TARGETS
# =============================================================================

def normalize_cube(cube, band_mean, band_std):
    cube = np.asarray(cube, dtype=np.float32)
    finite = np.isfinite(cube)

    safe = np.where(
        finite,
        cube,
        band_mean[None, None, :],
    ).astype(np.float32)

    normalized = (
        safe - band_mean[None, None, :]
    ) / band_std[None, None, :]

    return normalized.astype(np.float32), finite


def structure_targets_numpy(target):
    """
    Soft edge target and undirected orientation representation.

    The orientation is encoded as:
        [cos(2 theta), sin(2 theta)]

    This is preferable to raw theta because a line has the same orientation
    after a 180-degree reversal.
    """
    target = np.asarray(target, dtype=np.float32)

    gy, gx = np.gradient(target)
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)

    finite_mag = mag[np.isfinite(mag)]

    if finite_mag.size:
        scale = float(np.quantile(finite_mag, 0.90))
    else:
        scale = 1.0

    scale = max(scale, 1e-5)

    # Soft edge confidence rather than a brittle binary Canny-like target.
    edge = np.clip(
        mag / scale,
        0.0,
        1.0,
    ).astype(np.float32)

    denom = gx * gx + gy * gy + 1e-8

    cos2 = (
        (gx * gx - gy * gy) / denom
    ).astype(np.float32)

    sin2 = (
        (2.0 * gx * gy) / denom
    ).astype(np.float32)

    # Flat pixels do not have a meaningful orientation.
    flat = edge < 0.05
    cos2[flat] = 0.0
    sin2[flat] = 0.0

    orientation = np.stack(
        [cos2, sin2],
        axis=0,
    ).astype(np.float32)

    return edge, orientation, gx, gy, mag



def signed_contrast_target_numpy(
    target,
):
    """
    Normalized signed local contrast.

    Negative values indicate locally darker-than-background material and
    positive values indicate locally brighter-than-background material.

    This gives Stage A an explicit representation for the user's critical
    failure case: a truly dark lamina must not become a gray average inside the
    hole.
    """
    target = np.asarray(
        target,
        dtype=np.float32,
    )

    background = gaussian_filter(
        target,
        sigma=2.2,
        mode="nearest",
    ).astype(
        np.float32
    )

    residual = (
        target
        - background
    ).astype(
        np.float32
    )

    finite = np.abs(
        residual[
            np.isfinite(
                residual
            )
        ]
    )

    if finite.size:
        scale = float(
            np.quantile(
                finite,
                0.90,
            )
        )
    else:
        scale = 1.0

    scale = max(
        scale,
        1e-5,
    )

    return np.clip(
        residual / scale,
        -1.0,
        1.0,
    ).astype(
        np.float32
    )

def crossing_score_at(
    y,
    x,
    gx,
    gy,
    mag,
    edge_threshold,
    half_extent,
):
    """
    Approximate whether an edge/lamina passes through the future hole.

    We estimate the local line tangent from the image gradient and test whether
    edge energy exists on BOTH sides of the candidate center along that tangent.
    """
    gxx = float(gx[y, x])
    gyy = float(gy[y, x])
    m = float(mag[y, x])

    if not np.isfinite(m) or m <= edge_threshold:
        return 0.0

    # Tangent is perpendicular to the image gradient.
    tx = -gyy / (m + 1e-8)
    ty = gxx / (m + 1e-8)

    distance = float(max(half_extent + 3, 6))

    points = []
    for sign in (-1.0, 1.0):
        yy = int(round(y + sign * distance * ty))
        xx = int(round(x + sign * distance * tx))
        points.append((yy, xx))

    h, w = mag.shape

    if not all(
        1 <= yy < h - 1 and 1 <= xx < w - 1
        for yy, xx in points
    ):
        return 0.0

    side_values = []

    for yy, xx in points:
        # Small neighborhood makes the test robust to one-pixel endpoint shift.
        local = mag[
            yy - 1:yy + 2,
            xx - 1:xx + 2,
        ]
        side_values.append(float(np.max(local)))

    left_score = side_values[0] / (edge_threshold + 1e-8)
    right_score = side_values[1] / (edge_threshold + 1e-8)

    return float(min(left_score, right_score))


def structure_aware_mask(target_norm, full_valid, rng):
    """
    Prefer holes that actually interrupt a structure.

    Priority:
      1) line/edge crossing with context visible on both sides,
      2) diagonal/oblique strong edge,
      3) generic strong edge,
      4) random valid interior point.
    """
    h, w = target_norm.shape

    edge, _orientation, gx, gy, mag = structure_targets_numpy(
        target_norm
    )

    allowed_mag = mag[np.isfinite(mag)]
    edge_threshold = (
        float(np.quantile(allowed_mag, 0.70))
        if allowed_mag.size
        else 0.0
    )

    shape_order = list(range(len(MASK_SHAPES)))
    rng.shuffle(shape_order)

    for shape_index in shape_order:
        mh, mw = MASK_SHAPES[shape_index]
        hh = mh // 2
        hw = mw // 2

        margin = max(18, hh + 6, hw + 6)

        allowed = np.zeros((h, w), dtype=bool)
        allowed[
            margin:h - margin,
            margin:w - margin,
        ] = True

        if not np.any(allowed):
            continue

        candidates = np.argwhere(allowed)

        if len(candidates) == 0:
            continue

        chosen_pool = []
        chosen_type = "random"

        if float(rng.random()) < P_STRUCTURE_CENTER:
            strong = [
                (int(y), int(x))
                for y, x in candidates
                if mag[int(y), int(x)] >= edge_threshold
            ]

            if strong:
                # First try a true structure-crossing center.
                if float(rng.random()) < P_CROSSING_CENTER:
                    crossing = []

                    half_extent = max(hh, hw)

                    # We cap candidate evaluation for speed.
                    sample_ids = rng.permutation(
                        len(strong)
                    )[: min(len(strong), 350)]

                    for idx in sample_ids:
                        y, x = strong[int(idx)]
                        score = crossing_score_at(
                            y=y,
                            x=x,
                            gx=gx,
                            gy=gy,
                            mag=mag,
                            edge_threshold=max(edge_threshold, 1e-6),
                            half_extent=half_extent,
                        )
                        if score >= 0.85:
                            crossing.append(
                                (y, x, score)
                            )

                    if crossing:
                        crossing.sort(
                            key=lambda item: item[2],
                            reverse=True,
                        )
                        top = crossing[
                            : max(
                                1,
                                min(
                                    len(crossing),
                                    80,
                                ),
                            )
                        ]
                        chosen_pool = [
                            (y, x)
                            for y, x, _score in top
                        ]
                        chosen_type = "crossing_structure"

                # Otherwise prefer diagonal/oblique edges.
                if not chosen_pool and float(rng.random()) < P_DIAGONAL_CENTER:
                    diagonal = []

                    for y, x in strong:
                        m = float(mag[y, x]) + 1e-8
                        x_fraction = abs(float(gx[y, x])) / m
                        y_fraction = abs(float(gy[y, x])) / m

                        # Both derivatives substantial -> neither purely
                        # horizontal nor purely vertical.
                        if (
                            x_fraction > 0.22
                            and y_fraction > 0.22
                        ):
                            diagonal.append((y, x))

                    if diagonal:
                        chosen_pool = diagonal
                        chosen_type = "diagonal_edge"

                if not chosen_pool:
                    chosen_pool = strong
                    chosen_type = "strong_edge"

        if not chosen_pool:
            chosen_pool = [
                (int(y), int(x))
                for y, x in candidates
            ]
            chosen_type = "random"

        order = rng.permutation(len(chosen_pool))

        for idx in order[: min(len(order), 300)]:
            cy, cx = chosen_pool[int(idx)]

            y0 = cy - hh
            y1 = cy + hh + 1
            x0 = cx - hw
            x1 = cx + hw + 1

            region_valid = full_valid[y0:y1, x0:x1]

            if (
                region_valid.shape == (mh, mw)
                and np.all(region_valid)
            ):
                mask = np.zeros((h, w), dtype=bool)
                mask[y0:y1, x0:x1] = True

                return mask, chosen_type, (mh, mw)

    return None


# =============================================================================
# DATASET
# =============================================================================

class GeometryLearnedCorrespondenceDataset(Dataset):
    def __init__(
        self,
        cubes,
        band_mean,
        band_std,
        samples,
        seed,
        training,
    ):
        self.cubes = [
            np.asarray(cube)
            for cube in cubes
        ]

        self.band_mean = np.asarray(
            band_mean,
            dtype=np.float32,
        )

        self.band_std = np.maximum(
            np.asarray(
                band_std,
                dtype=np.float32,
            ),
            1e-6,
        )

        self.samples = int(samples)
        self.seed = int(seed)
        self.training = bool(training)
        self.epoch = 0

        self.n_bands = int(
            self.cubes[0].shape[-1]
        )

        for cube in self.cubes:
            if int(cube.shape[-1]) != self.n_bands:
                raise RuntimeError(
                    "All cubes must share the same spectral dimension."
                )

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
            + int(epoch_term) * 1000003
        )

    def __getitem__(self, index):
        rng = self._rng(index)
        p = int(PATCH_SIZE)

        for _attempt in range(120):
            cube_index = int(
                rng.integers(
                    0,
                    len(self.cubes),
                )
            )

            cube = self.cubes[cube_index]

            h, w, b = cube.shape

            if h < p or w < p:
                continue

            y0 = int(
                rng.integers(
                    0,
                    h - p + 1,
                )
            )
            x0 = int(
                rng.integers(
                    0,
                    w - p + 1,
                )
            )

            patch = np.asarray(
                cube[
                    y0:y0 + p,
                    x0:x0 + p,
                    :,
                ],
                dtype=np.float32,
            )

            normalized, finite = normalize_cube(
                patch,
                self.band_mean,
                self.band_std,
            )

            if float(np.mean(finite)) < 0.985:
                continue

            target_band = int(
                rng.integers(0, b)
            )

            target_norm = normalized[
                :,
                :,
                target_band,
            ]

            if float(
                np.mean(
                    finite[
                        :,
                        :,
                        target_band,
                    ]
                )
            ) < 0.995:
                continue

            full_valid = np.all(
                finite,
                axis=2,
            )

            mask_info = structure_aware_mask(
                target_norm,
                full_valid,
                rng,
            )

            if mask_info is None:
                continue

            (
                mask2d,
                center_type,
                mask_shape,
            ) = mask_info

            corrupted = normalized.copy()

            # TRUE whole-spectrum spatial hole:
            # every spectral band is absent at the same spatial target pixels.
            corrupted[mask2d, :] = 0.0

            observed_target = target_norm.copy()
            observed_target[mask2d] = 0.0

            valid_fraction = np.mean(
                finite.astype(np.float32),
                axis=2,
            ).astype(np.float32)

            spatial_valid = (
                (~mask2d)
                & (valid_fraction > 0.95)
            ).astype(np.float32)

            distance = distance_transform_edt(
                mask2d
            ).astype(np.float32)

            if np.max(distance) > 0:
                distance = (
                    distance / np.max(distance)
                ).astype(np.float32)

            band_position = (
                2.0
                * float(target_band)
                / max(b - 1, 1)
                - 1.0
            )

            edge_target, orient_target, _gx, _gy, _mag = (
                structure_targets_numpy(
                    target_norm
                )
            )

            contrast_target = signed_contrast_target_numpy(
                target_norm
            )

            # Stage-A structure supervision is concentrated on the missing
            # region plus a narrow context ring so entrance/exit geometry is
            # explicitly aligned.
            supervision_mask = binary_dilation(
                mask2d,
                iterations=2,
            ).astype(np.float32)

            # Inner boundary of the hole: where a line should enter/leave.
            inner_eroded = ~binary_dilation(
                ~mask2d,
                iterations=1,
            )
            endpoint_mask = (
                mask2d
                & (~inner_eroded)
            ).astype(np.float32)

            return {
                "cube": torch.from_numpy(
                    np.moveaxis(
                        corrupted,
                        -1,
                        0,
                    ).astype(np.float32)
                ),
                "target": torch.from_numpy(
                    target_norm[
                        None,
                        ...,
                    ].astype(np.float32)
                ),
                "observed_target": torch.from_numpy(
                    observed_target[
                        None,
                        ...,
                    ].astype(np.float32)
                ),
                "mask": torch.from_numpy(
                    mask2d[
                        None,
                        ...,
                    ].astype(np.float32)
                ),
                "spatial_valid": torch.from_numpy(
                    spatial_valid[
                        None,
                        ...,
                    ].astype(np.float32)
                ),
                "valid_fraction": torch.from_numpy(
                    valid_fraction[
                        None,
                        ...,
                    ].astype(np.float32)
                ),
                "distance": torch.from_numpy(
                    distance[
                        None,
                        ...,
                    ].astype(np.float32)
                ),
                "edge_target": torch.from_numpy(
                    edge_target[
                        None,
                        ...,
                    ].astype(np.float32)
                ),
                "orientation_target": torch.from_numpy(
                    orient_target.astype(
                        np.float32
                    )
                ),
                "contrast_target": torch.from_numpy(
                    contrast_target[
                        None,
                        ...,
                    ].astype(np.float32)
                ),
                "structure_supervision": torch.from_numpy(
                    supervision_mask[
                        None,
                        ...,
                    ].astype(np.float32)
                ),
                "endpoint_mask": torch.from_numpy(
                    endpoint_mask[
                        None,
                        ...,
                    ].astype(np.float32)
                ),
                "band": torch.tensor(
                    target_band,
                    dtype=torch.long,
                ),
                "band_position": torch.tensor(
                    band_position,
                    dtype=torch.float32,
                ),
                "center_type": center_type,
                "mask_h": torch.tensor(
                    int(mask_shape[0]),
                    dtype=torch.long,
                ),
                "mask_w": torch.tensor(
                    int(mask_shape[1]),
                    dtype=torch.long,
                ),
            }

        raise RuntimeError(
            "Could not sample a valid Geometry learned-correspondence patch after 120 attempts."
        )


# =============================================================================
# NETWORK BUILDING BLOCKS
# =============================================================================

def choose_groups(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class PartialConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super().__init__()

        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        self.input_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )

        self.register_buffer(
            "mask_kernel",
            torch.ones(
                1,
                1,
                self.kernel_size,
                self.kernel_size,
            ),
            persistent=False,
        )

        self.slide_winsize = float(
            self.kernel_size
            * self.kernel_size
        )

    def forward(self, x, mask):
        raw = self.input_conv(
            x * mask
        )

        with torch.no_grad():
            mask_sum = F.conv2d(
                mask,
                self.mask_kernel,
                stride=self.stride,
                padding=self.padding,
            )

            new_mask = (
                mask_sum > 0
            ).to(dtype=x.dtype)

            ratio = (
                self.slide_winsize
                / mask_sum.clamp_min(1.0)
            ) * new_mask

        if self.input_conv.bias is not None:
            bias = self.input_conv.bias.view(
                1,
                -1,
                1,
                1,
            )
            output = (
                (raw - bias)
                * ratio
                + bias
            )
        else:
            output = raw * ratio

        return output * new_mask, new_mask


class PartialBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
    ):
        super().__init__()

        groups = choose_groups(
            out_channels
        )

        self.partial = PartialConv2d(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=stride,
            padding=2,
        )

        self.norm = nn.GroupNorm(
            groups,
            out_channels,
        )

        self.refine = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                groups,
                out_channels,
            ),
            nn.SiLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                groups,
                out_channels,
            ),
        )

    def forward(self, x, mask):
        x, new_mask = self.partial(
            x,
            mask,
        )

        x = F.silu(
            self.norm(x)
        )

        x = F.silu(
            x + 0.5 * self.refine(x)
        )

        return x, new_mask


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dilation=1,
    ):
        super().__init__()

        groups = choose_groups(
            out_channels
        )

        padding = int(dilation)

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(
                groups,
                out_channels,
            ),
            nn.SiLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(
                groups,
                out_channels,
            ),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class BottleneckAttention(nn.Module):
    def __init__(
        self,
        channels,
        heads,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            channels
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=0.0,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(
            channels
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                channels,
                channels * 2,
            ),
            nn.GELU(),
            nn.Linear(
                channels * 2,
                channels,
            ),
        )

    def forward(self, x):
        n, c, h, w = x.shape

        tokens = x.flatten(
            2
        ).transpose(
            1,
            2,
        )

        q = self.norm1(
            tokens
        )

        attended, _ = self.attention(
            q,
            q,
            q,
            need_weights=False,
        )

        tokens = (
            tokens + attended
        )

        tokens = (
            tokens
            + self.mlp(
                self.norm2(
                    tokens
                )
            )
        )

        return tokens.transpose(
            1,
            2,
        ).reshape(
            n,
            c,
            h,
            w,
        )


# =============================================================================
# LEARNED INTACT-CONTEXT CORRESPONDENCE
# =============================================================================

class IntactContextCrossAttention(nn.Module):
    """
    Missing-region queries attend to INTACT context tokens only.

    This is the central Geometry change. No explicit line pair, track, donor patch,
    or hand-written correspondence is supplied. The attention weights are
    learned from training examples.
    """
    def __init__(
        self,
        channels,
        heads,
    ):
        super().__init__()

        self.channels = int(
            channels
        )

        self.query_norm = nn.LayerNorm(
            channels
        )

        self.context_norm = nn.LayerNorm(
            channels
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=0.0,
            batch_first=True,
        )

        self.output_norm = nn.LayerNorm(
            channels
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                channels,
                channels * 2,
            ),
            nn.GELU(),
            nn.Linear(
                channels * 2,
                channels,
            ),
        )

        # Encodes (normalized y, normalized x, hole membership).
        self.position_mlp = nn.Sequential(
            nn.Linear(
                3,
                channels,
            ),
            nn.GELU(),
            nn.Linear(
                channels,
                channels,
            ),
        )

        self.hole_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                channels,
            )
        )

        nn.init.normal_(
            self.hole_token,
            mean=0.0,
            std=0.02,
        )

    def forward(
        self,
        x,
        mask,
        spatial_valid,
    ):
        n, c, h, w = x.shape

        mask_small = F.interpolate(
            mask,
            size=(h, w),
            mode="nearest",
        ).clamp(
            0.0,
            1.0,
        )

        valid_small = F.interpolate(
            spatial_valid,
            size=(h, w),
            mode="nearest",
        ).clamp(
            0.0,
            1.0,
        )

        yy = torch.linspace(
            -1.0,
            1.0,
            h,
            dtype=x.dtype,
            device=x.device,
        )

        xx = torch.linspace(
            -1.0,
            1.0,
            w,
            dtype=x.dtype,
            device=x.device,
        )

        grid_y, grid_x = torch.meshgrid(
            yy,
            xx,
            indexing="ij",
        )

        coords = torch.stack(
            [
                grid_y,
                grid_x,
            ],
            dim=-1,
        ).view(
            1,
            h * w,
            2,
        ).expand(
            n,
            -1,
            -1,
        )

        hole_flat = mask_small.flatten(
            2
        ).transpose(
            1,
            2,
        )

        pos_input = torch.cat(
            [
                coords,
                hole_flat,
            ],
            dim=-1,
        )

        position = self.position_mlp(
            pos_input
        )

        tokens = x.flatten(
            2
        ).transpose(
            1,
            2,
        )

        query = (
            tokens
            + position
            + hole_flat
            * self.hole_token
        )

        context = (
            tokens
            + position
        )

        # Keys/values from the synthetic hole are hidden. This forces missing
        # queries to retrieve information from intact spatial context.
        key_padding_mask = (
            (
                mask_small > 0.5
            )
            | (
                valid_small < 0.5
            )
        ).flatten(
            1
        )

        # Defensive fallback for a pathological sample.
        all_blocked = key_padding_mask.all(
            dim=1
        )

        if bool(
            all_blocked.any()
        ):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[
                all_blocked
            ] = False

        attended, _weights = self.attention(
            self.query_norm(
                query
            ),
            self.context_norm(
                context
            ),
            self.context_norm(
                context
            ),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # Correspondence matters most inside the hole, but a small context
        # update keeps the representation coherent around its boundary.
        gate = (
            0.20
            + 0.80
            * hole_flat
        )

        tokens = (
            tokens
            + gate
            * attended
        )

        tokens = (
            tokens
            + self.ffn(
                self.output_norm(
                    tokens
                )
            )
        )

        return tokens.transpose(
            1,
            2,
        ).reshape(
            n,
            c,
            h,
            w,
        )


# =============================================================================
# STAGE A — LEARNED CORRESPONDENCE + STRUCTURE / CONTRAST COMPLETION
# =============================================================================

class StructureCompletionNet(nn.Module):
    ARCHITECTURE_VERSION = (
        "TPSSR_GEOMETRY_LEARNED_CORRESPONDENCE_STRUCTURE"
    )

    def __init__(
        self,
        n_bands,
        spectral_width=SPECTRAL_WIDTH,
        base_width=BASE_WIDTH,
        attention_heads=ATTENTION_HEADS,
    ):
        super().__init__()

        self.n_bands = int(
            n_bands
        )

        self.spectral_compressor = nn.Sequential(
            nn.Conv2d(
                self.n_bands,
                spectral_width,
                1,
                bias=False,
            ),
            nn.GroupNorm(
                choose_groups(
                    spectral_width
                ),
                spectral_width,
            ),
            nn.SiLU(),
            nn.Conv2d(
                spectral_width,
                spectral_width,
                1,
                bias=False,
            ),
            nn.SiLU(),
        )

        # compressed raw HSI + observed target + valid fraction + mask +
        # distance-to-hole + target-band position.
        input_channels = (
            spectral_width
            + 5
        )

        self.context0 = PartialBlock(
            input_channels,
            base_width,
            stride=1,
        )

        self.context1 = PartialBlock(
            base_width,
            base_width + 32,
            stride=2,
        )

        self.context2 = PartialBlock(
            base_width + 32,
            base_width * 2,
            stride=2,
        )

        bottleneck_channels = (
            base_width * 2
        )

        self.pre_correspondence = nn.Sequential(
            ConvBlock(
                bottleneck_channels,
                bottleneck_channels,
                dilation=2,
            ),
            ConvBlock(
                bottleneck_channels,
                bottleneck_channels,
                dilation=3,
            ),
        )

        self.correspondence = IntactContextCrossAttention(
            bottleneck_channels,
            attention_heads,
        )

        self.post_correspondence = ConvBlock(
            bottleneck_channels,
            bottleneck_channels,
            dilation=1,
        )

        self.decode1 = ConvBlock(
            bottleneck_channels
            + base_width
            + 32,
            base_width + 32,
        )

        self.decode0 = ConvBlock(
            base_width
            + 32
            + base_width,
            base_width,
        )

        # edge logit + cos(2theta) + sin(2theta) + signed local contrast.
        self.structure_head = nn.Sequential(
            nn.Conv2d(
                base_width,
                base_width,
                3,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv2d(
                base_width,
                4,
                1,
            ),
        )

    def forward(
        self,
        corrupted_cube,
        observed_target,
        mask,
        spatial_valid,
        valid_fraction,
        distance,
        band_position,
    ):
        spectral = self.spectral_compressor(
            corrupted_cube
        )

        n, _c, h, w = spectral.shape

        band_map = band_position.view(
            n,
            1,
            1,
            1,
        ).expand(
            -1,
            -1,
            h,
            w,
        )

        x = torch.cat(
            [
                spectral,
                observed_target,
                valid_fraction,
                mask,
                distance,
                band_map,
            ],
            dim=1,
        )

        c0, m0 = self.context0(
            x,
            spatial_valid,
        )

        c1, m1 = self.context1(
            c0,
            m0,
        )

        c2, _m2 = self.context2(
            c1,
            m1,
        )

        z = self.pre_correspondence(
            c2
        )

        z = self.correspondence(
            z,
            mask=mask,
            spatial_valid=spatial_valid,
        )

        z = self.post_correspondence(
            z
        )

        u1 = F.interpolate(
            z,
            size=c1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        u1 = self.decode1(
            torch.cat(
                [
                    u1,
                    c1,
                ],
                dim=1,
            )
        )

        u0 = F.interpolate(
            u1,
            size=c0.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        u0 = self.decode0(
            torch.cat(
                [
                    u0,
                    c0,
                ],
                dim=1,
            )
        )

        raw = self.structure_head(
            u0
        )

        edge_logit = raw[
            :,
            0:1,
        ]

        orient_raw = raw[
            :,
            1:3,
        ]

        orient_norm = torch.sqrt(
            orient_raw.square().sum(
                dim=1,
                keepdim=True,
            )
            + 1e-8
        )

        orientation = (
            orient_raw
            / orient_norm
        )

        contrast_hint = torch.tanh(
            raw[
                :,
                3:4,
            ]
        )

        return {
            "edge_logit": edge_logit,
            "edge_prob": torch.sigmoid(
                edge_logit
            ),
            "orientation": orientation,
            "contrast_hint": contrast_hint,
            "feature": u0,
        }


# =============================================================================
# STAGE B — STRUCTURE-CONDITIONED RADIOMETRIC RECONSTRUCTION / REFINEMENT
# =============================================================================

class StructureConditionedReconNet(nn.Module):
    ARCHITECTURE_VERSION = (
        "TPSSR_GEOMETRY_STRUCTURE_CONDITIONED_RADIOMETRIC_REFINER"
    )

    def __init__(
        self,
        n_bands,
        spectral_width=SPECTRAL_WIDTH,
        base_width=BASE_WIDTH,
        attention_heads=ATTENTION_HEADS,
    ):
        super().__init__()

        self.n_bands = int(
            n_bands
        )

        self.spectral_compressor = nn.Sequential(
            nn.Conv2d(
                self.n_bands,
                spectral_width,
                1,
                bias=False,
            ),
            nn.GroupNorm(
                choose_groups(
                    spectral_width
                ),
                spectral_width,
            ),
            nn.SiLU(),
            nn.Conv2d(
                spectral_width,
                spectral_width,
                1,
                bias=False,
            ),
            nn.SiLU(),
        )

        # Raw HSI context + observed target + mask geometry + 4-channel
        # structure/contrast prediction.
        input_channels = (
            spectral_width
            + 9
        )

        self.context0 = PartialBlock(
            input_channels,
            base_width,
            stride=1,
        )

        self.context1 = PartialBlock(
            base_width,
            base_width + 32,
            stride=2,
        )

        self.context2 = PartialBlock(
            base_width + 32,
            base_width * 2,
            stride=2,
        )

        # Stage-A learned correspondence feature + explicit structure maps.
        self.structure0 = ConvBlock(
            base_width + 4,
            base_width,
            stride=1,
        )

        self.structure1 = ConvBlock(
            base_width,
            base_width,
            stride=2,
        )

        self.structure2 = ConvBlock(
            base_width,
            base_width,
            stride=2,
        )

        self.fuse0 = ConvBlock(
            base_width
            + base_width,
            base_width,
        )

        self.fuse1 = ConvBlock(
            base_width
            + 32
            + base_width,
            base_width + 32,
        )

        bottleneck_channels = (
            base_width * 2
        )

        self.fuse2 = ConvBlock(
            bottleneck_channels
            + base_width,
            bottleneck_channels,
        )

        self.pre_correspondence = ConvBlock(
            bottleneck_channels,
            bottleneck_channels,
            dilation=2,
        )

        # A second learned retrieval step lets the radiometric branch retrieve
        # intact reflectance patterns after geometry has already been inferred.
        self.correspondence = IntactContextCrossAttention(
            bottleneck_channels,
            attention_heads,
        )

        self.post_correspondence = ConvBlock(
            bottleneck_channels,
            bottleneck_channels,
            dilation=1,
        )

        self.decode1 = ConvBlock(
            bottleneck_channels
            + base_width
            + 32,
            base_width + 32,
        )

        self.decode0 = ConvBlock(
            base_width
            + 32
            + base_width,
            base_width,
        )

        self.coarse_head = nn.Sequential(
            nn.Conv2d(
                base_width,
                base_width,
                3,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv2d(
                base_width,
                1,
                1,
            ),
        )

        # Final detail correction sees both the radiometric decoder and the
        # Stage-A correspondence feature. It is deliberately residual/limited.
        self.detail_head = nn.Sequential(
            ConvBlock(
                base_width
                + base_width
                + 4,
                base_width,
            ),
            nn.Conv2d(
                base_width,
                1,
                1,
            ),
        )

    def forward(
        self,
        corrupted_cube,
        observed_target,
        mask,
        spatial_valid,
        valid_fraction,
        distance,
        band_position,
        structure_edge,
        structure_orientation,
        structure_contrast,
        structure_feature,
    ):
        spectral = self.spectral_compressor(
            corrupted_cube
        )

        n, _c, h, w = spectral.shape

        band_map = band_position.view(
            n,
            1,
            1,
            1,
        ).expand(
            -1,
            -1,
            h,
            w,
        )

        structure_map = torch.cat(
            [
                structure_edge,
                structure_orientation,
                structure_contrast,
            ],
            dim=1,
        )

        context_input = torch.cat(
            [
                spectral,
                observed_target,
                valid_fraction,
                mask,
                distance,
                band_map,
                structure_map,
            ],
            dim=1,
        )

        c0, m0 = self.context0(
            context_input,
            spatial_valid,
        )

        c1, m1 = self.context1(
            c0,
            m0,
        )

        c2, _m2 = self.context2(
            c1,
            m1,
        )

        s0 = self.structure0(
            torch.cat(
                [
                    structure_feature,
                    structure_map,
                ],
                dim=1,
            )
        )

        s1 = self.structure1(
            s0
        )

        s2 = self.structure2(
            s1
        )

        f0 = self.fuse0(
            torch.cat(
                [
                    c0,
                    s0,
                ],
                dim=1,
            )
        )

        f1 = self.fuse1(
            torch.cat(
                [
                    c1,
                    s1,
                ],
                dim=1,
            )
        )

        f2 = self.fuse2(
            torch.cat(
                [
                    c2,
                    s2,
                ],
                dim=1,
            )
        )

        z = self.pre_correspondence(
            f2
        )

        z = self.correspondence(
            z,
            mask=mask,
            spatial_valid=spatial_valid,
        )

        z = self.post_correspondence(
            z
        )

        u1 = F.interpolate(
            z,
            size=f1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        u1 = self.decode1(
            torch.cat(
                [
                    u1,
                    f1,
                ],
                dim=1,
            )
        )

        u0 = F.interpolate(
            u1,
            size=f0.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        u0 = self.decode0(
            torch.cat(
                [
                    u0,
                    f0,
                ],
                dim=1,
            )
        )

        coarse_direct = self.coarse_head(
            u0
        )

        detail_input = torch.cat(
            [
                u0,
                structure_feature,
                structure_map,
            ],
            dim=1,
        )

        detail_residual = (
            DETAIL_RESIDUAL_SCALE
            * torch.tanh(
                self.detail_head(
                    detail_input
                )
            )
        )

        direct_prediction = (
            coarse_direct
            + detail_residual
        )

        coarse_prediction = (
            observed_target
            * (
                1.0
                - mask
            )
            + coarse_direct
            * mask
        )

        prediction = (
            observed_target
            * (
                1.0
                - mask
            )
            + direct_prediction
            * mask
        )

        return {
            "prediction": prediction,
            "direct_prediction": direct_prediction,
            "coarse_prediction": coarse_prediction,
            "detail_residual": detail_residual,
        }


class GeometryGuidanceModel(nn.Module):
    ARCHITECTURE_VERSION = (
        "TPSSR_GEOMETRY_LEARNED_CORRESPONDENCE_STRUCTURE_RADIOMETRIC"
    )

    def __init__(
        self,
        n_bands,
        spectral_width=SPECTRAL_WIDTH,
        base_width=BASE_WIDTH,
        attention_heads=ATTENTION_HEADS,
    ):
        super().__init__()

        self.structure_net = StructureCompletionNet(
            n_bands=n_bands,
            spectral_width=spectral_width,
            base_width=base_width,
            attention_heads=attention_heads,
        )

        self.reconstruction_net = StructureConditionedReconNet(
            n_bands=n_bands,
            spectral_width=spectral_width,
            base_width=base_width,
            attention_heads=attention_heads,
        )

    def forward(
        self,
        corrupted_cube,
        observed_target,
        mask,
        spatial_valid,
        valid_fraction,
        distance,
        band_position,
    ):
        structure = self.structure_net(
            corrupted_cube=corrupted_cube,
            observed_target=observed_target,
            mask=mask,
            spatial_valid=spatial_valid,
            valid_fraction=valid_fraction,
            distance=distance,
            band_position=band_position,
        )

        reconstruction = self.reconstruction_net(
            corrupted_cube=corrupted_cube,
            observed_target=observed_target,
            mask=mask,
            spatial_valid=spatial_valid,
            valid_fraction=valid_fraction,
            distance=distance,
            band_position=band_position,
            structure_edge=structure[
                "edge_prob"
            ],
            structure_orientation=structure[
                "orientation"
            ],
            structure_contrast=structure[
                "contrast_hint"
            ],
            structure_feature=structure[
                "feature"
            ],
        )

        return {
            **structure,
            **reconstruction,
        }


# =============================================================================
# LOSS HELPERS
# =============================================================================

def masked_mean(value, mask, eps=1e-6):
    weight = mask.to(
        dtype=value.dtype
    )

    return (
        (value * weight).sum()
        / weight.sum().clamp_min(eps)
    )


def soft_dice_loss(
    prediction,
    target,
    mask,
    eps=1e-6,
):
    prediction = (
        prediction * mask
    )

    target = (
        target * mask
    )

    intersection = (
        prediction * target
    ).sum(
        dim=(1, 2, 3)
    )

    denominator = (
        prediction.sum(
            dim=(1, 2, 3)
        )
        + target.sum(
            dim=(1, 2, 3)
        )
    )

    dice = (
        2.0 * intersection
        + eps
    ) / (
        denominator
        + eps
    )

    return (
        1.0 - dice
    ).mean()


def structure_loss(
    output,
    batch,
):
    edge_logit = output["edge_logit"]
    edge_prob = output["edge_prob"]
    orientation = output["orientation"]
    contrast_hint = output["contrast_hint"]

    edge_target = batch["edge_target"]
    orientation_target = batch[
        "orientation_target"
    ]

    contrast_target = batch[
        "contrast_target"
    ]

    supervision = batch[
        "structure_supervision"
    ]

    endpoint = batch[
        "endpoint_mask"
    ]

    # Soft edge BCE with extra weight on actual structures.
    bce_map = F.binary_cross_entropy_with_logits(
        edge_logit,
        edge_target,
        reduction="none",
    )

    edge_weight = (
        1.0 + 3.0 * edge_target
    )

    edge_bce = masked_mean(
        bce_map * edge_weight,
        supervision,
    )

    edge_dice = soft_dice_loss(
        edge_prob,
        edge_target,
        supervision,
    )

    # Orientation is meaningful where an edge exists.
    dot = (
        orientation
        * orientation_target
    ).sum(
        dim=1,
        keepdim=True,
    )

    orientation_error = (
        1.0 - dot
    ).clamp(
        0.0,
        2.0,
    )

    orient_weight = (
        edge_target.square()
        * supervision
    )

    orientation_loss = masked_mean(
        orientation_error,
        orient_weight,
    )

    # Give entrance/exit pixels of the missing region additional pressure.
    endpoint_edge = masked_mean(
        torch.abs(
            edge_prob
            - edge_target
        ),
        endpoint,
    )

    endpoint_orient = masked_mean(
        orientation_error,
        endpoint
        * edge_target.clamp_min(0.05),
    )

    endpoint_loss = (
        endpoint_edge
        + endpoint_orient
    )

    contrast_loss = masked_mean(
        torch.abs(
            contrast_hint
            - contrast_target
        ),
        supervision,
    )

    total = (
        W_STRUCT_EDGE_BCE
        * edge_bce
        + W_STRUCT_EDGE_DICE
        * edge_dice
        + W_STRUCT_ORIENTATION
        * orientation_loss
        + W_STRUCT_ENDPOINT
        * endpoint_loss
        + W_STRUCT_CONTRAST
        * contrast_loss
    )

    return total, {
        "edge_bce": edge_bce.detach(),
        "edge_dice": edge_dice.detach(),
        "orientation": orientation_loss.detach(),
        "endpoint": endpoint_loss.detach(),
        "contrast": contrast_loss.detach(),
    }


def sobel_xy(image):
    kx = torch.tensor(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=image.dtype,
        device=image.device,
    ) / 8.0

    ky = kx.t()

    gx = F.conv2d(
        image,
        kx[None, None, ...],
        padding=1,
    )

    gy = F.conv2d(
        image,
        ky[None, None, ...],
        padding=1,
    )

    return gx, gy


def diagonal_responses(image):
    k45 = torch.tensor(
        [
            [-1.0, -1.0, 0.0],
            [-1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=image.dtype,
        device=image.device,
    ) / 4.0

    k135 = torch.tensor(
        [
            [0.0, 1.0, 1.0],
            [-1.0, 0.0, 1.0],
            [-1.0, -1.0, 0.0],
        ],
        dtype=image.dtype,
        device=image.device,
    ) / 4.0

    r45 = F.conv2d(
        image,
        k45[None, None, ...],
        padding=1,
    )

    r135 = F.conv2d(
        image,
        k135[None, None, ...],
        padding=1,
    )

    return r45, r135


def laplacian(image):
    kernel = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=image.dtype,
        device=image.device,
    )

    return F.conv2d(
        image,
        kernel[None, None, ...],
        padding=1,
    )


def inner_boundary(mask):
    eroded = (
        1.0
        - F.max_pool2d(
            1.0 - mask,
            kernel_size=3,
            stride=1,
            padding=1,
        )
    )

    return (
        mask - eroded
    ).clamp(
        0.0,
        1.0,
    )


def masked_ssim_loss(
    prediction,
    target,
    mask,
    window=7,
):
    pad = (
        window // 2
    )

    mu_x = F.avg_pool2d(
        prediction,
        window,
        stride=1,
        padding=pad,
    )

    mu_y = F.avg_pool2d(
        target,
        window,
        stride=1,
        padding=pad,
    )

    var_x = (
        F.avg_pool2d(
            prediction.square(),
            window,
            stride=1,
            padding=pad,
        )
        - mu_x.square()
    ).clamp_min(0.0)

    var_y = (
        F.avg_pool2d(
            target.square(),
            window,
            stride=1,
            padding=pad,
        )
        - mu_y.square()
    ).clamp_min(0.0)

    cov = (
        F.avg_pool2d(
            prediction * target,
            window,
            stride=1,
            padding=pad,
        )
        - mu_x * mu_y
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim = (
        (
            2.0 * mu_x * mu_y
            + c1
        )
        * (
            2.0 * cov
            + c2
        )
    ) / (
        (
            mu_x.square()
            + mu_y.square()
            + c1
        )
        * (
            var_x
            + var_y
            + c2
        )
    ).clamp_min(
        1e-6
    )

    return masked_mean(
        1.0 - ssim,
        mask,
    )


def undirected_gradient_orientation_loss(
    pred_gx,
    pred_gy,
    targ_gx,
    targ_gy,
    mask,
):
    pred_mag = torch.sqrt(
        pred_gx.square()
        + pred_gy.square()
        + 1e-8
    )

    targ_mag = torch.sqrt(
        targ_gx.square()
        + targ_gy.square()
        + 1e-8
    )

    cosine = (
        pred_gx * targ_gx
        + pred_gy * targ_gy
    ) / (
        pred_mag * targ_mag
        + 1e-6
    )

    # Undirected line orientation: +gradient and -gradient are equivalent.
    error = (
        1.0 - cosine.abs()
    ).clamp(
        0.0,
        1.0,
    )

    scale = (
        targ_mag.flatten(1)
        .mean(
            dim=1,
            keepdim=True,
        )
        .view(
            -1,
            1,
            1,
            1,
        )
        .clamp_min(
            1e-4
        )
    )

    structure_weight = torch.clamp(
        targ_mag
        / (
            1.5 * scale
        ),
        0.0,
        2.0,
    )

    return masked_mean(
        error
        * structure_weight,
        mask,
    )



def local_residual(
    image,
    window=11,
):
    local_mean = F.avg_pool2d(
        image,
        kernel_size=window,
        stride=1,
        padding=window // 2,
    )

    return (
        image
        - local_mean
    )


def local_contrast_polarity_loss(
    prediction,
    target,
    mask,
):
    pred_res = local_residual(
        prediction,
        window=11,
    )

    targ_res = local_residual(
        target,
        window=11,
    )

    target_strength = torch.abs(
        targ_res
    )

    per_sample_scale = (
        target_strength.flatten(
            1
        )
        .mean(
            dim=1,
            keepdim=True,
        )
        .view(
            -1,
            1,
            1,
            1,
        )
        .clamp_min(
            1e-4
        )
    )

    strength_weight = (
        1.0
        + torch.clamp(
            target_strength
            / (
                1.5
                * per_sample_scale
            ),
            0.0,
            2.5,
        )
    )

    contrast = masked_mean(
        torch.abs(
            pred_res
            - targ_res
        )
        * strength_weight,
        mask,
    )

    # Differentiable polarity margin:
    # if GT is locally dark, prediction must remain sufficiently negative;
    # if GT is locally bright, prediction must remain sufficiently positive.
    target_sign = torch.sign(
        targ_res
    ).detach()

    strong = (
        target_strength
        > (
            0.45
            * per_sample_scale
        )
    ).to(
        dtype=prediction.dtype
    )

    polarity_mask = (
        mask
        * strong
    )

    margin = (
        0.40
        * target_strength.detach()
    )

    signed_prediction = (
        target_sign
        * pred_res
    )

    polarity = masked_mean(
        F.relu(
            margin
            - signed_prediction
        ),
        polarity_mask,
    )

    return (
        contrast,
        polarity,
    )


def shift_replicate(
    image,
    dy,
    dx,
):
    """
    Integer spatial shift with replicate padding; no wrap-around.
    """
    dy = int(
        dy
    )

    dx = int(
        dx
    )

    pad_y = abs(
        dy
    )

    pad_x = abs(
        dx
    )

    padded = F.pad(
        image,
        (
            pad_x,
            pad_x,
            pad_y,
            pad_y,
        ),
        mode="replicate",
    )

    h = image.shape[
        -2
    ]

    w = image.shape[
        -1
    ]

    y0 = (
        pad_y
        + dy
    )

    x0 = (
        pad_x
        + dx
    )

    return padded[
        :,
        :,
        y0:y0 + h,
        x0:x0 + w,
    ]


def cross_section_profile_loss(
    prediction,
    target,
    mask,
):
    """
    Compare signed local-contrast profiles crossing every masked pixel in four
    directions. This explicitly supervises dark/bright transitions and feature
    thickness, rather than only aggregate edge energy.
    """
    pred_res = local_residual(
        prediction,
        window=9,
    )

    targ_res = local_residual(
        target,
        window=9,
    )

    directions = (
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
    )

    offsets = (
        -4,
        -2,
        0,
        2,
        4,
    )

    total_map = torch.zeros_like(
        prediction
    )

    count = 0

    for dy, dx in directions:
        for offset in offsets:
            pred_shift = shift_replicate(
                pred_res,
                offset * dy,
                offset * dx,
            )

            targ_shift = shift_replicate(
                targ_res,
                offset * dy,
                offset * dx,
            )

            total_map = (
                total_map
                + torch.abs(
                    pred_shift
                    - targ_shift
                )
            )

            count += 1

    total_map = (
        total_map
        / float(
            max(
                count,
                1,
            )
        )
    )

    target_strength = torch.abs(
        targ_res
    )

    scale = (
        target_strength.flatten(
            1
        )
        .mean(
            dim=1,
            keepdim=True,
        )
        .view(
            -1,
            1,
            1,
            1,
        )
        .clamp_min(
            1e-4
        )
    )

    weight = (
        1.0
        + torch.clamp(
            target_strength
            / (
                1.5
                * scale
            ),
            0.0,
            2.0,
        )
    )

    return masked_mean(
        total_map
        * weight,
        mask,
    )

def reconstruction_loss(
    output,
    batch,
):
    prediction = output[
        "prediction"
    ]

    target = batch[
        "target"
    ]

    mask = batch[
        "mask"
    ]

    error_map = torch.sqrt(
        (
            prediction - target
        ).square()
        + 1e-6
    )

    pixel = masked_mean(
        error_map,
        mask,
    )

    pred_gx, pred_gy = sobel_xy(
        prediction
    )

    targ_gx, targ_gy = sobel_xy(
        target
    )

    targ_mag = torch.sqrt(
        targ_gx.square()
        + targ_gy.square()
        + 1e-8
    )

    scale = (
        targ_mag.flatten(1)
        .mean(
            dim=1,
            keepdim=True,
        )
        .view(
            -1,
            1,
            1,
            1,
        )
        .clamp_min(
            1e-4
        )
    )

    edge_weight = (
        1.0
        + 2.0
        * torch.clamp(
            targ_mag
            / (
                2.0
                * scale
            ),
            0.0,
            2.0,
        )
    )

    edge_pixel = masked_mean(
        error_map
        * edge_weight,
        mask,
    )

    grad_mask = F.max_pool2d(
        mask,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    gradient = masked_mean(
        torch.abs(
            pred_gx
            - targ_gx
        )
        + torch.abs(
            pred_gy
            - targ_gy
        ),
        grad_mask,
    )

    orient = (
        undirected_gradient_orientation_loss(
            pred_gx,
            pred_gy,
            targ_gx,
            targ_gy,
            grad_mask,
        )
    )

    pred45, pred135 = diagonal_responses(
        prediction
    )

    targ45, targ135 = diagonal_responses(
        target
    )

    diagonal = masked_mean(
        torch.abs(
            pred45 - targ45
        )
        + torch.abs(
            pred135 - targ135
        ),
        grad_mask,
    )

    ssim = masked_ssim_loss(
        prediction,
        target,
        mask,
    )

    boundary_mask = inner_boundary(
        mask
    )

    boundary = masked_mean(
        torch.abs(
            pred_gx
            - targ_gx
        )
        + torch.abs(
            pred_gy
            - targ_gy
        ),
        boundary_mask,
    )

    lap = masked_mean(
        torch.abs(
            laplacian(
                prediction
            )
            - laplacian(
                target
            )
        ),
        grad_mask,
    )

    local_contrast, polarity = local_contrast_polarity_loss(
        prediction,
        target,
        mask,
    )

    profile = cross_section_profile_loss(
        prediction,
        target,
        mask,
    )

    coarse_prediction = output[
        "coarse_prediction"
    ]

    coarse = masked_mean(
        torch.sqrt(
            (
                coarse_prediction
                - target
            ).square()
            + 1e-6
        ),
        mask,
    )

    total = (
        W_PIXEL
        * pixel
        + W_EDGE_PIXEL
        * edge_pixel
        + W_GRADIENT
        * gradient
        + W_RECON_ORIENTATION
        * orient
        + W_DIAGONAL
        * diagonal
        + W_SSIM
        * ssim
        + W_BOUNDARY
        * boundary
        + W_LAPLACIAN
        * lap
        + W_LOCAL_CONTRAST
        * local_contrast
        + W_POLARITY
        * polarity
        + W_PROFILE
        * profile
        + W_COARSE
        * coarse
    )

    return total, {
        "pixel": pixel.detach(),
        "edge_pixel": edge_pixel.detach(),
        "gradient": gradient.detach(),
        "orientation": orient.detach(),
        "diagonal": diagonal.detach(),
        "ssim": ssim.detach(),
        "boundary": boundary.detach(),
        "laplacian": lap.detach(),
        "local_contrast": local_contrast.detach(),
        "polarity": polarity.detach(),
        "profile": profile.detach(),
        "coarse": coarse.detach(),
    }


# =============================================================================
# TRAIN / VALIDATION UTILITIES
# =============================================================================

def move_batch(batch, device):
    output = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(
                device,
                non_blocking=True,
            )
        else:
            output[key] = value

    return output


def structure_forward(model, batch):
    return model(
        corrupted_cube=batch["cube"],
        observed_target=batch["observed_target"],
        mask=batch["mask"],
        spatial_valid=batch["spatial_valid"],
        valid_fraction=batch["valid_fraction"],
        distance=batch["distance"],
        band_position=batch["band_position"],
    )


def full_forward(model, batch):
    return model(
        corrupted_cube=batch["cube"],
        observed_target=batch["observed_target"],
        mask=batch["mask"],
        spatial_valid=batch["spatial_valid"],
        valid_fraction=batch["valid_fraction"],
        distance=batch["distance"],
        band_position=batch["band_position"],
    )


@torch.no_grad()
def evaluate_structure(
    model,
    loader,
    device,
):
    model.eval()

    sums = {
        "loss": 0.0,
        "edge_bce": 0.0,
        "edge_dice": 0.0,
        "orientation": 0.0,
        "endpoint": 0.0,
        "steps": 0.0,
    }

    for batch in loader:
        batch = move_batch(
            batch,
            device,
        )

        with torch.amp.autocast(
            "cuda",
            enabled=True,
        ):
            output = structure_forward(
                model,
                batch,
            )

            loss, parts = structure_loss(
                output,
                batch,
            )

        sums["loss"] += float(
            loss.item()
        )

        for key in (
            "edge_bce",
            "edge_dice",
            "orientation",
            "endpoint",
        ):
            sums[key] += float(
                parts[key].item()
            )

        sums["steps"] += 1.0

    global_sums = {
        key: all_reduce_sum(
            value,
            device,
        )
        for key, value in sums.items()
    }

    steps = max(
        global_sums["steps"],
        1.0,
    )

    return {
        key: (
            global_sums[key] / steps
        )
        for key in (
            "loss",
            "edge_bce",
            "edge_dice",
            "orientation",
            "endpoint",
        )
    }


@torch.no_grad()
def evaluate_joint(
    model,
    loader,
    device,
):
    model.eval()

    sums = {
        "total_loss": 0.0,
        "recon_loss": 0.0,
        "structure_loss": 0.0,
        "prediction_sq": 0.0,
        "n": 0.0,
        "recon_orientation": 0.0,
        "diagonal": 0.0,
        "ssim": 0.0,
        "boundary": 0.0,
        "local_contrast": 0.0,
        "polarity": 0.0,
        "profile": 0.0,
        "coarse": 0.0,
        "struct_edge_bce": 0.0,
        "struct_edge_dice": 0.0,
        "struct_orientation": 0.0,
        "steps": 0.0,
    }

    for batch in loader:
        batch = move_batch(
            batch,
            device,
        )

        with torch.amp.autocast(
            "cuda",
            enabled=True,
        ):
            output = full_forward(
                model,
                batch,
            )

            recon_loss_value, recon_parts = (
                reconstruction_loss(
                    output,
                    batch,
                )
            )

            structure_loss_value, structure_parts = (
                structure_loss(
                    output,
                    batch,
                )
            )

            total_loss = (
                recon_loss_value
                + W_STRUCTURE_DURING_JOINT
                * structure_loss_value
            )

        mask_bool = (
            batch["mask"] > 0.5
        )

        prediction = output[
            "prediction"
        ].float()

        target = batch[
            "target"
        ].float()

        sums["prediction_sq"] += float(
            (
                (
                    prediction
                    - target
                ).square()
            )[
                mask_bool
            ].sum().item()
        )

        sums["n"] += float(
            mask_bool.sum().item()
        )

        sums["total_loss"] += float(
            total_loss.item()
        )

        sums["recon_loss"] += float(
            recon_loss_value.item()
        )

        sums["structure_loss"] += float(
            structure_loss_value.item()
        )

        sums["recon_orientation"] += float(
            recon_parts[
                "orientation"
            ].item()
        )

        sums["diagonal"] += float(
            recon_parts[
                "diagonal"
            ].item()
        )

        sums["ssim"] += float(
            recon_parts[
                "ssim"
            ].item()
        )

        sums["boundary"] += float(
            recon_parts[
                "boundary"
            ].item()
        )

        sums["local_contrast"] += float(
            recon_parts[
                "local_contrast"
            ].item()
        )

        sums["polarity"] += float(
            recon_parts[
                "polarity"
            ].item()
        )

        sums["profile"] += float(
            recon_parts[
                "profile"
            ].item()
        )

        sums["coarse"] += float(
            recon_parts[
                "coarse"
            ].item()
        )

        sums["struct_edge_bce"] += float(
            structure_parts[
                "edge_bce"
            ].item()
        )

        sums["struct_edge_dice"] += float(
            structure_parts[
                "edge_dice"
            ].item()
        )

        sums["struct_orientation"] += float(
            structure_parts[
                "orientation"
            ].item()
        )

        sums["steps"] += 1.0

    global_sums = {
        key: all_reduce_sum(
            value,
            device,
        )
        for key, value in sums.items()
    }

    steps = max(
        global_sums["steps"],
        1.0,
    )

    n = max(
        global_sums["n"],
        1.0,
    )

    return {
        "total_loss": (
            global_sums[
                "total_loss"
            ] / steps
        ),
        "recon_loss": (
            global_sums[
                "recon_loss"
            ] / steps
        ),
        "structure_loss": (
            global_sums[
                "structure_loss"
            ] / steps
        ),
        "prediction_rmse": math.sqrt(
            global_sums[
                "prediction_sq"
            ] / n
        ),
        "recon_orientation": (
            global_sums[
                "recon_orientation"
            ] / steps
        ),
        "diagonal": (
            global_sums[
                "diagonal"
            ] / steps
        ),
        "ssim": (
            global_sums[
                "ssim"
            ] / steps
        ),
        "boundary": (
            global_sums[
                "boundary"
            ] / steps
        ),
        "local_contrast": (
            global_sums[
                "local_contrast"
            ] / steps
        ),
        "polarity": (
            global_sums[
                "polarity"
            ] / steps
        ),
        "profile": (
            global_sums[
                "profile"
            ] / steps
        ),
        "coarse": (
            global_sums[
                "coarse"
            ] / steps
        ),
        "struct_edge_bce": (
            global_sums[
                "struct_edge_bce"
            ] / steps
        ),
        "struct_edge_dice": (
            global_sums[
                "struct_edge_dice"
            ] / steps
        ),
        "struct_orientation": (
            global_sums[
                "struct_orientation"
            ] / steps
        ),
    }


# =============================================================================
# CHECKPOINTS
# =============================================================================

def unwrap(model):
    return (
        model.module
        if isinstance(model, DDP)
        else model
    )


def save_structure_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_val,
    metadata_checkpoint,
    metadata_checkpoint_path,
):
    raw_model = unwrap(
        model
    )

    checkpoint = {
        "architecture": (
            StructureCompletionNet.ARCHITECTURE_VERSION
        ),
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val": float(best_val),
        "patch_size": int(PATCH_SIZE),
        "n_bands": int(
            metadata_checkpoint[
                "n_bands"
            ]
        ),
        "band_mean": np.asarray(
            metadata_checkpoint[
                "band_mean"
            ],
            dtype=np.float32,
        ),
        "band_std": np.asarray(
            metadata_checkpoint[
                "band_std"
            ],
            dtype=np.float32,
        ),
        "wavelengths_nm": np.asarray(
            metadata_checkpoint[
                "wavelengths_nm"
            ],
            dtype=np.float32,
        ),
        "normalization_metadata_source": str(
            metadata_checkpoint_path
        ),
        "settings": {
            "spectral_width": int(
                SPECTRAL_WIDTH
            ),
            "base_width": int(
                BASE_WIDTH
            ),
            "attention_heads": int(
                ATTENTION_HEADS
            ),
        },
    }

    torch.save(
        checkpoint,
        path,
    )


def save_joint_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_val,
    metadata_checkpoint,
    metadata_checkpoint_path,
    world_size,
):
    raw_model = unwrap(
        model
    )

    checkpoint = {
        "architecture": (
            GeometryGuidanceModel.ARCHITECTURE_VERSION
        ),
        "structure_architecture": (
            StructureCompletionNet.ARCHITECTURE_VERSION
        ),
        "reconstruction_architecture": (
            StructureConditionedReconNet.ARCHITECTURE_VERSION
        ),
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val": float(best_val),
        "patch_size": int(PATCH_SIZE),
        "n_bands": int(
            metadata_checkpoint[
                "n_bands"
            ]
        ),
        "band_mean": np.asarray(
            metadata_checkpoint[
                "band_mean"
            ],
            dtype=np.float32,
        ),
        "band_std": np.asarray(
            metadata_checkpoint[
                "band_std"
            ],
            dtype=np.float32,
        ),
        "wavelengths_nm": np.asarray(
            metadata_checkpoint[
                "wavelengths_nm"
            ],
            dtype=np.float32,
        ),
        "normalization_metadata_source": str(
            metadata_checkpoint_path
        ),
        "uses_unused_expert_forward_during_training": False,
        "uses_spatial_mean_prior": False,
        "uses_topology_prior": False,
        "settings": {
            "spectral_width": int(
                SPECTRAL_WIDTH
            ),
            "base_width": int(
                BASE_WIDTH
            ),
            "attention_heads": int(
                ATTENTION_HEADS
            ),
            "learned_intact_context_cross_attention": True,
            "stage_a_signed_contrast": True,
            "uses_explicit_path_pairing": False,
            "uses_exemplar_donor_input": False,
            "uses_unused_expert_reconstruction_prior": False,
            "world_size": int(
                world_size
            ),
            "mask_shapes": [
                [
                    int(a),
                    int(b),
                ]
                for a, b in MASK_SHAPES
            ],
            "structure_epochs": int(
                STRUCTURE_EPOCHS
            ),
            "joint_epochs": int(
                JOINT_EPOCHS
            ),
        },
    }

    torch.save(
        checkpoint,
        path,
    )


# =============================================================================
# DATA LOADERS
# =============================================================================

def make_loaders(
    train_ds,
    val_ds,
    rank,
    world_size,
):
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

    # persistent_workers=False is deliberate:
    # train_ds.set_epoch() must reach the worker copies every epoch so that
    # synthetic masks do not repeat identically.
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_PER_GPU,
        sampler=train_sampler,
        shuffle=False,
        num_workers=NUM_WORKERS_PER_GPU,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE_PER_GPU,
        sampler=val_sampler,
        shuffle=False,
        num_workers=NUM_WORKERS_PER_GPU,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
    )

    return (
        train_loader,
        val_loader,
        train_sampler,
        val_sampler,
    )


# =============================================================================
# PHASE A
# =============================================================================

def train_structure_phase(
    rank,
    local_rank,
    world_size,
    device,
    train_ds,
    val_ds,
    metadata_checkpoint,
    metadata_checkpoint_path,
    model_dir,
):
    main_process = (
        rank == 0
    )

    (
        train_loader,
        val_loader,
        train_sampler,
        _val_sampler,
    ) = make_loaders(
        train_ds,
        val_ds,
        rank,
        world_size,
    )

    model = StructureCompletionNet(
        n_bands=int(
            metadata_checkpoint[
                "n_bands"
            ]
        ),
        spectral_width=SPECTRAL_WIDTH,
        base_width=BASE_WIDTH,
        attention_heads=ATTENTION_HEADS,
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
        lr=STRUCTURE_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(
            1,
            STRUCTURE_EPOCHS,
        ),
        eta_min=STRUCTURE_LR * 0.12,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )

    history = []
    best_val = float("inf")

    if main_process:
        parameter_count = sum(
            parameter.numel()
            for parameter in unwrap(
                model
            ).parameters()
        )

        print("")
        print("-" * 108)
        print("PHASE A — LEARNED CORRESPONDENCE + STRUCTURE COMPLETION")
        print("-" * 108)
        print(
            "Structure parameters:",
            f"{parameter_count / 1e6:.3f} M",
        )
        print(
            "Goal: predict edge probability + cos(2θ)/sin(2θ) inside the hole."
        )
        print("")

    for epoch in range(
        1,
        STRUCTURE_EPOCHS + 1,
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
            "edge_bce": 0.0,
            "edge_dice": 0.0,
            "orientation": 0.0,
            "endpoint": 0.0,
            "steps": 0.0,
        }

        progress = tqdm(
            train_loader,
            desc=(
                f"Structure {epoch:02d}/"
                f"{STRUCTURE_EPOCHS}"
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
                output = structure_forward(
                    model,
                    batch,
                )

                loss, parts = structure_loss(
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

            for key in (
                "edge_bce",
                "edge_dice",
                "orientation",
                "endpoint",
            ):
                local[key] += float(
                    parts[key].item()
                )

            local["steps"] += 1.0

            if main_process:
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    dice=f"{parts['edge_dice'].item():.3f}",
                    ori=f"{parts['orientation'].item():.3f}",
                    end=f"{parts['endpoint'].item():.3f}",
                )

        global_local = {
            key: all_reduce_sum(
                value,
                device,
            )
            for key, value in local.items()
        }

        train_steps = max(
            global_local["steps"],
            1.0,
        )

        val = evaluate_structure(
            model,
            val_loader,
            device,
        )

        scheduler.step()

        if main_process:
            row = {
                "epoch": int(
                    epoch
                ),
                "learning_rate": float(
                    optimizer.param_groups[
                        0
                    ]["lr"]
                ),
                "train_loss": (
                    global_local[
                        "loss"
                    ] / train_steps
                ),
                "train_edge_bce": (
                    global_local[
                        "edge_bce"
                    ] / train_steps
                ),
                "train_edge_dice": (
                    global_local[
                        "edge_dice"
                    ] / train_steps
                ),
                "train_orientation": (
                    global_local[
                        "orientation"
                    ] / train_steps
                ),
                "train_endpoint": (
                    global_local[
                        "endpoint"
                    ] / train_steps
                ),
                "val_loss": float(
                    val[
                        "loss"
                    ]
                ),
                "val_edge_bce": float(
                    val[
                        "edge_bce"
                    ]
                ),
                "val_edge_dice": float(
                    val[
                        "edge_dice"
                    ]
                ),
                "val_orientation": float(
                    val[
                        "orientation"
                    ]
                ),
                "val_endpoint": float(
                    val[
                        "endpoint"
                    ]
                ),
            }

            history.append(
                row
            )

            is_best = (
                row[
                    "val_loss"
                ] < best_val
            )

            if is_best:
                best_val = float(
                    row[
                        "val_loss"
                    ]
                )

                save_structure_checkpoint(
                    model_dir
                    / BEST_STRUCTURE_MODEL,
                    model,
                    optimizer,
                    epoch,
                    best_val,
                    metadata_checkpoint,
                    metadata_checkpoint_path,
                )

            with (
                model_dir
                / STRUCTURE_HISTORY_CSV
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

            print(
                f"Structure epoch {epoch:02d} | "
                f"train={row['train_loss']:.5f} | "
                f"val={row['val_loss']:.5f} | "
                f"diceLoss={row['val_edge_dice']:.4f} | "
                f"orientation={row['val_orientation']:.4f} | "
                f"endpoint={row['val_endpoint']:.4f}"
            )

            if is_best:
                print(
                    "  -> NEW BEST STRUCTURE MODEL"
                )

            print("")

        ddp_barrier(
            local_rank
        )

    ddp_barrier(
        local_rank
    )

    return (
        model_dir
        / BEST_STRUCTURE_MODEL
    )


# =============================================================================
# PHASE B
# =============================================================================

def train_joint_phase(
    rank,
    local_rank,
    world_size,
    device,
    train_ds,
    val_ds,
    metadata_checkpoint,
    metadata_checkpoint_path,
    model_dir,
    structure_checkpoint_path,
):
    main_process = (
        rank == 0
    )

    (
        train_loader,
        val_loader,
        train_sampler,
        _val_sampler,
    ) = make_loaders(
        train_ds,
        val_ds,
        rank,
        world_size,
    )

    try:
        structure_checkpoint = torch.load(
            structure_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        structure_checkpoint = torch.load(
            structure_checkpoint_path,
            map_location="cpu",
        )

    model = GeometryGuidanceModel(
        n_bands=int(
            metadata_checkpoint[
                "n_bands"
            ]
        ),
        spectral_width=SPECTRAL_WIDTH,
        base_width=BASE_WIDTH,
        attention_heads=ATTENTION_HEADS,
    ).to(device)

    model.structure_net.load_state_dict(
        structure_checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

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

    raw_model = unwrap(
        model
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": raw_model.structure_net.parameters(),
                "lr": JOINT_STRUCTURE_LR,
            },
            {
                "params": raw_model.reconstruction_net.parameters(),
                "lr": JOINT_RECON_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(
            1,
            JOINT_EPOCHS,
        ),
        eta_min=JOINT_STRUCTURE_LR * 0.10,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )

    history = []
    best_val = float("inf")

    if main_process:
        parameter_count = sum(
            parameter.numel()
            for parameter in raw_model.parameters()
        )

        structure_count = sum(
            parameter.numel()
            for parameter in raw_model.structure_net.parameters()
        )

        recon_count = sum(
            parameter.numel()
            for parameter in raw_model.reconstruction_net.parameters()
        )

        print("")
        print("-" * 108)
        print("PHASE B — STRUCTURE-CONDITIONED RADIOMETRIC RECONSTRUCTION")
        print("-" * 108)
        print(
            "Total parameters:",
            f"{parameter_count / 1e6:.3f} M",
        )
        print(
            "Structure branch:",
            f"{structure_count / 1e6:.3f} M",
        )
        print(
            "Reconstruction branch:",
            f"{recon_count / 1e6:.3f} M",
        )
        print(
            "The reconstruction branch receives the PREDICTED structure map, "
            "not ground-truth structure."
        )
        print("")

    for epoch in range(
        1,
        JOINT_EPOCHS + 1,
    ):
        train_ds.set_epoch(
            STRUCTURE_EPOCHS
            + epoch
        )

        train_sampler.set_epoch(
            STRUCTURE_EPOCHS
            + epoch
        )

        model.train()

        local = {
            "total_loss": 0.0,
            "recon_loss": 0.0,
            "structure_loss": 0.0,
            "recon_orientation": 0.0,
            "diagonal": 0.0,
            "ssim": 0.0,
            "local_contrast": 0.0,
            "polarity": 0.0,
            "profile": 0.0,
            "struct_orientation": 0.0,
            "steps": 0.0,
        }

        progress = tqdm(
            train_loader,
            desc=(
                f"Joint {epoch:02d}/"
                f"{JOINT_EPOCHS}"
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
                output = full_forward(
                    model,
                    batch,
                )

                recon_loss_value, recon_parts = (
                    reconstruction_loss(
                        output,
                        batch,
                    )
                )

                structure_loss_value, structure_parts = (
                    structure_loss(
                        output,
                        batch,
                    )
                )

                total_loss = (
                    recon_loss_value
                    + W_STRUCTURE_DURING_JOINT
                    * structure_loss_value
                )

            scaler.scale(
                total_loss
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

            local["total_loss"] += float(
                total_loss.item()
            )

            local["recon_loss"] += float(
                recon_loss_value.item()
            )

            local["structure_loss"] += float(
                structure_loss_value.item()
            )

            local["recon_orientation"] += float(
                recon_parts[
                    "orientation"
                ].item()
            )

            local["diagonal"] += float(
                recon_parts[
                    "diagonal"
                ].item()
            )

            local["ssim"] += float(
                recon_parts[
                    "ssim"
                ].item()
            )

            local["local_contrast"] += float(
                recon_parts[
                    "local_contrast"
                ].item()
            )

            local["polarity"] += float(
                recon_parts[
                    "polarity"
                ].item()
            )

            local["profile"] += float(
                recon_parts[
                    "profile"
                ].item()
            )

            local["struct_orientation"] += float(
                structure_parts[
                    "orientation"
                ].item()
            )

            local["steps"] += 1.0

            if main_process:
                progress.set_postfix(
                    total=f"{total_loss.item():.4f}",
                    recon=f"{recon_loss_value.item():.4f}",
                    sOri=f"{structure_parts['orientation'].item():.3f}",
                    rOri=f"{recon_parts['orientation'].item():.3f}",
                    ctr=f"{recon_parts['local_contrast'].item():.3f}",
                    pol=f"{recon_parts['polarity'].item():.3f}",
                )

        global_local = {
            key: all_reduce_sum(
                value,
                device,
            )
            for key, value in local.items()
        }

        train_steps = max(
            global_local[
                "steps"
            ],
            1.0,
        )

        val = evaluate_joint(
            model,
            val_loader,
            device,
        )

        scheduler.step()

        if main_process:
            row = {
                "epoch": int(
                    epoch
                ),
                "structure_lr": float(
                    optimizer.param_groups[
                        0
                    ]["lr"]
                ),
                "reconstruction_lr": float(
                    optimizer.param_groups[
                        1
                    ]["lr"]
                ),
                "train_total_loss": (
                    global_local[
                        "total_loss"
                    ] / train_steps
                ),
                "train_recon_loss": (
                    global_local[
                        "recon_loss"
                    ] / train_steps
                ),
                "train_structure_loss": (
                    global_local[
                        "structure_loss"
                    ] / train_steps
                ),
                "train_recon_orientation": (
                    global_local[
                        "recon_orientation"
                    ] / train_steps
                ),
                "train_diagonal": (
                    global_local[
                        "diagonal"
                    ] / train_steps
                ),
                "train_ssim_loss": (
                    global_local[
                        "ssim"
                    ] / train_steps
                ),
                "train_local_contrast": (
                    global_local[
                        "local_contrast"
                    ] / train_steps
                ),
                "train_polarity": (
                    global_local[
                        "polarity"
                    ] / train_steps
                ),
                "train_profile": (
                    global_local[
                        "profile"
                    ] / train_steps
                ),
                "train_structure_orientation": (
                    global_local[
                        "struct_orientation"
                    ] / train_steps
                ),
                "val_total_loss": float(
                    val[
                        "total_loss"
                    ]
                ),
                "val_recon_loss": float(
                    val[
                        "recon_loss"
                    ]
                ),
                "val_structure_loss": float(
                    val[
                        "structure_loss"
                    ]
                ),
                "val_geometry_rmse_norm": float(
                    val[
                        "prediction_rmse"
                    ]
                ),
                "val_recon_orientation": float(
                    val[
                        "recon_orientation"
                    ]
                ),
                "val_diagonal": float(
                    val[
                        "diagonal"
                    ]
                ),
                "val_ssim_loss": float(
                    val[
                        "ssim"
                    ]
                ),
                "val_boundary": float(
                    val[
                        "boundary"
                    ]
                ),
                "val_local_contrast": float(
                    val[
                        "local_contrast"
                    ]
                ),
                "val_polarity": float(
                    val[
                        "polarity"
                    ]
                ),
                "val_profile": float(
                    val[
                        "profile"
                    ]
                ),
                "val_coarse": float(
                    val[
                        "coarse"
                    ]
                ),
                "val_structure_edge_bce": float(
                    val[
                        "struct_edge_bce"
                    ]
                ),
                "val_structure_edge_dice": float(
                    val[
                        "struct_edge_dice"
                    ]
                ),
                "val_structure_orientation": float(
                    val[
                        "struct_orientation"
                    ]
                ),
            }

            history.append(
                row
            )

            is_best = (
                row[
                    "val_total_loss"
                ] < best_val
            )

            if is_best:
                best_val = float(
                    row[
                        "val_total_loss"
                    ]
                )

            save_joint_checkpoint(
                model_dir
                / LAST_MODEL,
                model,
                optimizer,
                epoch,
                best_val,
                metadata_checkpoint,
                metadata_checkpoint_path,
                world_size,
            )

            if is_best:
                save_joint_checkpoint(
                    model_dir
                    / BEST_MODEL,
                    model,
                    optimizer,
                    epoch,
                    best_val,
                    metadata_checkpoint,
                    metadata_checkpoint_path,
                    world_size,
                )

            with (
                model_dir
                / JOINT_HISTORY_CSV
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

            print(
                f"Joint epoch {epoch:02d} | "
                f"train={row['train_total_loss']:.5f} | "
                f"val={row['val_total_loss']:.5f} | "
                f"RMSE={row['val_geometry_rmse_norm']:.5f} | "
                f"reconOri={row['val_recon_orientation']:.4f} | "
                f"diag={row['val_diagonal']:.4f} | "
                f"structOri={row['val_structure_orientation']:.4f}"
            )

            if is_best:
                print(
                    "  -> NEW BEST Geometry MODEL"
                )

            print("")

        ddp_barrier(
            local_rank
        )

    return history


# =============================================================================
# WORKER
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

        set_seed(
            SEED + rank
        )

        expected_train = {
            str(x).upper()
            for x in getattr(config, "ALL_SENSOR_DATASETS", [])
            if str(x).upper() != str(config.TEST_DATASET_FOLDER).upper()
        }
        current_train = {str(item).upper() for item in config.TRAIN_DATASET_FOLDERS}
        if not expected_train or current_train != expected_train:
            raise RuntimeError(
                f"Invalid configured train/test split for Geometry: held-out={config.TEST_DATASET_FOLDER}, "
                f"train={config.TRAIN_DATASET_FOLDERS}"
            )

        (
            metadata_checkpoint,
            metadata_checkpoint_path,
        ) = load_normalization_metadata()

        band_mean = np.asarray(
            metadata_checkpoint[
                "band_mean"
            ],
            dtype=np.float32,
        )

        band_std = np.maximum(
            np.asarray(
                metadata_checkpoint[
                    "band_std"
                ],
                dtype=np.float32,
            ),
            1e-6,
        )

        n_bands = int(
            metadata_checkpoint[
                "n_bands"
            ]
        )

        (
            train_cubes,
            val_cubes,
            wavelengths,
        ) = load_global_training_data()

        wavelengths = np.asarray(
            wavelengths,
            dtype=np.float32,
        )

        if int(
            np.asarray(
                train_cubes[0]
            ).shape[-1]
        ) != n_bands:
            raise RuntimeError(
                "Normalization metadata band count does not match SWIR cubes."
            )

        if not np.allclose(
            wavelengths,
            np.asarray(
                metadata_checkpoint[
                    "wavelengths_nm"
                ],
                dtype=np.float32,
            ),
            rtol=0.0,
            atol=1e-3,
        ):
            raise RuntimeError(
                "Normalization wavelength grid does not match the current SWIR fold."
            )

        model_dir = (
            Path(config.MODELS_DIR)
            / "tpssr"
            / MODEL_SUBDIR
        )

        if main_process:
            model_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            print("")
            print("=" * 108)
            print(
                "TPSSR — Geometry LEARNED-CORRESPONDENCE WHOLE-SPECTRUM RECONSTRUCTOR"
            )
            print("=" * 108)
            print(
                "Held-out:",
                config.TEST_DATASET_FOLDER,
            )
            print(
                "Train:",
                ", ".join(
                    config.TRAIN_DATASET_FOLDERS
                ),
            )
            print(
                "Normalization metadata:",
                metadata_checkpoint_path,
            )
            print(
                "Patch:",
                f"{PATCH_SIZE}x{PATCH_SIZE}",
            )
            print(
                "Mask shapes:",
                MASK_SHAPES,
            )
            print(
                "Structure epochs:",
                STRUCTURE_EPOCHS,
            )
            print(
                "Joint epochs:",
                JOINT_EPOCHS,
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
            print(
                "Geometry fundamental change:"
            )
            print(
                "  raw context -> LEARNED INTACT-CONTEXT CORRESPONDENCE"
                " -> structure/contrast completion -> radiometric refinement"
            )
            print(
                "  Explicit path / track / donor correspondence: NO"
            )
            print(
                "  the previous experimental expert forward during training: 0"
            )
            print(
                "  Spatial-mean prior: NO"
            )
            print(
                "  Topology/Guard reconstruction prior: NO"
            )
            print("")

        train_ds = GeometryLearnedCorrespondenceDataset(
            cubes=train_cubes,
            band_mean=band_mean,
            band_std=band_std,
            samples=TRAIN_SAMPLES_PER_EPOCH,
            seed=SEED + 101,
            training=True,
        )

        val_ds = GeometryLearnedCorrespondenceDataset(
            cubes=val_cubes,
            band_mean=band_mean,
            band_std=band_std,
            samples=VAL_SAMPLES,
            seed=SEED + 202,
            training=False,
        )

        structure_checkpoint_path = (
            train_structure_phase(
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                device=device,
                train_ds=train_ds,
                val_ds=val_ds,
                metadata_checkpoint=metadata_checkpoint,
                metadata_checkpoint_path=metadata_checkpoint_path,
                model_dir=model_dir,
            )
        )

        ddp_barrier(
            local_rank
        )

        joint_history = train_joint_phase(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
            train_ds=train_ds,
            val_ds=val_ds,
            metadata_checkpoint=metadata_checkpoint,
            metadata_checkpoint_path=metadata_checkpoint_path,
            model_dir=model_dir,
            structure_checkpoint_path=structure_checkpoint_path,
        )

        ddp_barrier(
            local_rank
        )

        if main_process:
            best_row = min(
                joint_history,
                key=lambda item: item[
                    "val_total_loss"
                ],
            )

            summary = {
                "architecture": (
                    GeometryGuidanceModel.ARCHITECTURE_VERSION
                ),
                "held_out_test": str(config.TEST_DATASET_FOLDER),
                "training_datasets": list(
                    config.TRAIN_DATASET_FOLDERS
                ),
                "structure_epochs": int(
                    STRUCTURE_EPOCHS
                ),
                "joint_epochs": int(
                    JOINT_EPOCHS
                ),
                "best_joint_epoch": int(
                    best_row[
                        "epoch"
                    ]
                ),
                "best_val_total_loss": float(
                    best_row[
                        "val_total_loss"
                    ]
                ),
                "val_geometry_rmse_norm": float(
                    best_row[
                        "val_geometry_rmse_norm"
                    ]
                ),
                "val_recon_orientation": float(
                    best_row[
                        "val_recon_orientation"
                    ]
                ),
                "val_diagonal": float(
                    best_row[
                        "val_diagonal"
                    ]
                ),
                "val_ssim_loss": float(
                    best_row[
                        "val_ssim_loss"
                    ]
                ),
                "val_boundary": float(
                    best_row[
                        "val_boundary"
                    ]
                ),
                "val_local_contrast": float(
                    best_row[
                        "val_local_contrast"
                    ]
                ),
                "val_polarity": float(
                    best_row[
                        "val_polarity"
                    ]
                ),
                "val_profile": float(
                    best_row[
                        "val_profile"
                    ]
                ),
                "val_structure_edge_dice": float(
                    best_row[
                        "val_structure_edge_dice"
                    ]
                ),
                "val_structure_orientation": float(
                    best_row[
                        "val_structure_orientation"
                    ]
                ),
            }

            lines = [
                "TPSSR — Geometry LEARNED-CORRESPONDENCE WHOLE-SPECTRUM RECONSTRUCTOR",
                "=" * 100,
                "",
                "WHY Geometry",
                (
                    "  V25/V26 diagnostics showed that explicit paths and fixed "
                    "exemplar selection are too brittle for unique interrupted "
                    "sedimentary structures."
                ),
                (
                    "  Geometry therefore learns correspondence from synthetic "
                    "whole-spectrum holes instead of receiving a hand-engineered "
                    "path, pair, or donor."
                ),
                "",
                "STAGE A — LEARNED CORRESPONDENCE + STRUCTURE",
                (
                    "  Missing-region queries cross-attend only to intact context "
                    "tokens in the raw normalized HSI patch."
                ),
                "  Outputs: edge probability",
                "           cos(2theta), sin(2theta) orientation field",
                "           signed local-contrast hint",
                "",
                "STAGE B — RADIOMETRIC RECONSTRUCTION",
                (
                    "  Uses raw corrupted HSI + learned Stage-A correspondence "
                    "features + predicted structure/contrast."
                ),
                (
                    "  A coarse reconstruction is followed by a limited residual "
                    "refiner so detail correction cannot freely overwrite geometry."
                ),
                "",
                "ANTI-BLUR SUPERVISION",
                "  local signed-contrast loss",
                "  explicit dark/bright polarity margin loss",
                "  multi-direction cross-sectional profile loss",
                "  gradient/orientation/diagonal/boundary/SSIM losses",
                "",
                "IMPORTANT NEGATIVES",
                "  the previous experimental expert reconstruction as input: NO",
                "  Topology/Guard/Policy/V24/V25/V26 reconstruction as input: NO",
                "  explicit path/track matching: NO",
                "  explicit exemplar donor input: NO",
                "  spatial-mean reconstruction prior: NO",
                "",
                "BEST JOINT VALIDATION",
                (
                    f"  Best joint epoch: {summary['best_joint_epoch']}"
                ),
                (
                    f"  Val total loss: {summary['best_val_total_loss']:.8f}"
                ),
                (
                    f"  Normalized RMSE: {summary['val_geometry_rmse_norm']:.6f}"
                ),
                (
                    "  Reconstruction orientation loss: "
                    f"{summary['val_recon_orientation']:.6f}"
                ),
                (
                    f"  Diagonal loss: {summary['val_diagonal']:.6f}"
                ),
                (
                    f"  SSIM loss: {summary['val_ssim_loss']:.6f}"
                ),
                (
                    f"  Boundary loss: {summary['val_boundary']:.6f}"
                ),
                (
                    "  Structure edge Dice loss: "
                    f"{summary['val_structure_edge_dice']:.6f}"
                ),
                (
                    "  Structure orientation loss: "
                    f"{summary['val_structure_orientation']:.6f}"
                ),
                "",
                "NEXT HELD-OUT TEST",
                "  whole_spectrum_pixels S5 R1 and R4 on SWIR1",
                "  Visualize Stage-A predicted edge/orientation inside every component.",
                "  Compare the previous experimental expert / Topology / Current HF / Guard.1 / Geometry.",
                (
                    "  The key success criterion is real continuation of interrupted "
                    "diagonal/thin structures, not only lower pixel RMSE."
                ),
            ]

            (
                model_dir
                / SUMMARY_TXT
            ).write_text(
                "\n".join(
                    lines
                )
                + "\n",
                encoding="utf-8",
            )

            (
                model_dir
                / "training_summary.json"
            ).write_text(
                json.dumps(
                    summary,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            print("")
            print("=" * 108)
            print("Geometry TRAINING COMPLETE")
            print("=" * 108)
            print(
                "Best joint epoch:",
                summary[
                    "best_joint_epoch"
                ],
            )
            print(
                "Normalized RMSE:",
                f"{summary['val_geometry_rmse_norm']:.6f}",
            )
            print(
                "Best checkpoint:",
                model_dir
                / BEST_MODEL,
            )
            print(
                "Summary:",
                model_dir
                / SUMMARY_TXT,
            )

    finally:
        cleanup_ddp()


# =============================================================================
# LAUNCHER
# =============================================================================

def launcher_main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpus",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    if args.worker:
        worker_main()
        return

    if int(args.gpus) != 2:
        raise RuntimeError(
            "Geometry is configured for exactly --gpus 2."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"--gpus 2 requested but only "
            f"{torch.cuda.device_count()} GPUs are visible."
        )

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(HERE),
        "--gpus",
        "2",
        "--worker",
    ]

    print("")
    print(
        "Launching TPSSR Geometry on two GPUs:"
    )
    print(
        " ".join(
            command
        )
    )
    print("")

    env = os.environ.copy()
    env.setdefault(
        "OMP_NUM_THREADS",
        "1",
    )
    env.setdefault(
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "1",
    )

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        check=False,
    )

    if completed.returncode != 0:
        raise SystemExit(
            completed.returncode
        )


if __name__ == "__main__":
    launcher_main()
