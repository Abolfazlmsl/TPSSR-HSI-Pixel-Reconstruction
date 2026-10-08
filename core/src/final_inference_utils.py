"""Inference helpers for the final TPSSR architecture.

This file contains only the preview/context utilities needed by the final
hierarchical router.  It replaces the old experimental test.py dependency.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve()
CORE = next(p for p in HERE.parents if (p / "config.py").exists())
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
METHOD_DIR = CORE / "reconstruction_methods" / "texture_restoration_router_rf_v2"
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

import config
import train_swir1_deep_v31 as D31
from src.sparse_router_experts import build_sparse_router_features
from src.component_routing_utils import (
    component_features,
    component_band_stats,
    representative_pixels,
)


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_patch(array, y, x, size, pad_value=None):
    """Extract HxWxC patch without padding the entire HSI cube in memory."""
    radius = int(size) // 2
    h, w = array.shape[:2]
    y0 = max(0, int(y) - radius)
    y1 = min(h, int(y) + radius + 1)
    x0 = max(0, int(x) - radius)
    x1 = min(w, int(x) + radius + 1)
    patch = np.asarray(array[y0:y1, x0:x1, :]).copy()

    top = max(0, radius - int(y))
    bottom = max(0, int(y) + radius + 1 - h)
    left = max(0, radius - int(x))
    right = max(0, int(x) + radius + 1 - w)
    if top or bottom or left or right:
        if pad_value is None:
            patch = np.pad(
                patch,
                ((top, bottom), (left, right), (0, 0)),
                mode="edge",
            )
        else:
            patch = np.pad(
                patch,
                ((top, bottom), (left, right), (0, 0)),
                mode="constant",
                constant_values=pad_value,
            )
    return patch


def load_deep_v31(device):
    model_dir = Path(config.MODELS_DIR) / "texture_restoration_router_rf_v2"
    checkpoint_path = (
        model_dir
        / f"deep_v31_{str(config.TEST_DATASET_FOLDER).lower()}_dev_2gpu"
        / "best_model.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Deep-v3.1 checkpoint not found. Run train.py first:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch_load(checkpoint_path, "cpu")
    settings = checkpoint.get("settings", {})
    model = D31.DeepExpertV31(
        n_bands=int(checkpoint["n_bands"]),
        context_size=int(checkpoint["context_size"]),
        output_tile=int(checkpoint["output_tile"]),
        width=int(settings.get("width", 80)),
        anchor_width=int(settings.get("anchor_width", 32)),
        dilations=tuple(
            int(x)
            for x in settings.get("dilations", (1, 2, 4, 8, 4, 2, 1))
        ),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint, checkpoint_path


class RouterPixelDataset(Dataset):
    """One item per spatial pixel used to preview Spectral and Deep-v3.1."""

    def __init__(self, test_cube, mask, pixels, deep_checkpoint, router_checkpoint):
        self.test_cube = test_cube
        self.mask = mask
        self.pixels = np.asarray(pixels, dtype=np.int64)
        self.context_size = int(deep_checkpoint["context_size"])
        self.output_tile = int(deep_checkpoint["output_tile"])
        self.router_patch_size = int(
            router_checkpoint.get("settings", {}).get(
                "router_patch_size", config.SPARSE_ROUTER_PATCH_SIZE
            )
        )
        self.band_mean = np.asarray(deep_checkpoint["band_mean"], dtype=np.float32)
        self.band_std = np.maximum(
            np.asarray(deep_checkpoint["band_std"], dtype=np.float32), 1e-6
        )
        self.wavelengths_nm = np.asarray(
            deep_checkpoint["wavelengths_nm"], dtype=np.float32
        )

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        y, x = map(int, self.pixels[index])

        patch = extract_patch(
            self.test_cube, y, x, self.context_size, pad_value=None
        ).astype(np.float32)
        patch_mask = extract_patch(
            self.mask, y, x, self.context_size, pad_value=True
        ).astype(bool)
        finite = np.isfinite(patch)
        safe = np.where(
            finite, patch, self.band_mean[None, None, :]
        ).astype(np.float32)
        clean_norm = (
            safe - self.band_mean[None, None, :]
        ) / self.band_std[None, None, :]
        corrupted = clean_norm.copy()
        corrupted[patch_mask] = 0.0

        c = self.context_size // 2
        half = self.output_tile // 2
        y0, y1 = c - half, c + half + 1
        x0, x1 = c - half, c + half + 1
        clean_tile = clean_norm[y0:y1, x0:x1, :].astype(np.float32)
        mask_tile = patch_mask[y0:y1, x0:x1, :].astype(bool)
        anchor, anchor_available = D31.anchor_map_for_tile(
            clean_tile, mask_tile, self.wavelengths_nm
        )

        router_patch = extract_patch(
            self.test_cube, y, x, self.router_patch_size, pad_value=None
        ).astype(np.float32)
        router_mask = extract_patch(
            self.mask, y, x, self.router_patch_size, pad_value=True
        ).astype(bool)
        features = build_sparse_router_features(
            router_patch,
            router_mask,
            self.band_mean,
            self.band_std,
            self.wavelengths_nm,
        )

        return {
            "deep_cube": torch.from_numpy(
                np.moveaxis(corrupted, -1, 0).astype(np.float32)
            ),
            "deep_mask": torch.from_numpy(
                np.moveaxis(patch_mask.astype(np.float32), -1, 0)
            ),
            "anchor": torch.from_numpy(
                np.moveaxis(anchor, -1, 0).astype(np.float32)
            ),
            "anchor_available": torch.from_numpy(
                np.moveaxis(anchor_available, -1, 0).astype(np.float32)
            ),
            "router_input": torch.from_numpy(features["input"].copy()),
            "spectral_candidate": torch.from_numpy(features["experts"][:, 0].copy()),
            "spectral_available": torch.from_numpy(
                features["availability"][:, 0].copy()
            ),
            "spectral_confidence": torch.from_numpy(
                features["confidence"][:, 0].copy()
            ),
            "geometry": torch.from_numpy(features["geometry"].copy()),
            "pixel_index": torch.tensor(int(index), dtype=torch.long),
        }


@torch.no_grad()
def predict_pixel_previews(
    deep_model,
    deep_checkpoint,
    router_checkpoint,
    test_cube,
    mask,
    pixels,
    device,
    batch_size,
    progress_desc=None,
):
    """Return normalized Spectral/Deep previews for requested spatial pixels."""
    pixels = np.asarray(pixels, dtype=np.int64)
    if len(pixels) == 0:
        b = int(deep_checkpoint["n_bands"])
        return {
            "pixels": pixels,
            "spectral_norm": np.empty((0, b), np.float32),
            "deep_norm": np.empty((0, b), np.float32),
            "spectral_available": np.empty((0, b), bool),
            "spectral_confidence": np.empty((0, b), np.float32),
        }

    dataset = RouterPixelDataset(
        test_cube, mask, pixels, deep_checkpoint, router_checkpoint
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    n_bands = int(deep_checkpoint["n_bands"])
    spectral = np.full((len(pixels), n_bands), np.nan, np.float32)
    deep = np.full((len(pixels), n_bands), np.nan, np.float32)
    available = np.zeros((len(pixels), n_bands), bool)
    confidence = np.zeros((len(pixels), n_bands), np.float32)
    center = int(deep_checkpoint["output_tile"]) // 2
    rank = int(os.environ.get("RANK", "0"))

    desc = progress_desc or f"GPU {rank} Spectral/Deep previews"
    for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        ids = batch["pixel_index"].numpy().astype(np.int64)
        deep_cube = batch["deep_cube"].to(device, non_blocking=True)
        deep_mask = batch["deep_mask"].to(device, non_blocking=True)
        anchor = batch["anchor"].to(device, non_blocking=True)
        anchor_av = batch["anchor_available"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            deep_result = deep_model(deep_cube, deep_mask, anchor, anchor_av)
            deep_norm = deep_result["prediction"][:, :, center, center]
        spectral[ids] = batch["spectral_candidate"].numpy().astype(np.float32)
        deep[ids] = deep_norm.float().cpu().numpy().astype(np.float32)
        available[ids] = batch["spectral_available"].numpy() > 0.5
        confidence[ids] = batch["spectral_confidence"].numpy().astype(np.float32)

    return {
        "pixels": pixels,
        "spectral_norm": spectral,
        "deep_norm": deep,
        "spectral_available": available,
        "spectral_confidence": confidence,
    }


def component_router_context(
    test_cube,
    mask,
    component,
    preview,
    pixel_lookup,
    deep_checkpoint,
    router_checkpoint,
):
    coords = np.argwhere(component)
    cy = int(round(float(coords[:, 0].mean())))
    cx = int(round(float(coords[:, 1].mean())))
    patch_size = int(
        router_checkpoint.get("settings", {}).get(
            "router_patch_size", config.SPARSE_ROUTER_PATCH_SIZE
        )
    )
    band_mean = np.asarray(deep_checkpoint["band_mean"], dtype=np.float32)
    band_std = np.maximum(
        np.asarray(deep_checkpoint["band_std"], dtype=np.float32), 1e-6
    )
    wavelengths = np.asarray(deep_checkpoint["wavelengths_nm"], dtype=np.float32)
    rp = extract_patch(test_cube, cy, cx, patch_size, pad_value=None).astype(np.float32)
    rm = extract_patch(mask, cy, cx, patch_size, pad_value=True).astype(bool)
    features = build_sparse_router_features(rp, rm, band_mean, band_std, wavelengths)

    reps = representative_pixels(
        component,
        int(getattr(config, "V31_COMPONENT_ROUTER_PREVIEW_PIXELS", 32)),
    )
    rows = [pixel_lookup[(int(y), int(x))] for y, x in reps]
    n_bands = len(band_mean)
    spec = np.zeros(n_bands, np.float32)
    deep = np.zeros(n_bands, np.float32)
    sav = np.zeros(n_bands, np.float32)
    sconf = np.zeros(n_bands, np.float32)
    for b in range(n_bands):
        use = [
            r
            for r, (y, x) in zip(rows, reps)
            if mask[int(y), int(x), b]
        ]
        if not use:
            use = rows
        if use:
            av = preview["spectral_available"][use, b]
            sav[b] = float(np.mean(av))
            sconf[b] = float(np.nanmean(preview["spectral_confidence"][use, b]))
            deep[b] = float(np.nanmean(preview["deep_norm"][use, b]))
            if np.any(av):
                spec[b] = float(
                    np.nanmean(
                        preview["spectral_norm"][np.asarray(use)[av], b]
                    )
                )
    bfrac, bcount, counts = component_band_stats(mask, component)
    cfeat = component_features(mask, component)
    return features, spec, sav, sconf, deep, bfrac, bcount, counts, cfeat
