"""Minimal training helpers for the final hierarchical router.

This module contains only the Spectral/Deep preview and synthetic component
sampling code required by train_swir1_hierarchical_router.py.  It intentionally
excludes the abandoned flat three-expert Region pipeline.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve()
CORE = next(p for p in HERE.parents if (p / "config.py").exists())
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import config
from src.sparse_router_experts import build_sparse_router_features
from src.texture_v2_corruptions import SCENARIOS, generate_texture_v2_training_mask
import train_swir1_deep_v31 as D31

SCENARIO_TO_ID = {name: i for i, name in enumerate(SCENARIOS)}
DEEP_BATCH = int(getattr(config, "V31_COMPONENT_ROUTER_DEEP_BATCH", 8))


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_deep(device):
    path = (
        Path(config.MODELS_DIR)
        / "texture_restoration_router_rf_v2"
        / f"deep_v31_{str(config.TEST_DATASET_FOLDER).lower()}_dev_2gpu"
        / "best_model.pt"
    )
    if not path.exists():
        raise FileNotFoundError(f"Deep-v3.1 checkpoint missing:\n{path}")
    ck = torch_load(path, "cpu")
    s = ck["settings"]
    model = D31.DeepExpertV31(
        n_bands=int(ck["n_bands"]),
        context_size=int(ck["context_size"]),
        output_tile=int(ck["output_tile"]),
        width=int(s["width"]),
        anchor_width=int(s["anchor_width"]),
        dilations=tuple(int(x) for x in s["dilations"]),
    ).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    return model, ck, path


def extract_patch(array, y, x, size, pad_value=None):
    r = int(size) // 2
    h, w = array.shape[:2]
    y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
    x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
    p = np.asarray(array[y0:y1, x0:x1, ...]).copy()
    top = max(0, r - int(y))
    bottom = max(0, int(y) + r + 1 - h)
    left = max(0, r - int(x))
    right = max(0, int(x) + r + 1 - w)
    if top or bottom or left or right:
        pads = ((top, bottom), (left, right)) + ((0, 0),) * (p.ndim - 2)
        if pad_value is None:
            p = np.pad(p, pads, mode="edge")
        else:
            p = np.pad(p, pads, mode="constant", constant_values=pad_value)
    return p


def connected_components_4(binary):
    m = np.asarray(binary, dtype=bool)
    h, w = m.shape
    seen = np.zeros_like(m)
    out = []
    for sy, sx in np.argwhere(m):
        sy, sx = int(sy), int(sx)
        if seen[sy, sx]:
            continue
        stack = [(sy, sx)]
        seen[sy, sx] = True
        comp = []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and m[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        out.append(np.asarray(comp, dtype=np.int64))
    return out


def component_features(mask, component):
    comp = np.asarray(component, dtype=bool)
    coords = np.argwhere(comp)
    area = max(len(coords), 1)
    h, w, _ = mask.shape
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    bh, bw = int(y1 - y0 + 1), int(x1 - x0 + 1)
    fill = area / max(bh * bw, 1)
    boundary = 0
    for y, x in coords:
        y, x = int(y), int(x)
        if any(
            not (0 <= yy < h and 0 <= xx < w and comp[yy, xx])
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
        ):
            boundary += 1
    missfrac = mask[comp, :].mean(axis=1) if area else np.array([0.0])
    whole = float(np.mean(missfrac >= 0.999))
    vox = float(mask[comp, :].mean()) if area else 0.0
    aspect = max(bh, bw) / max(min(bh, bw), 1)
    return np.asarray(
        [
            math.log1p(area) / math.log1p(max(h * w, 2)),
            bh / max(h, 1),
            bw / max(w, 1),
            fill,
            math.log1p(aspect) / math.log1p(max(h, w, 2)),
            boundary / max(area, 1),
            vox,
            float(np.mean(missfrac)),
            float(np.std(missfrac)),
            float(np.max(missfrac)),
            whole,
            min(1.0, area / 225.0),
        ],
        dtype=np.float32,
    )


def component_band_stats(mask, component):
    comp = np.asarray(component, dtype=bool)
    area = max(int(comp.sum()), 1)
    counts = mask[comp, :].sum(axis=0).astype(np.float32)
    frac = counts / float(area)
    count_norm = np.log1p(counts) / math.log1p(float(area))
    return frac.astype(np.float32), count_norm.astype(np.float32)


class ComponentRouterSamples(Dataset):
    def __init__(
        self,
        cubes,
        wavelengths,
        band_mean,
        band_std,
        samples,
        seed,
        deep_ck,
        region_patch,
    ):
        self.cubes = [np.asarray(c) for c in cubes]
        self.wavelengths = np.asarray(wavelengths, dtype=np.float32)
        self.band_mean = np.asarray(band_mean, dtype=np.float32)
        self.band_std = np.maximum(np.asarray(band_std, dtype=np.float32), 1e-6)
        self.samples = int(samples)
        self.seed = int(seed)
        self.deep_context = int(deep_ck["context_size"])
        self.router_patch = int(config.SPARSE_ROUTER_PATCH_SIZE)
        self.master = max(int(region_patch), self.deep_context, self.router_patch)
        if self.master % 2 != 1:
            raise ValueError("Router master patch must be odd.")

    def __len__(self):
        return self.samples

    def rng(self, index):
        return np.random.default_rng(self.seed + int(index) * 104729)

    def _clean(self, rng):
        r = self.master // 2
        for _ in range(120):
            cube = self.cubes[int(rng.integers(0, len(self.cubes)))]
            h, w, _ = cube.shape
            if h <= 2 * r + 2 or w <= 2 * r + 2:
                continue
            y = int(rng.integers(r, h - r))
            x = int(rng.integers(r, w - r))
            p = np.asarray(cube[y - r : y + r + 1, x - r : x + r + 1, :], dtype=np.float32).copy()
            if np.all(np.isfinite(p[r, r, :])) and float(np.mean(np.isfinite(p))) >= 0.985:
                return p
        raise RuntimeError("Could not sample component-router patch.")

    def build(self, index, include_expert_inputs=False):
        rng = self.rng(index)
        c = self.master // 2
        for _ in range(180):
            clean = self._clean(rng)
            mask, scenario, severity = generate_texture_v2_training_mask(
                clean.shape, rng, severity=int(rng.integers(1, 6))
            )
            mask &= np.isfinite(clean)
            if not np.any(mask[c, c, :]):
                continue
            spatial = np.any(mask, axis=2)
            comps = connected_components_4(spatial)
            component = None
            for comp in comps:
                if np.any((comp[:, 0] == c) & (comp[:, 1] == c)):
                    component = np.zeros(spatial.shape, dtype=bool)
                    component[comp[:, 0], comp[:, 1]] = True
                    break
            if component is None:
                continue
            missing_bands = np.flatnonzero(np.any(mask[component, :], axis=0))
            if len(missing_bands) == 0:
                continue
            center_missing = np.flatnonzero(mask[c, c, :])
            target_band = (
                int(center_missing[int(rng.integers(0, len(center_missing)))])
                if len(center_missing)
                else int(missing_bands[int(rng.integers(0, len(missing_bands)))])
            )
            coords = np.argwhere(component)
            cy = int(round(coords[:, 0].mean()))
            cx = int(round(coords[:, 1].mean()))
            rp = extract_patch(clean, cy, cx, self.router_patch, None)
            rm = extract_patch(mask, cy, cx, self.router_patch, True).astype(bool)
            feat = build_sparse_router_features(
                rp, rm, self.band_mean, self.band_std, self.wavelengths
            )
            bfrac, bcount = component_band_stats(mask, component)
            cfeat = component_features(mask, component)
            out = {
                "sample_index": torch.tensor(int(index), dtype=torch.long),
                "router_input": torch.from_numpy(feat["input"].copy()),
                "geometry": torch.from_numpy(feat["geometry"].copy()),
                "component_band_fraction": torch.from_numpy(bfrac),
                "component_band_count": torch.from_numpy(bcount),
                "component_features": torch.from_numpy(cfeat),
                "target_band": torch.tensor(target_band, dtype=torch.long),
                "scenario_id": torch.tensor(SCENARIO_TO_ID[scenario], dtype=torch.long),
                "severity": torch.tensor(int(severity), dtype=torch.long),
            }
            if include_expert_inputs:
                out.update(
                    {
                        "clean": clean,
                        "mask": mask,
                        "component": component,
                        "centroid": (cy, cx),
                    }
                )
            return out
        raise RuntimeError("Could not construct component-router sample.")

    def __getitem__(self, index):
        return self.build(index, False)


def _pixel_previews(
    deep_model,
    deep_ck,
    clean,
    mask,
    pixels,
    band_mean,
    band_std,
    wavelengths,
    device,
):
    pixels = np.asarray(pixels, dtype=np.int64)
    n, b = len(pixels), len(band_mean)
    spec = np.full((n, b), np.nan, np.float32)
    sav = np.zeros((n, b), np.float32)
    sconf = np.zeros((n, b), np.float32)
    deep = np.full((n, b), np.nan, np.float32)

    for i, (y, x) in enumerate(pixels):
        rp = extract_patch(clean, int(y), int(x), int(config.SPARSE_ROUTER_PATCH_SIZE), None)
        rm = extract_patch(mask, int(y), int(x), int(config.SPARSE_ROUTER_PATCH_SIZE), True).astype(bool)
        f = build_sparse_router_features(rp, rm, band_mean, band_std, wavelengths)
        spec[i] = f["experts"][:, 0]
        sav[i] = f["availability"][:, 0]
        sconf[i] = f["confidence"][:, 0]

    cs = int(deep_ck["context_size"])
    ot = int(deep_ck["output_tile"])
    center = ot // 2
    for start in range(0, n, DEEP_BATCH):
        cubes, masks, anchors, anchor_av = [], [], [], []
        for y, x in pixels[start : start + DEEP_BATCH]:
            p = extract_patch(clean, int(y), int(x), cs, None).astype(np.float32)
            pm = extract_patch(mask, int(y), int(x), cs, True).astype(bool)
            finite = np.isfinite(p)
            safe = np.where(finite, p, band_mean[None, None, :]).astype(np.float32)
            norm = (safe - band_mean[None, None, :]) / band_std[None, None, :]
            corr = norm.copy()
            corr[pm] = 0.0
            cc, hh = cs // 2, ot // 2
            tile = norm[cc - hh : cc + hh + 1, cc - hh : cc + hh + 1, :]
            mt = pm[cc - hh : cc + hh + 1, cc - hh : cc + hh + 1, :]
            a, av = D31.anchor_map_for_tile(tile, mt, wavelengths)
            cubes.append(np.moveaxis(corr, -1, 0))
            masks.append(np.moveaxis(pm.astype(np.float32), -1, 0))
            anchors.append(np.moveaxis(a, -1, 0))
            anchor_av.append(np.moveaxis(av, -1, 0))
        with torch.no_grad():
            out = deep_model(
                torch.from_numpy(np.asarray(cubes, np.float32)).to(device),
                torch.from_numpy(np.asarray(masks, np.float32)).to(device),
                torch.from_numpy(np.asarray(anchors, np.float32)).to(device),
                torch.from_numpy(np.asarray(anchor_av, np.float32)).to(device),
            )
            deep[start : start + len(cubes)] = (
                out["prediction"][:, :, center, center].float().cpu().numpy()
            )
    return spec, sav, sconf, deep


def _choose_pixels(coords, max_n):
    coords = np.asarray(coords, dtype=np.int64)
    if len(coords) <= max_n:
        return coords
    ids = np.linspace(0, len(coords) - 1, max_n).round().astype(np.int64)
    return coords[ids]
