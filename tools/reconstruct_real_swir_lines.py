#!/usr/bin/env python3
"""Reconstruct genuine SWIR detector-line defects using the final TPSSR.

Runs one full-line TPSSR inference per GPU (plus a Router-only global-count
prepass) without redundant 128-row outer tiling. Emits full-resolution 2D
before/after images using the dataset-specific HSI viewer contrast settings.
No missing-value ground truth, synthetic masks or checkpoint changes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CORE = ROOT / "core"
ROI_SIZE = 128


@dataclass
class LineCandidate:
    band: int
    x: int
    roi_y: int
    roi_x: int
    score: float
    missing_fraction: float
    flank_missing_fraction: float
    width_estimate: int
    wavelength_nm: float


def parser():
    p = argparse.ArgumentParser(description="Real SWIR vertical-line TPSSR reconstruction for publication")
    p.add_argument("--datasets", nargs="+", choices=("SWIR5", "SWIR10"), default=["SWIR5", "SWIR10"])
    p.add_argument("--preset", default="swir_8x2", choices=["swir_8x2"])
    p.add_argument("--image-scope", choices=["full", "test-block"], default="full",
                   help="Full acquisition (default) or official held-out last 20 percent")
    p.add_argument("--gpus", type=int, choices=[1, 2], default=2)
    p.add_argument("--roi-only", action="store_true",
                   help="Use the original 128x128 ROI figures instead of full-acquisition band images")
    p.add_argument("--outdir", type=Path, default=ROOT / "figures" / "real_vertical_lines_fast")
    p.add_argument("--band-swir5", type=int, default=None, help="Manual zero-based SWIR5 band")
    p.add_argument("--line-x-swir5", type=int, default=None, help="Manual SWIR5 full-image column")
    p.add_argument("--roi-y-swir5", type=int, default=None, help="Manual SWIR5 full-image ROI top row")
    p.add_argument("--band-swir10", type=int, default=None)
    p.add_argument("--line-x-swir10", type=int, default=None)
    p.add_argument("--roi-y-swir10", type=int, default=None)
    p.add_argument("--line-width", type=int, default=0,
                   help="0: detect narrow invalid-column group; otherwise center a forced width")
    p.add_argument("--min-line-fraction", type=float, default=0.75,
                   help="Minimum actual invalid-pixel fraction along the 128-row target line")
    p.add_argument("--list-only", action="store_true",
                   help="Only identify promising real vertical-line defects; do not load TPSSR")
    p.add_argument("--max-candidates", type=int, default=30)
    p.add_argument("--region-batch", type=int, default=16)
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--full-line-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--workdir", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker-dataset", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--front-source-swir5", choices=("processed", "raw-calibrated"), default="processed",
                   help="2D display source for SWIR5; model input stays processed")
    p.add_argument("--front-source-swir10", choices=("processed", "raw-calibrated"), default="raw-calibrated",
                   help="2D display source for SWIR10; model input stays processed")
    p.add_argument("--front-gamma-swir5", type=float, default=0.72)
    p.add_argument("--front-gamma-swir10", type=float, default=0.25)
    p.add_argument("--front-brightness-swir5", type=float, default=1.08)
    p.add_argument("--front-brightness-swir10", type=float, default=1.08)
    p.add_argument("--front-low-pct", type=float, default=1.0)
    p.add_argument("--front-high-pct", type=float, default=99.0)
    p.add_argument("--raw-dir", type=Path, default=None,
                   help="Parent input_raw/ directory containing SWIR5 and SWIR10")
    return p


def experiment_environment(active_dataset):
    from experiment_config import resolve_experiment
    from _launcher_common import env_for
    from types import SimpleNamespace
    spec = resolve_experiment(SimpleNamespace(preset="swir_8x2", dataset=None,
                                               active_test=None, train_datasets=None,
                                               test_datasets=None, experiment_name=None))
    if active_dataset not in spec.tests:
        raise ValueError(f"{active_dataset} must be one of held-out datasets {spec.tests}")
    env = env_for(spec, spec.primary_test)
    # The same observed-donor policy as the final TPSSR ROI visualizer.
    env["TPSSR_FUSION_DONOR_MATCH"] = "robust"
    env["PYTHONPATH"] = os.pathsep.join([str(CORE), str(ROOT), env.get("PYTHONPATH", "")])
    return env, spec


def load_source(dataset, image_scope):
    from src.evaluation_utils import load_dataset, split_blocks
    cube, invalid, split, wavelengths = load_dataset(dataset)
    offset = 0
    if image_scope == "test-block":
        _, _, cube = split_blocks(cube, split)
        _, _, invalid = split_blocks(invalid, split)
        offset = int(split["val_end"])
    if cube.shape[:2] != invalid.shape[:2] or cube.shape[2] != len(wavelengths):
        raise RuntimeError("Invalid cube/mask/wavelength shapes")
    return cube, invalid, np.asarray(wavelengths), offset


def detect_lines(invalid, waves, offset, max_candidates=30):
    """Scan 128-row segments: a narrow missing column with intact flanks.

    Reads the *real* processed invalid_mask.npy, never invents a missing line.
    Overlapping 128-row windows avoid overlooking defects confined to part
    of the acquisition. A candidate is verified on the full-resolution crop.
    """
    h, w, b = invalid.shape
    if h < ROI_SIZE or w < ROI_SIZE:
        raise ValueError(f"Image {h}x{w} is smaller than the {ROI_SIZE}x{ROI_SIZE} ROI")
    row_step = max(1, math.ceil(h / 850))
    reduced = np.asarray(invalid[::row_step], dtype=np.bool_)
    sampled_rows = reduced.shape[0]
    nwin = max(8, math.ceil(ROI_SIZE / row_step))
    stride = max(4, nwin // 2)
    # Evaluate a manageable number of windows. Full-resolution recheck follows.
    starts = list(range(0, max(1, sampled_rows - nwin + 1), stride))
    last_start = max(0, sampled_rows - nwin)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    best = {}
    for start in starts:
        block = reduced[start:start + nwin]
        rate = block.mean(axis=0)  # column x band
        # The adjacent flanks must be mostly observed; skip wide missing blocks.
        shoulder = np.ones_like(rate)
        for x in range(9, w - 9):
            shoulder[x] = 0.5 * (rate[x-8:x-3].mean(axis=0) + rate[x+4:x+9].mean(axis=0))
        score = rate - shoulder
        eligible = (rate >= 0.78) & (shoulder <= 0.33) & (score >= 0.55)
        eligible[:9, :] = False
        eligible[-9:, :] = False
        choices = np.argwhere(eligible)
        if not choices.size:
            continue
        iy = min(max(round((start + nwin / 2) * row_step - ROI_SIZE / 2), 0), h - ROI_SIZE)
        for x, band in choices:
            x, band = int(x), int(band)
            ix = min(max(x - ROI_SIZE // 2, 0), w - ROI_SIZE)
            observed = np.asarray(invalid[iy:iy + ROI_SIZE, x, band], bool)
            fraction = float(observed.mean())
            if fraction < 0.70:
                continue
            # Check that the line isn't a globally dead spatial column in all bands.
            every_band_rate = np.mean(np.asarray(invalid[iy:iy + ROI_SIZE:4, x, :], bool))
            if every_band_rate >= 0.70:
                continue
            flank = float(shoulder[x, band])
            if not np.isfinite(waves[band]):
                continue
            width = 1
            for xx in (x-1, x+1):
                if 0 <= xx < w and np.mean(np.asarray(invalid[iy:iy+ROI_SIZE, xx, band], bool)) >= 0.70:
                    width += 1
            val = float(fraction - flank + 0.10 * (1 - every_band_rate))
            candidate = LineCandidate(band, x, int(iy + offset), int(ix), val,
                                      fraction, flank, width, float(waves[band]))
            key = (band, x)
            if key not in best or best[key].score < val:
                best[key] = candidate
    return sorted(best.values(), key=lambda z: z.score, reverse=True)[:max_candidates]


def save_candidates(folder, dataset, candidates):
    path = folder / f"{dataset}_detected_line_candidates.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(LineCandidate.__dataclass_fields__))
        writer.writeheader()
        for row in candidates:
            writer.writerow(asdict(row))
    return path


def preview_candidates(root, dataset, cube, invalid, candidates, offset, dpi):
    """Small contact sheet to verify that selected defects really exist."""
    if not candidates:
        return None
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, min(5, len(candidates)),
                             figsize=(3.4 * min(5, len(candidates)), 4.1), squeeze=False)
    for ax, case in zip(axes[0], candidates[:5]):
        y = case.roi_y - offset
        x0 = case.roi_x
        image = np.asarray(cube[y:y+ROI_SIZE, x0:x0+ROI_SIZE, case.band], np.float32).copy()
        bad = np.asarray(invalid[y:y+ROI_SIZE, x0:x0+ROI_SIZE, case.band], bool)
        good = (~bad) & np.isfinite(image)
        if np.any(good):
            lo, hi = np.percentile(image[good], [1, 99.4])
            image = np.clip((image - lo) / max(hi-lo, 1e-7), 0, 1)
            image[~good] = 0
        else:
            image.fill(0)
        ax.imshow(image, vmin=0, vmax=1, cmap="gray", interpolation="nearest")
        col = case.x - x0
        ax.add_patch(Rectangle((col-1.5, -0.5), 3, ROI_SIZE, fill=False,
                               edgecolor="#efb449", linewidth=1.2, linestyle="--"))
        ax.set_title(f"band {case.band} ({case.wavelength_nm:.1f} nm)\nx={case.x}  y={case.roi_y}", fontsize=10)
        ax.axis("off")
    path = root / f"{dataset}_candidate_preview.png"
    fig.savefig(path, dpi=min(200, dpi), bbox_inches="tight")
    plt.close(fig)
    return path


def line_columns(invalid_roi, band, x_in_roi, forced_width=0):
    if forced_width:
        start = max(0, int(x_in_roi) - forced_width // 2)
        stop = min(ROI_SIZE, start + forced_width)
        return np.arange(start, stop, dtype=int)
    rate = np.asarray(invalid_roi[:, :, band], bool).mean(axis=0)
    if not 0 <= x_in_roi < ROI_SIZE:
        raise ValueError("Chosen line lies outside ROI")
    # Only up to five contiguous defective columns: do not repair broad patches.
    selected = [x_in_roi]
    for direction in (-1, +1):
        x = x_in_roi + direction
        while 0 <= x < ROI_SIZE and len(selected) < 5 and rate[x] >= 0.70:
            selected.append(x)
            x += direction
    return np.asarray(sorted(selected), dtype=int)


def select_case(dataset, cube, invalid, waves, offset, args, auto_candidates):
    suffix = dataset.lower()
    explicit_band = getattr(args, f"band_{suffix}")
    explicit_x = getattr(args, f"line_x_{suffix}")
    explicit_y = getattr(args, f"roi_y_{suffix}")
    h, w, bands = cube.shape
    if (explicit_band is None) != (explicit_x is None):
        raise ValueError(f"{dataset}: supply --band-{suffix} AND --line-x-{suffix}, or neither")
    if explicit_band is None:
        if not auto_candidates:
            raise RuntimeError(f"No automatic narrow missing line detected in {dataset}. "
                               f"Pass --band-{suffix}, --line-x-{suffix} and --roi-y-{suffix} manually.")
        candidate = auto_candidates[0]
        band, x_full = candidate.band, candidate.x
        roi_y_abs = candidate.roi_y if explicit_y is None else explicit_y
    else:
        band, x_full = explicit_band, explicit_x
        matches = [c for c in auto_candidates if c.band == band and abs(c.x - x_full) <= 2]
        roi_y_abs = (matches[0].roi_y if matches else offset + max(0, (h - ROI_SIZE) // 2)) if explicit_y is None else explicit_y
    if not (0 <= band < bands and 0 <= x_full < w):
        raise ValueError(f"{dataset}: --band or --line-x is outside the cube")
    if roi_y_abs < offset or roi_y_abs + ROI_SIZE > offset + h:
        raise ValueError(f"ROI y={roi_y_abs} is outside selected {args.image_scope} image bounds")
    y = roi_y_abs - offset
    roi_x = min(max(x_full - ROI_SIZE // 2, 0), w - ROI_SIZE)
    roi = np.asarray(cube[y:y + ROI_SIZE, roi_x:roi_x + ROI_SIZE, :], np.float32).copy()
    bad = np.asarray(invalid[y:y + ROI_SIZE, roi_x:roi_x + ROI_SIZE, :], bool).copy()
    bad |= ~np.isfinite(roi)
    columns = line_columns(bad, band, x_full - roi_x, args.line_width)
    if not len(columns):
        raise RuntimeError("No line columns were selected")
    line_region = np.zeros((ROI_SIZE, ROI_SIZE), bool)
    line_region[:, columns] = True
    target = bad[:, :, band] & line_region
    target_count = int(target.sum())
    if target_count < args.min_line_fraction * ROI_SIZE * len(columns):
        raise RuntimeError(f"{dataset} band={band} x={x_full}: only {target_count} out of "
                           f"{ROI_SIZE * len(columns)} line samples are invalid in this ROI. "
                           "Change ROI, band or --min-line-fraction.")
    # The original raw mask is respected everywhere, but the experiment repairs
    # only the genuine invalid samples belonging to this selected detector line.
    roi[bad] = np.nan
    coords_yx = np.argwhere(target)
    coords = np.column_stack((coords_yx, np.full(len(coords_yx), band, np.int32))).astype(np.int32)
    return {
        "dataset": dataset, "band": int(band), "wavelength_nm": float(waves[band]),
        "line_x_full": int(x_full), "roi_y_full": int(roi_y_abs),
        "roi_x_full": int(roi_x), "line_columns_roi": columns.tolist(),
        "actual_missing_voxels": int(target_count), "line_total_samples": int(ROI_SIZE*len(columns)),
        "missing_fraction": float(target_count/(ROI_SIZE*len(columns))),
    }, roi, bad, coords


def make_environment(spec, dataset):
    from _launcher_common import env_for
    env = env_for(spec, spec.primary_test)
    env["TPSSR_FUSION_DONOR_MATCH"] = "robust"
    env["PYTHONPATH"] = os.pathsep.join([str(CORE), str(ROOT), env.get("PYTHONPATH", "")])
    return env


def worker(args):
    if not args.workdir or not args.worker_dataset:
        raise ValueError("Worker requires --workdir and --worker-dataset")
    from experiment_config import resolve_experiment
    from types import SimpleNamespace
    from _launcher_common import env_for
    spec = resolve_experiment(SimpleNamespace(preset="swir_8x2", dataset=None,
                                               active_test=None, train_datasets=None,
                                               test_datasets=None, experiment_name=None))
    os.environ.update(make_environment(spec, args.worker_dataset))
    if str(CORE) not in sys.path:
        sys.path.insert(0, str(CORE))
    import torch
    import torch.distributed as dist
    from src.pipeline import load_pipeline, predict_mask_learned_router3
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available() or local >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA GPU {local} is not accessible; use --gpus 1 if only one GPU works")
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")
    work = args.workdir / "_work"
    cube = np.load(work / "roi_cube.npy", mmap_mode="r")
    mask = np.load(work / "model_mask.npy", mmap_mode="r")
    coords = np.load(work / "coords.npy")
    if not len(coords) or not bool(np.all(mask[coords[:, 0], coords[:, 1], coords[:, 2]])):
        raise RuntimeError("Missing-line target indices do not match original invalid mask")
    # Keep all damaged bands of a spatial pixel assigned to the same GPU.
    _, reverse = np.unique(coords[:, :2], axis=0, return_inverse=True)
    indices = np.flatnonzero((reverse % world) == rank)
    distributed = world > 1
    if distributed:
        dist.init_process_group(backend="gloo")
    checkpoint = (load_pipeline(device, region_batch=args.region_batch,
                                verify_region_batch=True, learned_fusion=True,
                                learned_router3_lineaware_rankcv=True)
                  if len(indices) else None)
    counts = np.zeros(3, dtype=np.int64)
    if len(indices):
        with torch.inference_mode():
            route_only = predict_mask_learned_router3(checkpoint, cube, mask,
                                                       coords[indices], routing_only=True)
        counts = np.asarray(route_only["raw_route_counts"], np.int64)
    if distributed:
        counts_tensor = torch.from_numpy(counts.copy())
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
        counts = counts_tensor.numpy()
    expected = len(np.unique(coords[:, :2], axis=0))
    if int(counts.sum()) != expected:
        raise RuntimeError(f"Router pixel-count check failed: {counts.sum()} != {expected}")
    if rank == 0:
        print(f"[RankCV] Global expert pixel counts: {counts.tolist()}", flush=True)
    if len(indices):
        with torch.inference_mode():
            prediction = predict_mask_learned_router3(
                checkpoint, cube, mask, coords[indices],
                region_progress=f"GPU {local} real detector-line reconstruction",
                global_minority_threshold=0.10, global_minority_counts=counts,
                global_minority_policy="hybrid_gap", three_expert_min_gap=0.10)
        values = np.asarray(prediction["prediction"], np.float32)
        route_lookup = {"router3_interp": 0, "router3_fusion": 1, "router3_region": 2}
        routes = np.asarray([route_lookup[str(s)] for s in prediction["candidate_method"]], np.int8)
        if values.shape != (len(indices),) or not np.all(np.isfinite(values)):
            raise RuntimeError("TPSSR produced missing/non-finite predictions")
    else:
        values = np.empty(0, np.float32)
        routes = np.empty(0, np.int8)
    np.savez_compressed(work / f"rank_{rank}.npz", indices=indices,
                        values=values, routes=routes)
    if distributed:
        dist.destroy_process_group()
    print(f"[GPU {local}] {len(indices)} real missing voxels reconstructed", flush=True)


def merge_predictions(folder, n_gpus):
    work = folder / "_work"
    coords = np.load(work / "coords.npy")
    values = np.full(len(coords), np.nan, np.float32)
    routes = np.full(len(coords), -1, np.int8)
    done = np.zeros(len(coords), bool)
    for rank in range(n_gpus):
        with np.load(work / f"rank_{rank}.npz") as r:
            take, v, methods = r["indices"], r["values"], r["routes"]
        if len(take) != len(v) or len(take) != len(methods):
            raise RuntimeError("GPU indices, values and routes lengths differ")
        if np.any(done[take]) or np.any(take < 0) or np.any(take >= len(coords)):
            raise RuntimeError("Invalid or duplicated GPU indices")
        done[take] = True
        values[take], routes[take] = v, methods
    if not np.all(done) or not np.all(np.isfinite(values)) or np.any(routes < 0):
        raise RuntimeError("Not all genuine missing-line voxels were reconstructed")
    return coords, values, routes


def display_planes(original, repaired, band):
    observed = original[:, :, band]
    valid = np.isfinite(observed)
    if int(valid.sum()) < 200:
        raise RuntimeError("Too few observed pixels to determine consistent display contrast")
    low, high = np.percentile(observed[valid], [1.0, 99.4])
    high = max(float(high), float(low) + 1e-7)
    def normalize(plane):
        plane = (np.asarray(plane, np.float32) - low) / (high - low)
        plane = np.clip(plane, 0.0, 1.0)
        plane[~np.isfinite(plane)] = 0.0
        return plane
    return normalize(observed), normalize(repaired[:, :, band]), [float(low), float(high)]


def plot_case(folder, metadata, original, repaired, dpi):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if any(f.name == "Times New Roman" for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = "Times New Roman"
    else:
        plt.rcParams["font.family"] = "serif"
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.facecolor": "white"})
    band = metadata["band"]
    before, after, limits = display_planes(original, repaired, band)
    x1, x2 = min(metadata["line_columns_roi"]), max(metadata["line_columns_roi"])
    xc = (x1 + x2) // 2
    # A narrow vertical detail crop makes a 1-pixel detector defect legible in print.
    xmin = max(0, xc - 16)
    xmax = min(ROI_SIZE, xc + 17)
    if xmax - xmin < 32:
        xmin = max(0, xmax - 32)
    ystart, yend = 38, 90
    tiles = [before, after, before[ystart:yend, xmin:xmax], after[ystart:yend, xmin:xmax]]
    titles = ["Observed band", "TPSSR reconstruction", "Observed (detail)", "TPSSR (detail)"]
    fig, axes = plt.subplots(1, 4, figsize=(15.8, 4.15), layout="constrained")
    for i, (ax, img, title) in enumerate(zip(axes, tiles, titles)):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
        ax.set_title(title, pad=8)
        ax.axis("off")
    fig.suptitle(f"{metadata['dataset']}  |  band {band} ({metadata['wavelength_nm']:.1f} nm)", fontsize=14)
    figure = folder / f"{metadata['dataset']}_real_vertical_line.png"
    fig.savefig(figure, dpi=dpi, bbox_inches="tight")
    fig.savefig(figure.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    metadata["display_limits_from_observed_band"] = limits
    # Preserve pixels and target mask to make the article figure reproducible.
    np.save(folder / "roi_reconstructed_tpssr.npy", repaired)
    np.save(folder / "roi_original_missing.npy", original)
    return tiles


def plot_combined(root, records, dpi):
    if len(records) != 2:
        return None
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if any(f.name == "Times New Roman" for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = "Times New Roman"
    else:
        plt.rcParams["font.family"] = "serif"
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 8.25), layout="constrained")
    headers = ["Observed band", "TPSSR reconstruction", "Observed (detail)", "TPSSR (detail)"]
    for row, (meta, tiles) in enumerate(records):
        for col, (img, title) in enumerate(zip(tiles, headers)):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
            ax.axis("off")
            if row == 0:
                ax.set_title(title, pad=10)
        axes[row, 0].text(-0.07, 0.5, f"{meta['dataset']}\nband {meta['band']}\n{meta['wavelength_nm']:.1f} nm",
                          transform=axes[row, 0].transAxes, ha="right", va="center",
                          fontsize=11, rotation=90)
    outfile = root / "Figure_3_5_Real_SWIR_vertical_lines.png"
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return outfile



def prepare_full_line_coords(folder, meta, cube, invalid):
    """Gather every genuinely missing sample along the selected detector line.

    Keep all targets together rather than dividing the full image into 64-row
    chunks. The TPSSR Region expert performs its own contextual patching.
    """
    h, w, bands = cube.shape
    band = int(meta["band"])
    x0 = int(meta["roi_x_full"])
    columns = np.asarray(meta["line_columns_roi"], np.int64) + x0
    if np.any(columns < 0) or np.any(columns >= w):
        raise ValueError("Detector line outside spatial bounds")
    missing = (np.asarray(invalid[:, columns, band], bool) |
               ~np.isfinite(np.asarray(cube[:, columns, band])))
    iy, ic = np.nonzero(missing)
    if not len(iy):
        raise RuntimeError("No genuine missing values in selected detector line")
    coords = np.column_stack((iy, columns[ic], np.full(len(iy), band))).astype(np.int32)
    work = folder / "_work"
    work.mkdir(parents=True, exist_ok=True)
    np.save(work / "full_line_coords.npy", coords)
    manifest = {
        "dataset": meta["dataset"], "image_scope": meta["image_scope"],
        "image_shape": list(cube.shape), "target_band": band,
        "line_columns_full": columns.tolist(), "target_count": int(len(coords)),
        "worker_partition": "spatial_pixel_round_robin",
        "outer_spatial_tiling": False,
    }
    (work / "full_line_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def full_line_worker(args):
    """One full-line inference call per GPU; Global Minority uses a short prepass."""
    if not args.workdir or not args.worker_dataset:
        raise ValueError("Worker requires --workdir and --worker-dataset")
    from experiment_config import resolve_experiment
    from types import SimpleNamespace
    spec = resolve_experiment(SimpleNamespace(
        preset="swir_8x2", dataset=None, active_test=None,
        train_datasets=None, test_datasets=None, experiment_name=None))
    os.environ.update(make_environment(spec, args.worker_dataset))
    if str(CORE) not in sys.path:
        sys.path.insert(0, str(CORE))
    import torch
    import torch.distributed as dist
    from src.pipeline import load_pipeline, predict_mask_learned_router3

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA GPU {local_rank} is not accessible")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    work = args.workdir / "_work"
    manifest = json.loads((work / "full_line_manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset"] != args.worker_dataset or manifest["image_scope"] != args.image_scope:
        raise RuntimeError("Dataset/scope differ from prepared full-line coordinates")
    coords = np.load(work / "full_line_coords.npy")
    cube_raw, invalid_raw, _, _ = load_source(args.worker_dataset, args.image_scope)
    if list(cube_raw.shape) != manifest["image_shape"]:
        raise RuntimeError("The source cube dimensions changed")

    # Preserve original invalid data as missing everywhere, just as tile_inputs
    # did previously. RAM is preferable to reloading dozens of spatial tiles.
    cube = np.array(cube_raw, dtype=np.float32, copy=True)
    mask = np.asarray(invalid_raw, dtype=bool).copy()
    mask |= ~np.isfinite(cube)
    cube[mask] = np.nan
    if not np.all(mask[coords[:, 0], coords[:, 1], coords[:, 2]]):
        raise RuntimeError("Line includes a non-missing coordinate")

    # All bands of the same spatial pixel stay on the same worker.
    _, inverse = np.unique(coords[:, :2], axis=0, return_inverse=True)
    local_indices = np.flatnonzero((inverse % world) == rank)
    if world > 1:
        dist.init_process_group(backend="gloo")
    pipeline = load_pipeline(device, region_batch=args.region_batch,
                             verify_region_batch=True, learned_fusion=True,
                             learned_router3_lineaware_rankcv=True)

    counts = np.zeros(3, dtype=np.int64)
    if local_indices.size:
        with torch.inference_mode():
            preliminary = predict_mask_learned_router3(
                pipeline, cube, mask, coords[local_indices], routing_only=True)
        counts = np.asarray(preliminary["raw_route_counts"], np.int64)
    if world > 1:
        global_tensor = torch.from_numpy(counts.copy())
        dist.all_reduce(global_tensor, op=dist.ReduceOp.SUM)
        counts = global_tensor.numpy()
    if int(counts.sum()) != len(coords):
        raise RuntimeError("Incorrect global RankCV route counts")
    if rank == 0:
        print(f"[RankCV] Full-line expert counts {counts.tolist()}", flush=True)

    if local_indices.size:
        with torch.inference_mode():
            result = predict_mask_learned_router3(
                pipeline, cube, mask, coords[local_indices],
                region_progress=f"GPU {local_rank} real full-line Region",
                global_minority_threshold=0.10,
                global_minority_counts=counts,
                global_minority_policy="hybrid_gap", three_expert_min_gap=0.10)
        values = np.asarray(result["prediction"], np.float32)
        route_map = {"router3_interp": 0, "router3_fusion": 1, "router3_region": 2}
        routes = np.asarray([route_map[str(s)] for s in result["candidate_method"]], np.int8)
        if values.shape != (len(local_indices),) or not np.isfinite(values).all():
            raise RuntimeError("Incomplete full-line TPSSR predictions")
    else:
        values = np.empty(0, np.float32)
        routes = np.empty(0, np.int8)
    np.savez_compressed(work / f"rank_{rank}.npz",
                        indices=local_indices, values=values, routes=routes)
    if world > 1:
        dist.destroy_process_group()
    print(f"[GPU {local_rank}] One complete inference, {len(local_indices)} line voxels", flush=True)


def merge_full_line_predictions(manifest, workdir, cube, invalid, n_gpus):
    coords = np.load(workdir / "full_line_coords.npy")
    band = int(manifest["target_band"])
    original = np.array(cube[:, :, band], dtype=np.float32, copy=True)
    original[np.asarray(invalid[:, :, band], dtype=bool)] = np.nan
    original[~np.isfinite(original)] = np.nan
    repaired = original.copy()
    target = np.zeros(original.shape, dtype=bool)
    done = np.zeros(len(coords), bool)
    expert_counts = np.zeros(3, np.int64)
    for rank in range(n_gpus):
        file = workdir / f"rank_{rank}.npz"
        if not file.is_file():
            raise RuntimeError(f"GPU {rank} prediction file missing")
        with np.load(file) as data:
            indices = np.asarray(data["indices"], np.int64)
            values = np.asarray(data["values"], np.float32)
            routes = np.asarray(data["routes"], np.int8)
        if (len(indices) != len(values) or len(indices) != len(routes) or
                np.any(indices < 0) or np.any(indices >= len(coords)) or
                np.any(done[indices]) or not np.isfinite(values).all() or
                np.any((routes < 0) | (routes > 2))):
            raise RuntimeError("Incomplete, repeated or invalid GPU results")
        done[indices] = True
        ys = coords[indices, 0]
        xs = coords[indices, 1]
        target[ys, xs] = True
        repaired[ys, xs] = values
        expert_counts += np.bincount(routes, minlength=3)
    if not np.all(done) or int(target.sum()) != manifest["target_count"]:
        raise RuntimeError("The full detector line was not fully reconstructed")
    if not np.array_equal(original[~target], repaired[~target], equal_nan=True):
        raise RuntimeError("An observed or non-target value changed")
    return original, repaired, target, expert_counts


def raw_calibrated_display_band(dataset, target_band, shape, raw_dir):
    """Reproduce the 3D front-view raw ENVI calibration for one band only."""
    from src.envi_io import EnviCube
    root = raw_dir if raw_dir is not None else ROOT / "input_raw"
    if raw_dir is None and not root.exists():
        root = ROOT / "Input_raw"
    dataset_dir = next((p for p in (root / "test" / dataset,
                                   root / "train" / dataset,
                                   root / dataset) if p.is_dir()), root / dataset)
    headers = sorted(dataset_dir.rglob("*.hdr")) + sorted(dataset_dir.rglob("*.HDR"))
    main_hdr = [p for p in headers if "whiteref" not in p.name.lower()
                and "darkref" not in p.name.lower()]
    white_hdr = [p for p in headers if "whiteref" in p.name.lower()]
    dark_hdr = [p for p in headers if "darkref" in p.name.lower()]
    if len(main_hdr) != 1 or len(white_hdr) != 1 or len(dark_hdr) != 1:
        raise RuntimeError(f"Expected one main/white/dark ENVI file in {dataset_dir}; got "
                           f"{len(main_hdr)}, {len(white_hdr)}, {len(dark_hdr)}")
    main, white, dark = (EnviCube(p) for p in (main_hdr[0], white_hdr[0], dark_hdr[0]))
    if (main.lines, main.samples, main.bands) != tuple(shape):
        raise ValueError("Raw ENVI and processed cube dimensions are inconsistent")
    if white.samples != main.samples or white.bands != main.bands or dark.samples != main.samples or dark.bands != main.bands:
        raise ValueError("White/dark ENVI references do not match main acquisition")
    band = int(target_band)
    # Only read the requested band, but keep the same row-wise mean reference
    # calibration as render_hsi_cube_front.py's full-band implementation.
    raw = np.asarray(main.yxb()[:, :, band], np.float32)
    wr = np.asarray(white.yxb()[:, :, band], np.float32).mean(axis=0)[None, :]
    dr = np.asarray(dark.yxb()[:, :, band], np.float32).mean(axis=0)[None, :]
    denom = np.where(np.abs(wr - dr) > 1e-12, wr - dr, np.nan)
    return (raw - dr) / denom


def display_to_uint(plane, lo, hi, gamma, brightness, sixteen_bit):
    """Exactly the percentile -> gamma -> brightness sequence of 3D front."""
    scaled = np.clip((np.nan_to_num(plane, nan=0.0, posinf=1.0, neginf=0.0) - lo)
                     / (hi - lo), 0.0, 1.0)
    scaled = np.clip(np.power(scaled, gamma) * brightness, 0.0, 1.0)
    if sixteen_bit:
        return np.rint(scaled * 65535).astype(np.uint16)
    return np.rint(scaled * 255).astype(np.uint8)


def save_individual_full_band_images(root, folder, meta, original, repaired, target, dpi, args):
    """Export two independent full-resolution 2D images, without annotations."""
    from PIL import Image
    dataset = meta["dataset"]
    source = getattr(args, f"front_source_{dataset.lower()}")
    gamma = float(getattr(args, f"front_gamma_{dataset.lower()}"))
    brightness = float(getattr(args, f"front_brightness_{dataset.lower()}"))
    if gamma <= 0 or brightness <= 0 or not (0 <= args.front_low_pct < args.front_high_pct <= 100):
        raise ValueError("Invalid gamma / brightness / percentile values")
    if source == "raw-calibrated":
        if args.image_scope != "full":
            raise ValueError("raw-calibrated display currently requires --image-scope full")
        display_cache = folder / "_work" / "raw_display_band.npy"
        raw_band = (np.load(display_cache, mmap_mode="r") if display_cache.is_file()
                    else raw_calibrated_display_band(dataset, int(meta["band"]),
                                                     meta["source_image_shape"], args.raw_dir))
        observed = np.isfinite(original) & np.isfinite(raw_band)
        if np.count_nonzero(observed) < 200:
            raise RuntimeError("Too few paired observed values in raw and processed images")
        diffs = np.abs(raw_band[observed] - original[observed])
        # Same radiometry is required before mixing raw-calibrated observed
        # pixels with processed-domain TPSSR values.
        p99 = float(np.percentile(diffs, 99))
        span = float(np.percentile(raw_band[observed], 99) - np.percentile(raw_band[observed], 1))
        if p99 > max(2e-3, 0.05 * max(span, 1e-6)):
            raise RuntimeError("Raw and processed calibration do not match sufficiently to overlay "
                               f"the reconstruction (99th-pct abs diff={p99:.5g}). "
                               "Use --front-source-swir10 processed or inspect radiometry.")
        display_before = raw_band.copy()
        display_before[~np.isfinite(original)] = np.nan
        display_after = display_before.copy()
        display_after[target] = repaired[target]
        # Raw front renderer first sets invalid raw values to zero and then
        # calculates the percentiles over the whole image (including zeros).
        limits_source = np.nan_to_num(raw_band, nan=0.0, posinf=1.0, neginf=0.0)
        lo, hi = np.percentile(limits_source, [args.front_low_pct, args.front_high_pct])
        calibration_difference = p99
    else:
        display_before = original
        display_after = repaired
        observed = np.isfinite(original)
        if np.count_nonzero(observed) < 200:
            raise RuntimeError("Too few observed reflectance values")
        lo, hi = np.percentile(original[observed],
                               [args.front_low_pct, args.front_high_pct])
        calibration_difference = None
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        raise RuntimeError("Degenerate percentile display range")
    print(f"[{dataset}] 2D display: source={source}, gamma={gamma:g}, "
          f"brightness={brightness:g}, percentiles={args.front_low_pct:g}/{args.front_high_pct:g}, "
          f"limits=({lo:.5g}, {hi:.5g})", flush=True)
    stem = f"{dataset}_band{meta['band']}_x{meta['line_x_full']}"
    paths = {}
    for label, image in (("before", display_before), ("TPSSR", display_after)):
        png_path = root / f"{stem}_{label}.png"
        tif_path = root / f"{stem}_{label}.tif"
        Image.fromarray(display_to_uint(image, lo, hi, gamma, brightness, False)).save(
            png_path, dpi=(dpi, dpi))
        Image.fromarray(display_to_uint(image, lo, hi, gamma, brightness, True)).save(
            tif_path, dpi=(dpi, dpi), compression="tiff_deflate")
        paths[label] = str(png_path)
    np.save(folder / "full_band_original.npy", original)
    np.save(folder / "full_band_reconstructed.npy", repaired)
    np.save(folder / "real_line_target_mask.npy", target)
    meta.update({"publication_png_files": paths, "display_source": source,
                 "gamma": gamma, "brightness": brightness,
                 "display_limits": [lo, hi], "raw_processed_abs_diff_p99": calibration_difference,
                 "full_image_size_yx": list(original.shape),
                 "publication_notes": "Unannotated 2D full-band, PNG8 and TIFF16; shared display limits"})
    return paths


def run_full_image(dataset, cube, invalid, waves, offset, args, candidates, spec):
    meta, _, _, _ = select_case(dataset, cube, invalid, waves, offset, args, candidates)
    meta.update({"preset": args.preset, "image_scope": args.image_scope,
                 "role": "real_missing_data_no_ground_truth",
                 "evaluation": "no_RMSE_possible_on_unobserved_values",
                 "checkpoint_model_tag": spec.primary_test,
                 "figure_extent": "full_original_acquisition_band",
                 "source_image_shape": list(cube.shape)})
    folder = args.outdir / f"{dataset}_band{meta['band']}_x{meta['line_x_full']}_FAST_FULL"
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError(f"Output folder is nonempty: {folder}. Choose a different --outdir.")
    source = getattr(args, f"front_source_{dataset.lower()}")
    raw_display = None
    if source == "raw-calibrated":
        if args.image_scope != "full":
            raise ValueError("raw-calibrated display needs --image-scope full")
        # Validate raw source and its alignment BEFORE the expensive GPU run.
        raw_display = raw_calibrated_display_band(
            dataset, int(meta["band"]), meta["source_image_shape"], args.raw_dir)
        original_plane = np.asarray(cube[:, :, int(meta["band"])], np.float32)
        observed = (~np.asarray(invalid[:, :, int(meta["band"])], bool) &
                    np.isfinite(original_plane) & np.isfinite(raw_display))
        if int(observed.sum()) < 200:
            raise RuntimeError("Too few valid raw/processed pairs to verify display calibration")
        absdiff = np.abs(raw_display[observed] - original_plane[observed])
        p99 = float(np.percentile(absdiff, 99))
        span = float(np.percentile(raw_display[observed], 99) -
                     np.percentile(raw_display[observed], 1))
        if p99 > max(2e-3, 0.05 * max(span, 1e-6)):
            raise RuntimeError(f"Raw and processed band calibration differ (p99={p99:.6g}); "
                               f"try --front-source-{dataset.lower()} processed")
        print(f"[{dataset}] Raw/processed 99th-percentile difference {p99:.3g}: "
              "calibration alignment passed", flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = prepare_full_line_coords(folder, meta, cube, invalid)
    if raw_display is not None:
        np.save(folder / "_work" / "raw_display_band.npy", raw_display)
        del raw_display
    meta["line_columns_full"] = manifest["line_columns_full"]
    meta["full_line_missing_voxels"] = manifest["target_count"]
    meta["outer_spatial_tiling"] = False
    print(f"[{dataset}] Full-image, ONE inference per GPU: band {meta['band']} "
          f"({meta['wavelength_nm']:.1f} nm), detector columns {manifest['line_columns_full']}, "
          f"{manifest['target_count']} missing voxels; {args.gpus} GPU(s)", flush=True)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={args.gpus}", str(Path(__file__).resolve()),
               "--full-line-worker", "--worker-dataset", dataset,
               "--image-scope", args.image_scope,
               "--workdir", str(folder), "--region-batch", str(args.region_batch)]
    subprocess.run(command, cwd=str(ROOT), env=make_environment(spec, dataset), check=True)
    original, repaired, target, routes = merge_full_line_predictions(
        manifest, folder / "_work", cube, invalid, args.gpus)
    paths = save_individual_full_band_images(
        args.outdir, folder, meta, original, repaired, target, args.dpi, args)
    meta["reconstructed_target_voxels"] = int(target.sum())
    meta["coverage_on_selected_full_line"] = float(np.isfinite(repaired[target]).mean())
    meta["expert_target_counts"] = dict(zip(("spectral", "fusion", "region"),
                                            [int(x) for x in routes]))
    meta["generated_at"] = datetime.now().isoformat(timespec="seconds")
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{dataset}] Complete: {int(target.sum())} line voxels; full-size 2D images", flush=True)
    for name, filename in paths.items():
        print(f"  {name}: {filename}", flush=True)
    return meta

def main(args):
    if not all(np.isfinite(float(v)) and float(v) > 0 for v in
               (args.front_gamma_swir5, args.front_gamma_swir10,
                args.front_brightness_swir5, args.front_brightness_swir10)):
        raise ValueError("Gamma and brightness must be positive")
    if not (0 <= args.front_low_pct < args.front_high_pct <= 100):
        raise ValueError("Invalid display percentiles")
    if not (CORE / "config.py").is_file():
        raise SystemExit(f"Place this script in the TPSSR project root alongside core/: {ROOT}")
    env, spec = experiment_environment("SWIR5")
    os.environ.update(env)
    if str(CORE) not in sys.path:
        sys.path.insert(0, str(CORE))
    args.outdir.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset in args.datasets:
        print(f"\n[{dataset}] Loading real calibrated cube + acquisition invalid mask", flush=True)
        cube, invalid, waves, offset = load_source(dataset, args.image_scope)
        candidates = detect_lines(invalid, waves, offset, max_candidates=args.max_candidates)
        csvfile = save_candidates(args.outdir, dataset, candidates)
        preview_path = preview_candidates(args.outdir, dataset, cube, invalid, candidates, offset, args.dpi)
        print(f"[{dataset}] Candidate list: {csvfile} ({len(candidates)} found)", flush=True)
        if preview_path is not None:
            print(f"[{dataset}] Candidate preview: {preview_path}", flush=True)
        for c in candidates[:5]:
            print(f"    band={c.band:3d}, wavelength={c.wavelength_nm:.1f} nm, "
                  f"x={c.x:4d}, ROI y={c.roi_y:5d}, missing={c.missing_fraction:.0%}", flush=True)
        if args.list_only:
            continue
        if not args.roi_only:
            run_full_image(dataset, cube, invalid, waves, offset, args, candidates, spec)
            del cube, invalid
            continue
        meta, roi, model_mask, coords = select_case(dataset, cube, invalid, waves, offset, args, candidates)
        del cube, invalid
        folder = args.outdir / f"{dataset}_band{meta['band']}_x{meta['line_x_full']}_y{meta['roi_y_full']}"
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError(f"Output folder already exists and is nonempty: {folder}\n"
                               "Move/delete it or choose another --outdir (existing results were not overwritten).")
        folder.mkdir(parents=True, exist_ok=True)
        work = folder / "_work"
        work.mkdir(exist_ok=True)
        np.save(work / "roi_cube.npy", roi)
        np.save(work / "model_mask.npy", model_mask)
        np.save(work / "coords.npy", coords)
        np.save(work / "original_line_target.npy", np.isin(np.arange(ROI_SIZE)[None, :], meta["line_columns_roi"]) & model_mask[:, :, meta["band"]])
        meta["preset"] = args.preset
        meta["image_scope"] = args.image_scope
        meta["role"] = "real_missing_data_no_ground_truth"
        meta["evaluation"] = "no_RMSE_possible_on_unobserved_values"
        meta["checkpoint_model_tag"] = spec.primary_test
        print(f"[{dataset}] Band {meta['band']} ({meta['wavelength_nm']:.1f} nm), "
              f"real line x={meta['line_x_full']}, ROI y={meta['roi_y_full']}, "
              f"targets={len(coords)} | using TPSSR RankCV, hybrid-gap 10%, {args.gpus} GPU(s)", flush=True)
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={args.gpus}", str(Path(__file__).resolve()),
               "--worker", "--worker-dataset", dataset,
               "--workdir", str(folder), "--region-batch", str(args.region_batch)]
        subprocess.run(cmd, cwd=str(ROOT), env=make_environment(spec, dataset), check=True)
        _, values, routes = merge_predictions(folder, args.gpus)
        repaired = roi.copy()
        repaired[coords[:, 0], coords[:, 1], coords[:, 2]] = values
        # All truly observed pixels must be strictly preserved. Nothing outside the line is inpainted.
        if not np.array_equal(repaired[~model_mask], roi[~model_mask], equal_nan=True):
            raise RuntimeError("An observed pixel was altered - refusing to publish figure")
        meta["reconstructed_target_voxels"] = int(len(coords))
        meta["coverage_on_selected_line"] = float(np.isfinite(values).mean())
        meta["expert_target_counts"] = {
            "spectral": int((routes == 0).sum()), "fusion": int((routes == 1).sum()),
            "region": int((routes == 2).sum())}
        meta["generated_at"] = datetime.now().isoformat(timespec="seconds")
        tiles = plot_case(folder, meta, roi, repaired, args.dpi)
        (folder / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        records.append((meta, tiles))
        print(f"[{dataset}] Saved publication figures to {folder}", flush=True)
    if args.list_only:
        print("\nCandidate lists only. Inspect CSVs and rerun without --list-only.", flush=True)
    else:
        if args.roi_only:
            combined = plot_combined(args.outdir, records, args.dpi)
            if combined:
                print(f"\nCombined article figure: {combined}", flush=True)
        else:
            print("\nSaved separate full-size, unannotated before/TPSSR PNG + TIFF images.", flush=True)
        print("Note: No RMSE is reported because real missing values have no ground truth.", flush=True)


if __name__ == "__main__":
    cli = parser().parse_args()
    if cli.full_line_worker:
        full_line_worker(cli)
    elif cli.worker:
        worker(cli)
    else:
        main(cli)
