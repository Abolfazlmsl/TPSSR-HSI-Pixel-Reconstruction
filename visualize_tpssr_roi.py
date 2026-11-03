#!/usr/bin/env python3
"""Interactive fixed-size ROI reconstruction with the trained TPSSR router.

Run this from the project root. The held-out test block is shown as a 2-D
preview; the square ROI can be moved but not resized. A local synthetic defect
is reconstructed with the same trained inference entry point as test.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"
ROI_SIZE = 128
DEFECT_SIZE = 31
ROUTER = "learned_router3_aligned"
SCENARIOS = (
    "spatial_patch", "spatio_spectral_block", "whole_spectrum_pixels",
    "mixed_saturation", "band_wide_saturation", "consecutive_spectral_bands",
    "isolated_voxels", "single_detector_line", "multiple_detector_lines",
    "partial_vertical_lines",
)


def arguments():
    p = argparse.ArgumentParser(description="Fixed 128x128 draggable ROI + real TPSSR reconstruction")
    p.add_argument("--preset", choices=("swir_8x2", "vnir_8x2"), default="swir_8x2")
    p.add_argument("--dataset", default="SWIR5", help="Held-out dataset, e.g. SWIR5 or SWIR10")
    p.add_argument("--scenario", choices=SCENARIOS, default="spatio_spectral_block")
    p.add_argument("--severity", type=int, choices=range(1, 6), default=5)
    p.add_argument("--gpus", type=int, choices=(1, 2), default=2)
    p.add_argument("--band", type=int, default=None, help="Zero-based band index; default: middle band")
    p.add_argument("--seed", type=int, default=4207)
    p.add_argument("--outdir", default=None)
    p.add_argument("--roi-y", type=int, default=None, help="Fixed ROI top coordinate inside the held-out test block")
    p.add_argument("--roi-x", type=int, default=None, help="Fixed ROI left coordinate inside the held-out test block")
    p.add_argument("--region-batch", type=int, default=16)
    p.add_argument("--preview-max-height", type=int, default=1300)
    p.add_argument("--preview-max-width", type=int, default=1000)
    p.add_argument("--select-only", action="store_true", help="Save a selection/mask preview; do not run the model")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def boundaries(y, x, height, width, size=ROI_SIZE):
    if height < size or width < size:
        raise ValueError(f"Test image is {height}x{width}; the fixed ROI requires at least {size}x{size}.")
    return max(0, min(int(y), height - size)), max(0, min(int(x), width - size))


def preview_image(test_cube, invalid_cube, band, max_h=1300, max_w=1000):
    h, w, _ = test_cube.shape
    step_y = max(1, math.ceil(h / max_h))
    step_x = max(1, math.ceil(w / max_w))
    values = np.asarray(test_cube[::step_y, ::step_x, band], dtype=np.float32).copy()
    bad = np.asarray(invalid_cube[::step_y, ::step_x, band], dtype=bool)
    good = np.isfinite(values) & ~bad
    if int(good.sum()) < 100:
        raise ValueError("Too few valid pixels in this preview band. Try --band with another index.")
    lo, hi = np.percentile(values[good], [1.0, 99.3])
    if hi <= lo:
        hi = lo + 1e-6
    image = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    image[~good] = 0.0
    return image, step_y, step_x


def select_box(image, height, width, step_y, step_x, dataset, band):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if plt.get_backend().lower() in {"agg", "template", "pdf", "svg", "ps", "cairo"} or "inline" in plt.get_backend().lower():
        raise RuntimeError("Matplotlib has a non-interactive backend. Run in Linux desktop/RDP, or pass --roi-y and --roi-x.")

    ry, rx = boundaries((height - ROI_SIZE) // 2, (width - ROI_SIZE) // 2, height, width)
    state = {"y": ry, "x": rx, "drag": None, "confirmed": False}
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest", origin="upper")
    r = Rectangle((rx / step_x - 0.5, ry / step_y - 0.5), ROI_SIZE / step_x,
                  ROI_SIZE / step_y, fill=False, lw=2.0, edgecolor="yellow")
    ax.add_patch(r)
    ax.set_title(f"{dataset} | test block | band {band}\nDrag yellow square or click to reposition; Enter = confirm, Esc = cancel", fontsize=11)
    ax.set_xlabel("Preview column")
    ax.set_ylabel("Preview row")
    txt = fig.text(0.02, 0.01, "", fontsize=10)

    def set_position(y, x):
        yy, xx = boundaries(y, x, height, width)
        state["y"], state["x"] = yy, xx
        r.set_xy((xx / step_x - 0.5, yy / step_y - 0.5))
        txt.set_text(f"ROI: x={xx}:{xx + ROI_SIZE}  y={yy}:{yy + ROI_SIZE}  | fixed {ROI_SIZE} x {ROI_SIZE} pixels")
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes is not ax or event.xdata is None or event.ydata is None or event.button != 1:
            return
        mx, my = int(round(event.xdata * step_x)), int(round(event.ydata * step_y))
        inside = (state["x"] <= mx <= state["x"] + ROI_SIZE and
                  state["y"] <= my <= state["y"] + ROI_SIZE)
        if inside:
            state["drag"] = (mx - state["x"], my - state["y"])
        else:
            state["drag"] = (ROI_SIZE // 2, ROI_SIZE // 2)
            set_position(my - ROI_SIZE // 2, mx - ROI_SIZE // 2)

    def on_motion(event):
        if state["drag"] is None or event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        dx, dy = state["drag"]
        set_position(int(round(event.ydata * step_y)) - dy,
                     int(round(event.xdata * step_x)) - dx)

    def on_release(_event):
        state["drag"] = None

    def on_key(event):
        if event.key in ("enter", "return", " "):
            state["confirmed"] = True
            plt.close(fig)
        elif event.key == "escape":
            plt.close(fig)
        elif event.key in ("left", "right", "up", "down"):
            dy, dx = {"left": (0, -8), "right": (0, 8), "up": (-8, 0), "down": (8, 0)}[event.key]
            set_position(state["y"] + dy, state["x"] + dx)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    set_position(ry, rx)
    plt.show()
    if not state["confirmed"]:
        raise SystemExit("ROI selection cancelled. No reconstruction was launched.")
    return state["y"], state["x"]


def make_local_mask(shape, scenario, severity, band, seed):
    if shape[0] != ROI_SIZE or shape[1] != ROI_SIZE:
        raise ValueError("Unexpected ROI dimensions")
    if not 0 <= band < shape[2]:
        raise ValueError("Band index outside cube")
    if str(scenario) not in SCENARIOS:
        raise ValueError("Invalid scenario")
    if not 1 <= int(severity) <= 5:
        raise ValueError("Invalid severity")

    from src.training_corruptions import generate_training_mask
    focus = min(DEFECT_SIZE, ROI_SIZE)
    y0 = (ROI_SIZE - focus) // 2
    x0 = (ROI_SIZE - focus) // 2
    rng = np.random.default_rng(seed)
    local, _, _ = generate_training_mask(
        (focus, focus, shape[2]), rng, scenario=scenario, severity=severity,
        center=(focus // 2, focus // 2), band_center=band,
    )
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y0 + focus, x0:x0 + focus] = local
    return mask


def normalize_with_limits(values, lo, hi):
    return np.clip((np.asarray(values, np.float32) - lo) / max(hi - lo, 1e-8), 0, 1)


def get_display_limits(cube_band, valid):
    good = valid & np.isfinite(cube_band)
    if not np.any(good):
        return 0.0, 1.0
    lo, hi = np.percentile(cube_band[good], [1, 99.5])
    return float(lo), float(max(hi, lo + 1e-6))


def save_selection(out, preview, y, x, sy, sx, test_offset, gt, synthetic, model_mask, band):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.imshow(preview, cmap="gray", vmin=0, vmax=1, origin="upper")
    ax.add_patch(Rectangle((x / sx - .5, y / sy - .5), ROI_SIZE / sx, ROI_SIZE / sy,
                           fill=False, edgecolor="yellow", lw=2))
    ax.set_title(f"Selected ROI | original cube y={test_offset + y}:{test_offset + y + ROI_SIZE}")
    fig.tight_layout()
    fig.savefig(out / "01_selected_roi.png", dpi=220)
    plt.close(fig)

    valid = np.isfinite(gt[:, :, band]) & ~model_mask[:, :, band]
    lo, hi = get_display_limits(gt[:, :, band], valid)
    truth = normalize_with_limits(gt[:, :, band], lo, hi)
    overlay = np.stack([truth, truth, truth], axis=-1)
    affected = synthetic[:, :, band]
    overlay[affected] = 0.4 * overlay[affected] + 0.6 * np.array([1, 0.1, 0.1])
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(overlay, vmin=0, vmax=1)
    ax.set_title("Masked location (red) | fixed ROI")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "02_mask_overlay.png", dpi=240)
    plt.close(fig)


def split_indices(coords, world_size, rank):
    pixels, inverse = np.unique(coords[:, :2], axis=0, return_inverse=True)
    return np.flatnonzero((inverse % world_size) == rank)


def run_worker(args):
    # All model definitions and weights are imported from the existing project.
    import torch
    from src.pipeline import load_pipeline, predict_mask_learned_router3
    from tqdm import tqdm  # project dependency; helps catch an incomplete installation early

    world = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local = int(os.getenv("LOCAL_RANK", "0"))
    if not torch.cuda.is_available() or local >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA GPU {local} is not accessible")
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")
    out = Path(args.outdir).resolve()
    work = out / "_work"
    cube = np.load(work / "roi_cube.npy", mmap_mode="r")
    mask = np.load(work / "model_mask.npy", mmap_mode="r")
    coords = np.load(work / "coords.npy")
    take = split_indices(coords, world, rank)
    if not len(take):
        np.savez_compressed(work / f"rank_{rank}.npz", indices=take, values=np.empty(0, np.float32), routes=np.empty(0, np.int8))
        return
    print(f"GPU {local}: loading trained TPSSR | {len(take)} voxels", flush=True)
    pipeline = load_pipeline(
        device, region_batch=args.region_batch, verify_region_batch=True,
        learned_fusion=True, learned_router3_aligned=True,
    )
    with torch.inference_mode():
        result = predict_mask_learned_router3(
            pipeline, cube, mask, coords[take],
            region_progress=f"GPU {local} Region reconstruction",
        )
    values = np.asarray(result["prediction"], dtype=np.float32)
    route_map = {"router3_interp": 0, "router3_fusion": 1, "router3_region": 2}
    routes = np.asarray([route_map[str(s)] for s in result["candidate_method"]], np.int8)
    if not np.isfinite(values).all() or len(values) != len(take):
        raise RuntimeError(f"GPU {local}: incomplete/nonfinite predictions")
    np.savez_compressed(work / f"rank_{rank}.npz", indices=take, values=values, routes=routes)
    print(f"GPU {local}: complete ({len(take)} reconstructed voxels)", flush=True)


def merge_outputs(out, gpus):
    work = out / "_work"
    coords = np.load(work / "coords.npy")
    values = np.full(len(coords), np.nan, dtype=np.float32)
    routes = np.full(len(coords), -1, dtype=np.int8)
    used = np.zeros(len(coords), dtype=bool)
    for rank in range(gpus):
        path = work / f"rank_{rank}.npz"
        if not path.is_file():
            raise RuntimeError(f"Missing inference output from GPU {rank}: {path}")
        with np.load(path) as result:
            idx = result["indices"]
            vv = result["values"]
            rr = result["routes"]
        if np.any(idx < 0) or np.any(idx >= len(coords)) or np.any(used[idx]):
            raise RuntimeError("Duplicated or out-of-range GPU output indices")
        if len(idx) != len(vv) or len(idx) != len(rr):
            raise RuntimeError("Mismatched output lengths")
        values[idx], routes[idx], used[idx] = vv, rr, True
    if not np.all(used) or not np.isfinite(values).all() or np.any(routes < 0):
        raise RuntimeError(f"Incomplete TPSSR reconstruction: {used.sum()}/{len(coords)}")
    return coords, values, routes


def save_reconstruction(out, band, wavelengths, coords, values, routes, params):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    work = out / "_work"
    original = np.load(work / "roi_cube.npy")
    mask = np.load(work / "model_mask.npy")
    synthetic = np.load(work / "synthetic_mask.npy")
    invalid = np.load(work / "roi_invalid.npy")
    valid_target = synthetic & ~invalid & np.isfinite(original)
    if int(valid_target.sum()) != len(coords):
        raise RuntimeError("Inconsistent target voxel count")
    corrupted = original.copy()
    corrupted[mask] = np.nan
    reconstructed = corrupted.copy()
    reconstructed[coords[:, 0], coords[:, 1], coords[:, 2]] = values
    if not np.all(np.isfinite(reconstructed[valid_target])):
        raise RuntimeError("Unfilled synthetic target voxels")
    original_target = original[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float64)
    e = values.astype(np.float64) - original_target
    stats = {
        **params,
        "router": ROUTER,
        "target_voxels": int(len(coords)),
        "masked_spatial_pixels": int(np.count_nonzero(np.any(valid_target, axis=2))),
        "coverage": 1.0,
        "RMSE": float(np.sqrt(np.mean(e * e))),
        "MAE": float(np.mean(np.abs(e))),
        "expert_voxels": {k: int(np.count_nonzero(routes == i)) for i, k in enumerate(("spectral", "fusion", "region"))},
    }
    np.save(out / "roi_ground_truth.npy", original)
    np.save(out / "roi_corrupted.npy", corrupted)
    np.save(out / "roi_reconstructed_tpssr.npy", reconstructed)
    np.save(out / "roi_synthetic_mask.npy", synthetic)
    np.save(out / "roi_expert_routes.npy", np.column_stack([coords, routes]))
    (out / "metrics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # All display panels use exactly the same reflectance limits.
    good = np.isfinite(original[:, :, band]) & ~invalid[:, :, band]
    lo, hi = get_display_limits(original[:, :, band], good)
    gt_img = normalize_with_limits(original[:, :, band], lo, hi)
    corrupted_img = normalize_with_limits(corrupted[:, :, band], lo, hi)
    corrupted_img[~np.isfinite(corrupted[:, :, band])] = 0.0
    recon_img = normalize_with_limits(reconstructed[:, :, band], lo, hi)
    recon_img[~np.isfinite(reconstructed[:, :, band])] = 0.0
    overlay = np.stack([gt_img, gt_img, gt_img], axis=-1)
    m = synthetic[:, :, band]
    overlay[m] = 0.35 * overlay[m] + 0.65 * np.array([1, 0.08, 0.08])
    error = np.full(original.shape[:2], np.nan, dtype=np.float32)
    diff = np.abs(reconstructed[:, :, band] - original[:, :, band])
    error[m & valid_target[:, :, band]] = diff[m & valid_target[:, :, band]]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    panels = [
        ("Ground truth", gt_img, "gray"), ("Mask overlay", overlay, None),
        ("Corrupted", corrupted_img, "gray"), ("TPSSR reconstructed", recon_img, "gray"),
        ("Absolute error (masked)", error, "magma"),
    ]
    for ax, (name, data, cm) in zip(axes, panels):
        im = ax.imshow(data, cmap=cm, vmin=0 if cm == "gray" else None,
                       vmax=1 if cm == "gray" else None, interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.axis("off")
        if name.startswith("Absolute"):
            fig.colorbar(im, ax=ax, fraction=.046, pad=.02)
    fig.suptitle(f"{params['dataset']} | {params['scenario']} S{params['severity']} | "
                 f"band {band} | RMSE={stats['RMSE']:.6f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "03_tpssr_comparison.png", dpi=300)
    plt.close(fig)

    route2d = np.full(original.shape[:2], np.nan, dtype=np.float32)
    for expert in range(3):
        selected = coords[routes == expert, :2]
        route2d[selected[:, 0], selected[:, 1]] = float(expert)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(gt_img, cmap="gray", vmin=0, vmax=1)
    ra = np.ma.masked_invalid(route2d)
    ax.imshow(ra, cmap=ListedColormap(["#4b91c7", "#e5a04c", "#bf5db7"]), vmin=-.5, vmax=2.5, alpha=.82)
    ax.legend(handles=[Patch(facecolor=c, label=t) for c, t in zip(
        ["#4b91c7", "#e5a04c", "#bf5db7"], ["Spectral", "Learned fusion", "Region"]
    )], loc="upper right", fontsize=9)
    ax.set_title("Learned Router decisions (masked pixels)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "04_route_map.png", dpi=280)
    plt.close(fig)

    pixels, counts = np.unique(coords[:, :2], return_counts=True, axis=0)
    py, px = map(int, pixels[int(np.argmax(counts))])
    x = np.asarray(wavelengths, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, original[py, px], label="Ground truth", lw=1.8)
    ax.plot(x, reconstructed[py, px], label="TPSSR", lw=1.5, ls="--")
    ax.plot(x, corrupted[py, px], label="Observed / corrupted", lw=1.0, alpha=.72)
    mb = np.flatnonzero(valid_target[py, px])
    ax.scatter(x[mb], reconstructed[py, px, mb], s=10, c="#c34244", zorder=3, label="Reconstructed bands")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title(f"Spectrum at affected pixel (y={py}, x={px})")
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "05_pixel_spectrum.png", dpi=300)
    plt.close(fig)
    return stats


def main():
    args = arguments()
    from experiment_config import PRESETS, resolve_experiment
    from _launcher_common import env_for

    args.dataset = args.dataset.upper()
    if args.dataset not in PRESETS[args.preset]["tests"]:
        raise SystemExit(f"{args.dataset} is not a held-out test image of {args.preset}: {PRESETS[args.preset]['tests']}")
    args.active_test = args.dataset
    args.train_datasets = None
    args.test_datasets = None
    args.experiment_name = None
    spec = resolve_experiment(args)
    os.environ.update(env_for(spec, args.dataset))
    if str(CORE) not in sys.path:
        sys.path.insert(0, str(CORE))
    if args.worker:
        run_worker(args)
        return

    import config
    from src.evaluation_utils import load_dataset, split_blocks
    cube, invalid, split, waves = load_dataset(args.dataset)
    _, _, test_cube = split_blocks(cube, split)
    _, _, test_invalid = split_blocks(invalid, split)
    height, width, bands = test_cube.shape
    test_start = int(split["val_end"])
    band = bands // 2 if args.band is None else args.band
    if not (0 <= band < bands):
        raise SystemExit(f"--band must be between 0 and {bands-1}")
    print(f"{args.dataset} held-out test block: {height} x {width} x {bands} | fixed ROI: {ROI_SIZE} x {ROI_SIZE}", flush=True)
    preview, sy, sx = preview_image(test_cube, test_invalid, band, args.preview_max_height, args.preview_max_width)
    if (args.roi_y is None) != (args.roi_x is None):
        raise SystemExit("Pass both --roi-y and --roi-x, or neither for interactive selection")
    if args.roi_y is None:
        y, x = select_box(preview, height, width, sy, sx, args.dataset, band)
    else:
        y, x = boundaries(args.roi_y, args.roi_x, height, width)
    out = (Path(args.outdir) if args.outdir else ROOT / "roi_visuals" / args.dataset /
           f"{args.scenario}_S{args.severity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output directory is not empty: {out}. Choose another --outdir.")
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)

    roi = np.array(test_cube[y:y + ROI_SIZE, x:x + ROI_SIZE], dtype=np.float32, copy=True)
    invalid_roi = np.asarray(test_invalid[y:y + ROI_SIZE, x:x + ROI_SIZE], dtype=bool).copy()
    synthetic = make_local_mask(roi.shape, args.scenario, args.severity, band, args.seed)
    target = synthetic & ~invalid_roi & np.isfinite(roi)
    model_mask = synthetic | invalid_roi | ~np.isfinite(roi)
    coords = np.argwhere(target).astype(np.int32)
    if len(coords) < 1:
        raise SystemExit("No valid masked pixels in this ROI. Move the square onto valid sediment.")
    if len(coords) > 250000 or np.count_nonzero(np.any(target, axis=2)) > 1200:
        raise SystemExit("The requested mask is too large for this visual test. Try another scenario or severity.")
    np.save(work / "roi_cube.npy", roi)
    np.save(work / "roi_invalid.npy", invalid_roi)
    np.save(work / "synthetic_mask.npy", synthetic)
    np.save(work / "model_mask.npy", model_mask)
    np.save(work / "coords.npy", coords)
    save_selection(out, preview, y, x, sy, sx, test_start, roi, synthetic, model_mask, band)
    params = {
        "dataset": args.dataset, "preset": args.preset,
        "scenario": args.scenario, "severity": args.severity,
        "visual_mask_source": "src.training_corruptions.generate_training_mask",
        "visual_mask_note": "Local synthetic training-style mask; not one of the 200 benchmark masks",
        "roi_size": ROI_SIZE, "defect_window_size": DEFECT_SIZE,
        "roi_y_in_test_block": y, "roi_x_in_test_block": x,
        "roi_y_in_full_image": test_start + y, "roi_x_in_full_image": x,
        "display_band": band, "display_wavelength_nm": float(waves[band]),
        "seed": args.seed, "requested_gpus": args.gpus,
    }
    (out / "selection.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"Selected ROI x={x}:{x + ROI_SIZE}, test y={y}:{y + ROI_SIZE} | mask targets: {len(coords):,} voxels", flush=True)
    if args.select_only:
        print(f"Selection only: {out}", flush=True)
        return

    import torch
    available = torch.cuda.device_count()
    if available < args.gpus:
        raise SystemExit(f"Requested {args.gpus} GPU(s), but PyTorch detects {available}. No inference was launched.")
    cmd = [sys.executable]
    if args.gpus == 2:
        cmd += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
    cmd += [str(Path(__file__).resolve()), "--preset", args.preset, "--dataset", args.dataset,
            "--scenario", args.scenario, "--severity", str(args.severity),
            "--outdir", str(out), "--region-batch", str(args.region_batch), "--worker"]
    print(f"Running trained {ROUTER} on {args.gpus} GPU(s)...", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env_for(spec, args.dataset), check=True)
    coords, values, routes = merge_outputs(out, args.gpus)
    stats = save_reconstruction(out, band, waves, coords, values, routes, params)
    shutil.rmtree(work)
    print(f"Complete | RMSE={stats['RMSE']:.6f} | MAE={stats['MAE']:.6f} | coverage=100%", flush=True)
    print(f"Comparison figure: {out / '03_tpssr_comparison.png'}", flush=True)
    print(f"Saved all outputs to: {out}", flush=True)


if __name__ == "__main__":
    main()
