#!/usr/bin/env python3
"""Interactive ROI reconstruction with the final split-specific TPSSR checkpoints.

Supports SWIR 8x2, VNIR 8x2, SWIR A-to-B, and SWIR B-to-A. The final
RankCV + Hybrid Gap 10% routing policy is active by default. Select a 128x128
ROI and drag a reproducible synthetic-defect window inside it.
These masks are local illustrative examples, not the official evaluation masks.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CORE = ROOT / "core"
ROI_SIZE = 128
DEFECT_SIZE = 31
ROUTER = "learned_router3_lineaware_rankcv"
DEFAULT_DATASETS = {
    "swir_8x2": "SWIR5",
    "vnir_8x2": "VNIR5",
    "swir_region_a_to_b": "SWIR6",
    "swir_region_b_to_a": "SWIR1",
}
SCENARIOS = (
    "spatial_patch", "spatio_spectral_block", "whole_spectrum_pixels",
    "mixed_saturation", "band_wide_saturation", "consecutive_spectral_bands",
    "isolated_voxels", "single_detector_line", "multiple_detector_lines",
    "partial_vertical_lines",
)


def arguments():
    p = argparse.ArgumentParser(description="Fixed 128x128 draggable ROI + real TPSSR reconstruction")
    p.add_argument("--preset", choices=("swir_8x2", "swir_region_a_to_b", "swir_region_b_to_a", "vnir_8x2"),
                   default="swir_8x2", help="Trained model experiment. Does not restrict the displayed image to test images.")
    p.add_argument("--dataset", default=None,
                   help="Image to display. If omitted, uses the first held-out test image "
                        "for the selected preset (SWIR5, VNIR5, SWIR6, or SWIR1).")
    flip = p.add_mutually_exclusive_group()
    flip.add_argument("--flip-x", dest="flip_x", action="store_true",
                      help="Mirror previews and figures horizontally; display only")
    flip.add_argument("--no-flip-x", dest="flip_x", action="store_false")
    p.set_defaults(flip_x=None)
    p.add_argument("--image-scope", choices=("auto", "full", "test-block"), default="auto",
                   help="auto: full image for training acquisitions, last held-out block for test acquisitions; "
                        "full: entire image; test-block: last held-out block (test acquisitions only)")
    p.add_argument("--scenario", choices=SCENARIOS, default="spatio_spectral_block")
    p.add_argument("--band-wide-mode", choices=("local", "full-roi"), default="local",
                   help="For band_wide_saturation: local 31x31 example (legacy), or full-roi "
                        "band deletion across all 128x128 spatial pixels (benchmark-style tile)")
    p.add_argument("--band-wide-eval", choices=("benchmark", "all"), default=None,
                   help="For full-roi band-wide: reconstruct ALL valid masked voxels by default; "
                        "benchmark optionally samples up to 5000 spatial pixels")
    p.add_argument("--severity", type=int, choices=range(1, 6), default=5)
    p.add_argument("--whole-spectrum-size", type=int, default=None,
                   help="Spatial side length for whole_spectrum_pixels. Default matches the benchmark: "
                        "S1..S5 = 1,3,5,9,15. Use 19 to reproduce the previous visualizer S5.")
    p.add_argument("--router", choices=("learned_router3_lineaware_rankcv", "learned_router3_lineaware",
                                         "learned_router3_aligned"), default=ROUTER,
                   help="Select a trained Router; defaults to the split-specific final RankCV")
    p.add_argument("--gpus", type=int, choices=(1, 2), default=2)
    p.add_argument("--spatial-consistency", action="store_true",
                   help="Legacy CV policy; incompatible with Global Minority")
    p.add_argument("--global-minority-percent", type=float, default=10.0,
                   help="Final two-expert minority threshold across the whole ROI (default: 10%%)")
    p.add_argument("--minority-policy", choices=("threshold", "hybrid", "hybrid_gap"), default="hybrid_gap",
                   help="Final policy: hybrid_gap (default); threshold/hybrid retained for audit")
    p.add_argument("--three-expert-min-gap-percent", type=float, default=10.0,
                   help="Minimum percentage-point gap between the two smallest expert shares (default: 10)")
    p.add_argument("--fusion-guard", action="store_true",
                   help="Apply source-calibrated Fusion Guard to eligible reconstructed pixels")
    p.add_argument("--band", type=int, default=None, help="Zero-based band index; default: middle band")
    p.add_argument("--seed", type=int, default=4207)
    p.add_argument("--outdir", default=None)
    p.add_argument("--roi-y", type=int, default=None, help="Fixed ROI top coordinate inside the selected image area")
    p.add_argument("--roi-x", type=int, default=None, help="Fixed ROI left coordinate inside the selected image area")
    p.add_argument("--missing-y", type=int, default=None,
                   help="Optional missing window top row within the 128x128 ROI (0..97); skips the second selection window")
    p.add_argument("--missing-x", type=int, default=None,
                   help="Optional missing window left column within the 128x128 ROI (0..97); skips the second selection window")
    p.add_argument("--region-batch", type=int, default=16)
    p.add_argument("--preview-max-height", type=int, default=1300)
    p.add_argument("--preview-max-width", type=int, default=1000)
    p.add_argument("--select-only", action="store_true", help="Save a selection/mask preview; do not run the model")
    p.add_argument("--maps-from-output", type=Path, default=None,
                   help="Recover the eight trained Region maps from a completed ROI output folder. "
                        "Does not repeat ROI selection or full TPSSR reconstruction.")
    p.add_argument("--region-figure", dest="region_figure", action="store_true", default=True,
                   help="Export real Geometry/Topology maps, including eight architecture maps (default)")
    p.add_argument("--no-region-figure", dest="region_figure", action="store_false",
                   help="Skip trained Region feature-map export")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def boundaries(y, x, height, width, size=ROI_SIZE):
    if height < size or width < size:
        raise ValueError(f"Selected image is {height}x{width}; the fixed ROI requires at least {size}x{size}.")
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


def select_box(image, height, width, step_y, step_x, dataset, band, area_label, flip_x=False):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if plt.get_backend().lower() in {"agg", "template", "pdf", "svg", "ps", "cairo"} or "inline" in plt.get_backend().lower():
        raise RuntimeError("Matplotlib has a non-interactive backend. Run in Linux desktop/RDP, or pass --roi-y and --roi-x.")

    ry, rx = boundaries((height - ROI_SIZE) // 2, (width - ROI_SIZE) // 2, height, width)
    state = {"y": ry, "x": rx, "drag": None, "confirmed": False}
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest", origin="upper")
    if flip_x:
        ax.invert_xaxis()
    r = Rectangle((rx / step_x - 0.5, ry / step_y - 0.5), ROI_SIZE / step_x,
                  ROI_SIZE / step_y, fill=False, lw=2.0, edgecolor="yellow")
    ax.add_patch(r)
    ax.set_title(f"{dataset} | {area_label} | band {band}\nDrag yellow square or click to reposition; Enter = confirm, Esc = cancel", fontsize=11)
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


def missing_window_size(scenario, severity, whole_spectrum_size=None):
    """Interactive defect footprint. Benchmark sizes for single compact components."""
    level = int(severity) - 1
    if scenario == "whole_spectrum_pixels":
        size = (1, 3, 5, 9, 15)[level] if whole_spectrum_size is None else int(whole_spectrum_size)
        if not 1 <= size <= ROI_SIZE:
            raise ValueError(f"Whole-spectrum size must be in 1..{ROI_SIZE}")
        return size
    if scenario in {"spatial_patch", "spatio_spectral_block"}:
        return (5, 9, 15, 25, 41)[level]
    if scenario == "mixed_saturation":
        # A local selection FOOTPRINT, not a claim of a square mixed mask.
        # S5 contains several kinds of spatial and spectral defects.
        return (15, 21, 25, 31, 41)[level]
    # Narrow detector and spectral defects use a local placement region.
    # For band-wide saturation this is a local illustrative sample, not a
    # simulation of full-image band-wide loss.
    return DEFECT_SIZE


def make_local_mask(shape, scenario, severity, band, seed, missing_y=None, missing_x=None,
                    whole_spectrum_size=None, band_wide_mode="local"):
    if band_wide_mode not in ("local", "full-roi"):
        raise ValueError(f"Invalid band-wide mode: {band_wide_mode}")
    if shape[0] != ROI_SIZE or shape[1] != ROI_SIZE:
        raise ValueError("Unexpected ROI dimensions")
    if not 0 <= band < shape[2]:
        raise ValueError("Band index outside cube")
    if scenario not in SCENARIOS:
        raise ValueError("Invalid scenario")
    if not 1 <= int(severity) <= 5:
        raise ValueError("Invalid severity")

    side = missing_window_size(scenario, severity, whole_spectrum_size)
    start = (ROI_SIZE - side) // 2
    y0 = start if missing_y is None else int(missing_y)
    x0 = start if missing_x is None else int(missing_x)
    if not (0 <= y0 <= ROI_SIZE - side and 0 <= x0 <= ROI_SIZE - side):
        raise ValueError(f"Missing window ({side}x{side}) must remain inside the {ROI_SIZE}x{ROI_SIZE} ROI")

    mask = np.zeros(shape, dtype=bool)
    if scenario == "band_wide_saturation" and band_wide_mode == "full-roi":
        # Match the benchmark's corruption geometry within ONE 128x128 tile:
        # all spatial positions lose the same contiguous wavelengths.
        width = min((1, 3, 7, 15, 31)[int(severity) - 1], shape[2])
        b0 = int(np.clip(band - width // 2, 0, shape[2] - width))
        mask[:, :, b0:b0 + width] = True
    elif scenario == "whole_spectrum_pixels":
        mask[y0:y0 + side, x0:x0 + side, :] = True
    elif scenario in {"spatial_patch", "spatio_spectral_block"}:
        # One spatial-spectral component at the exact benchmark S-level dimensions.
        widths = ((1, 3, 7, 15, 31) if scenario == "spatial_patch"
                  else (5, 11, 21, 41, 63))
        width = min(widths[int(severity) - 1], shape[2])
        b0 = max(0, min(band - width // 2, shape[2] - width))
        mask[y0:y0 + side, x0:x0 + side, b0:b0 + width] = True
    elif scenario == "mixed_saturation" and int(severity) == 5:
        # A compact illustrative mixture of benchmark-S5 COMPONENTS on one
        # user-selected location. The actual benchmark places components at
        # independent locations throughout the complete test image.
        b63 = min(63, shape[2])
        z63 = max(0, min(band - b63 // 2, shape[2] - b63))
        mask[y0:y0 + side, x0:x0 + side, z63:z63 + b63] = True
        # Whole-spectrum S5 component: 15x15, centered within the selected area.
        c = side // 2
        half = 15 // 2
        mask[y0+c-half:y0+c-half+15, x0+c-half:x0+c-half+15, :] = True
        # Several additional detector failures, corresponding to the mixed
        # mode's line components. Keep them reproducible for a chosen seed.
        rng = np.random.default_rng(seed)
        for _ in range(10):
            col = int(rng.integers(x0, x0 + side))
            spectral_band = int(rng.integers(shape[2]))
            mask[y0:y0 + side, col, spectral_band] = True
        # Full-band S3 component is also represented within the local footprint
        # (7 bands); these channels are already contained in the 63-band mask.
    else:
        # For other compound/line defects, use the existing local training-style
        # generator. The placement square need not equal the deletion area.
        from src.training_corruptions import generate_training_mask
        rng = np.random.default_rng(seed)
        local, _, _ = generate_training_mask(
            (side, side, shape[2]), rng, scenario=scenario, severity=severity,
            center=(side // 2, side // 2), band_center=band,
        )
        mask[y0:y0 + side, x0:x0 + side] = local
    return mask


def count_invalid_whole_spectrum_pixels(roi, invalid_roi, y, x, size):
    """Count pixels without a valid ground-truth spectrum in the proposed square.

    A whole-spectrum corruption must not silently leave holes where an original
    spectrum is partially invalid: these pixels cannot be scored as complete
    spectra, and keeping them would produce visible stripes in the mask.
    """
    data = np.asarray(roi[y:y + size, x:x + size])
    unavailable = np.asarray(invalid_roi[y:y + size, x:x + size], dtype=bool)
    if data.ndim != 3 or data.shape[:2] != (size, size) or unavailable.shape != data.shape:
        raise ValueError("Whole-spectrum window exceeds the ROI or has invalid array shapes")
    complete = np.all(np.isfinite(data) & ~unavailable, axis=2)
    return int(size * size - np.count_nonzero(complete))


def select_missing_window(roi, invalid_roi, scenario, severity, band, seed, whole_spectrum_size=None, flip_x=False):
    """Move the exact fixed-size synthetic corruption footprint inside the ROI.

    For whole-spectrum and compact patch scenarios, the yellow square marks
    the exact spatial footprint. For other scenarios it is the local placement
    region: the mask may contain lines, spectral gaps, or a mixture.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    backend = plt.get_backend().lower()
    if backend in {"agg", "template", "pdf", "svg", "ps", "cairo"} or "inline" in backend:
        raise RuntimeError("Interactive Missing selection requires a desktop Matplotlib backend. "
                           "Use --missing-y and --missing-x for non-interactive execution.")

    size = missing_window_size(scenario, severity, whole_spectrum_size)
    start = (ROI_SIZE - size) // 2
    state = {"y": start, "x": start, "grab": None, "confirmed": False,
             "invalid_whole_spectra": 0}
    # Fixed grayscale limits based on unmasked ROI, not each moving window.
    good = np.isfinite(roi[:, :, band]) & ~invalid_roi[:, :, band]
    lo, hi = get_display_limits(roi[:, :, band], good)
    image = normalize_with_limits(roi[:, :, band], lo, hi)
    image[~good] = 0.0

    # This local mask is the real corruption for the chosen scenario and seed.
    template = make_local_mask(roi.shape, scenario, severity, band, seed,
                               missing_y=0, missing_x=0, whole_spectrum_size=whole_spectrum_size)[:size, :size, :]
    covered = np.any(template, axis=2)
    on_band = template[:, :, band]
    alpha = np.zeros((ROI_SIZE, ROI_SIZE), dtype=np.float32)
    def paint(y0, x0):
        alpha.fill(0)
        # Orange indicates voxels masked at some wavelength; red is the
        # selected display band. This also makes single-band defects visible.
        region = alpha[y0:y0+size, x0:x0+size]
        region[covered] = 0.25
        region[on_band] = 0.57
        return alpha

    fig, ax = plt.subplots(figsize=(7.5, 8))
    fig.subplots_adjust(left=0.09, right=0.94, top=0.86, bottom=0.12)
    ax.imshow(image, cmap="gray", vmin=0, vmax=1,
              interpolation="nearest", origin="upper")
    rgba = np.zeros((ROI_SIZE, ROI_SIZE, 4), dtype=np.float32)
    rgba[..., :3] = (1.0, 0.12, 0.08)
    overlay = ax.imshow(rgba, origin="upper", interpolation="nearest", zorder=2)
    outline = Rectangle((start - 0.5, start - 0.5), size, size,
                        edgecolor="#ffe44d", lw=2.1, fill=False, zorder=3)
    ax.add_patch(outline)
    ax.set_xlim((ROI_SIZE - 0.5, -0.5) if flip_x else (-0.5, ROI_SIZE - 0.5))
    ax.set_ylim(ROI_SIZE - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Choose the Missing location within the ROI\n"
                 "Drag the yellow square, or click elsewhere to move it", fontsize=11)
    if scenario == "whole_spectrum_pixels":
        covered_count = int(np.count_nonzero(np.any(template, axis=2)))
        print(f"Whole-spectrum mask: {size}x{size} pixels deleted at ALL wavelengths. "
              "The yellow rectangle shows the exact mask position.", flush=True)
    fig.text(0.5, 0.06, "Left mouse: move / drag    Arrows: 1 pixel    Shift+arrows: 5 pixels\n"
             "Enter: confirm    Esc: cancel    Red: masked at shown band; pale red: other bands",
             ha="center", va="center", fontsize=9)
    info = fig.text(0.5, 0.025, "", ha="center", va="center", fontsize=9)

    def update(y0, x0):
        state["y"] = int(np.clip(round(y0), 0, ROI_SIZE - size))
        state["x"] = int(np.clip(round(x0), 0, ROI_SIZE - size))
        y1, x1 = state["y"], state["x"]
        outline.set_xy((x1 - 0.5, y1 - 0.5))
        rgba[..., 3] = paint(y1, x1)
        overlay.set_data(rgba)
        available = np.isfinite(roi[y1:y1+size, x1:x1+size]) & ~invalid_roi[y1:y1+size, x1:x1+size]
        n_target = int(np.count_nonzero(template & available))
        if scenario == "whole_spectrum_pixels":
            bad = count_invalid_whole_spectrum_pixels(roi, invalid_roi, y1, x1, size)
            state["invalid_whole_spectra"] = bad
            outline.set_edgecolor("#ff5252" if bad else "#ffe44d")
            if bad:
                info.set_text(f"NOT VALID: {bad} / {size*size} spectra already have missing bands. "
                              "Move the square to an intact area.")
                info.set_color("#c62828")
            else:
                info.set_text(f"VALID: entire {size}x{size} square has intact spectra "
                              f"({n_target:,} target voxels). Press Enter.")
                info.set_color("#167a32")
        else:
            info.set_text(f"Missing window x={x1}:{x1+size}  y={y1}:{y1+size}  |  valid target voxels: {n_target:,}")
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        cx, cy = int(round(event.xdata)), int(round(event.ydata))
        if (state["x"] <= cx < state["x"] + size and
                state["y"] <= cy < state["y"] + size):
            state["grab"] = (cx - state["x"], cy - state["y"])
        else:
            state["grab"] = (size // 2, size // 2)
            update(cy - size // 2, cx - size // 2)

    def on_motion(event):
        if state["grab"] is None or event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        dx, dy = state["grab"]
        update(event.ydata - dy, event.xdata - dx)

    def on_release(_event):
        state["grab"] = None

    def on_key(event):
        key = (event.key or "").lower()
        if key in ("enter", "return"):
            if scenario == "whole_spectrum_pixels" and state["invalid_whole_spectra"]:
                print(f"[WHOLE-SPECTRUM SELECTION] Rejected: "
                      f"{state['invalid_whole_spectra']} pixels have an already invalid band. "
                      "Move the square to fully valid spectra; no mask was created.", flush=True)
                return
            state["confirmed"] = True
            plt.close(fig)
        elif key in ("escape", "q"):
            plt.close(fig)
        else:
            step = 5 if "shift+" in key else 1
            direction = key.replace("shift+", "")
            movement = {"left": (0, -step), "right": (0, step),
                        "up": (-step, 0), "down": (step, 0)}.get(direction)
            if movement:
                update(state["y"] + movement[0], state["x"] + movement[1])

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    update(start, start)
    plt.show()
    if not state["confirmed"]:
        raise SystemExit("Missing-location selection cancelled. No inference was launched.")
    return state["y"], state["x"]


def normalize_with_limits(values, lo, hi):
    return np.clip((np.asarray(values, np.float32) - lo) / max(hi - lo, 1e-8), 0, 1)


def get_display_limits(cube_band, valid):
    good = valid & np.isfinite(cube_band)
    if not np.any(good):
        return 0.0, 1.0
    lo, hi = np.percentile(cube_band[good], [1, 99.5])
    return float(lo), float(max(hi, lo + 1e-6))


def save_selection(out, preview, y, x, sy, sx, image_offset, gt, synthetic, model_mask, band, flip_x=False):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.imshow(preview, cmap="gray", vmin=0, vmax=1, origin="upper")
    if flip_x:
        ax.invert_xaxis()
    ax.add_patch(Rectangle((x / sx - .5, y / sy - .5), ROI_SIZE / sx, ROI_SIZE / sy,
                           fill=False, edgecolor="yellow", lw=2))
    ax.set_title(f"Selected ROI | original cube y={image_offset + y}:{image_offset + y + ROI_SIZE}")
    fig.tight_layout()
    fig.savefig(out / "01_selected_roi.png", dpi=220)
    plt.close(fig)

    valid = np.isfinite(gt[:, :, band]) & ~model_mask[:, :, band]
    if not np.any(valid):
        # Full-roi Band-wide removes the entire displayed band. Ground truth
        # is allowed for figure scaling only; it never reaches inference.
        valid = np.isfinite(gt[:, :, band])
    lo, hi = get_display_limits(gt[:, :, band], valid)
    truth = normalize_with_limits(gt[:, :, band], lo, hi)
    overlay = np.stack([truth, truth, truth], axis=-1)
    affected = synthetic[:, :, band]
    overlay[affected] = 0.4 * overlay[affected] + 0.6 * np.array([1, 0.1, 0.1])
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(overlay, vmin=0, vmax=1)
    if flip_x:
        ax.invert_xaxis()
    ax.set_title("Masked location (red) | fixed ROI")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "02_mask_overlay.png", dpi=240)
    plt.close(fig)


def split_indices(coords, world_size, rank, spatial_consistency=False):
    pixels, inverse = np.unique(coords[:, :2], axis=0, return_inverse=True)
    if not spatial_consistency:
        return np.flatnonzero((inverse % world_size) == rank)

    # A spatial consistency decision needs *all* four-connected target pixels.
    # Split only at disconnected components, never through one damaged patch.
    lookup = {tuple(row): i for i, row in enumerate(pixels)}
    visited = np.zeros(len(pixels), dtype=bool)
    counts = np.bincount(inverse, minlength=len(pixels))
    groups = []
    for i in range(len(pixels)):
        if visited[i]:
            continue
        visited[i] = True
        stack = [i]
        component = []
        while stack:
            j = stack.pop()
            component.append(j)
            y, x = map(int, pixels[j])
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                q = lookup.get(neighbor)
                if q is not None and not visited[q]:
                    visited[q] = True
                    stack.append(q)
        groups.append(np.asarray(component, dtype=np.int64))
    assigned = np.full(len(pixels), -1, dtype=np.int8)
    totals = np.zeros(world_size, dtype=np.int64)
    for component in sorted(groups, key=lambda g: -int(counts[g].sum())):
        gpu = int(np.argmin(totals))
        assigned[component] = gpu
        totals[gpu] += counts[component].sum()
    assert np.all(assigned >= 0)
    if rank == 0:
        print(f"[SPATIAL ROUTER] {len(groups)} connected ROI component(s) allocated "
              f"across {world_size} GPU(s); target voxels by GPU: {totals.tolist()}", flush=True)
    return np.flatnonzero(assigned[inverse] == rank)


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
    take = split_indices(coords, world, rank, args.spatial_consistency)
    # Empty ranks must participate in the global-count reduction. This occurs,
    # for example, when a single-pixel S1 mask is split across two GPUs.
    distributed = world > 1 and args.global_minority_percent is not None
    if distributed:
        import torch.distributed as dist
        dist.init_process_group(backend="gloo")
    if not len(take) and not distributed:
        np.savez_compressed(work / f"rank_{rank}.npz", indices=take,
                            values=np.empty(0, np.float32), routes=np.empty(0, np.int8))
        return
    if len(take):
        print(f"GPU {local}: loading trained TPSSR | {len(take)} voxels", flush=True)
    router_flags = {
        "learned_router3_aligned": "learned_router3_aligned",
        "learned_router3_lineaware": "learned_router3_lineaware",
        "learned_router3_lineaware_rankcv": "learned_router3_lineaware_rankcv",
    }
    pipeline = (load_pipeline(
        device, region_batch=args.region_batch, verify_region_batch=True,
        learned_fusion=True, **{router_flags[args.router]: True},
    ) if len(take) else None)
    spatial_ck = None
    if args.spatial_consistency:
        if args.router != "learned_router3_lineaware_rankcv":
            raise ValueError("--spatial-consistency requires the RankCV Router")
        import joblib
        from training.spectral_gate import sha
        ckdir = Path(__import__("config").MODELS_DIR) / "tpssr" / "learned_router3_lineaware_rankcv"
        ckfile = ckdir / "spatial_consistency.joblib"
        if not ckfile.is_file():
            raise FileNotFoundError(f"Spatial Router checkpoint not found: {ckfile}")
        spatial_ck = joblib.load(ckfile)
        if (spatial_ck.get("architecture") != "spatial_consistency_source_cv_v1"
                or spatial_ck.get("router_sha256") != sha(ckdir / "best_model.joblib")):
            raise RuntimeError("Spatial Router calibration is incompatible with current RankCV checkpoint")
        print(f"GPU {local}: spatial consistency enabled, calibrated strength "
              f"{spatial_ck['strength']}", flush=True)
    if args.fusion_guard and len(take):
        from src.pipeline import load_fusion_guard_calibration
        fusion_guard_ck = load_fusion_guard_calibration(pipeline)
    else:
        fusion_guard_ck = None
    global_counts = None
    if args.global_minority_percent is not None:
        local_counts = np.zeros(3, dtype=np.int64)
        if len(take):
            with torch.inference_mode():
                routing = predict_mask_learned_router3(
                    pipeline, cube, mask, coords[take], routing_only=True)
            local_counts = np.asarray(routing["raw_route_counts"], np.int64)
        if distributed:
            from torch.distributed import all_reduce, ReduceOp
            counts_tensor = torch.from_numpy(local_counts.copy())
            all_reduce(counts_tensor, op=ReduceOp.SUM)
            global_counts = counts_tensor.numpy().copy()
        else:
            global_counts = local_counts
        if int(global_counts.sum()) != len(np.unique(coords[:, :2], axis=0)):
            raise RuntimeError("Global Minority did not count every evaluated ROI pixel exactly once")
        if rank == 0:
            print(f"[GLOBAL MINORITY] raw Expert counts={global_counts.tolist()} | "
                  f"threshold={args.global_minority_percent:g}%", flush=True)
    if not len(take):
        np.savez_compressed(work / f"rank_{rank}.npz", indices=take,
                            values=np.empty(0, np.float32), routes=np.empty(0, np.int8))
        if distributed:
            from torch.distributed import destroy_process_group
            destroy_process_group()
        return
    with torch.inference_mode():
        result = predict_mask_learned_router3(
            pipeline, cube, mask, coords[take],
            region_progress=f"GPU {local} Region reconstruction",
            spatial_consistency=spatial_ck,
            fusion_guard=fusion_guard_ck,
            global_minority_threshold=(args.global_minority_percent / 100.0
                                       if args.global_minority_percent is not None else None),
            global_minority_counts=global_counts,
            global_minority_policy=args.minority_policy,
            three_expert_min_gap=args.three_expert_min_gap_percent / 100.0,
        )
    if distributed:
        from torch.distributed import destroy_process_group
        destroy_process_group()
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
    if not np.all(valid_target[coords[:, 0], coords[:, 1], coords[:, 2]]):
        raise RuntimeError("Reconstruction coordinates contain invalid targets")
    # Only an explicitly requested benchmark-sampling run leaves some corrupted
    # voxels unfilled. Full-ROI evaluation predicts and scores all valid targets.
    corrupted = original.copy()
    corrupted[mask] = np.nan
    reconstructed = corrupted.copy()
    reconstructed[coords[:, 0], coords[:, 1], coords[:, 2]] = values
    if not np.all(np.isfinite(reconstructed[coords[:, 0], coords[:, 1], coords[:, 2]])):
        raise RuntimeError("Unfilled evaluated target voxels")
    original_target = original[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float64)
    e = values.astype(np.float64) - original_target
    stats = {
        **params,
        "router": params["router"],
        "target_voxels": int(len(coords)),
        "masked_spatial_pixels": int(np.count_nonzero(np.any(valid_target, axis=2))),
        "evaluated_spatial_pixels": int(len(np.unique(coords[:, :2], axis=0))),
        "evaluated_target_voxels": int(len(coords)),
        "masked_valid_target_voxels": int(valid_target.sum()),
        "coverage": 1.0,  # Coverage of scored target coordinates, not the full missing ROI.
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
    shown = f" | sampled {len(coords):,}/{int(valid_target.sum()):,} voxels" if len(coords) < int(valid_target.sum()) else ""
    fig.suptitle(f"{params['dataset']} | {params['scenario']} S{params['severity']} | "
                 f"band {band} | RMSE={stats['RMSE']:.6f}{shown}", fontsize=12)
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




def make_geometry_orientation_views(orientation, edge_probability, observed):
    """Make confidence-aware, correctly decoded views of a doubled-angle field.

    The Geometry head predicts gradient NORMAL orientation as [cos(2a), sin(2a)].
    The line tangent is a + pi/2 modulo pi. These operations are for display only.
    """
    from scipy.ndimage import gaussian_filter
    import matplotlib.pyplot as plt

    orient = np.asarray(orientation, dtype=np.float64)
    edge = np.nan_to_num(np.asarray(edge_probability, dtype=np.float64), nan=0.0)
    background = np.clip(np.nan_to_num(observed, nan=0.0), 0.0, 1.0)
    if orient.ndim != 3 or orient.shape[0] != 2 or orient.shape[1:] != edge.shape:
        raise ValueError('Geometry orientation must have shape (2, H, W)')

    # Two-angle encoding: theta here is the normal to the geological boundary.
    theta_normal = 0.5 * np.arctan2(orient[1], orient[0])
    theta_tangent = np.mod(theta_normal + np.pi / 2.0, np.pi)

    # Only display saturated orientation colors where the edge head is confident.
    edge_alpha = np.clip((edge - 0.25) / 0.45, 0.0, 0.92)
    base = np.repeat(background[..., None], 3, axis=-1)
    direction_rgb = plt.get_cmap('twilight_shifted')(theta_tangent / np.pi)[..., :3]
    overlay = (1.0 - edge_alpha[..., None]) * base + edge_alpha[..., None] * direction_rgb
    overlay = np.clip(overlay, 0.0, 1.0)

    # Vector-averaging must happen in (cos 2a, sin 2a) space, not in angles:
    # this avoids incorrect averaging around the 0/180 degree wrap boundary.
    weighted = np.clip(edge, 0, 1)
    weight_local = gaussian_filter(weighted, sigma=2.0)
    cx = gaussian_filter(weighted * orient[0], sigma=2.0)
    sy = gaussian_filter(weighted * orient[1], sigma=2.0)
    coherence = np.hypot(cx, sy) / np.maximum(weight_local, 1e-6)
    smoothed_tangent = np.mod(0.5 * np.arctan2(sy, cx) + np.pi / 2.0, np.pi)
    confidence = weight_local * np.clip(coherence, 0, 1)
    return overlay, smoothed_tangent, confidence


def save_orientation_line_view(path, observed, angle, confidence, mask=None):
    """Overlay sparse tangent line segments, avoiding noisy pixelwise hue plots."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    H, W = observed.shape
    fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
    ax.imshow(np.clip(observed, 0, 1), cmap='gray', vmin=0, vmax=1,
              origin='upper', interpolation='nearest')
    stride = max(5, int(round(min(H, W) / 10)))
    radius = max(1.6, stride * 0.38)
    threshold = max(0.30, float(np.quantile(confidence, 0.65)))
    used = 0
    for yy in range(stride // 2, H, stride):
        for xx in range(stride // 2, W, stride):
            if confidence[yy, xx] < threshold:
                continue
            a = angle[yy, xx]
            if not np.isfinite(a):
                continue
            dx, dy = radius * np.cos(a), radius * np.sin(a)
            ax.plot([xx - dx, xx + dx], [yy - dy, yy + dy],
                    color='#19a7f0', linewidth=1.7, solid_capstyle='round')
            used += 1
    if mask is not None and np.any(mask):
        ys, xs = np.where(mask)
        ax.add_patch(Rectangle((xs.min() - 0.5, ys.min() - 0.5),
                               xs.max() - xs.min() + 1, ys.max() - ys.min() + 1,
                               fill=False, linestyle='--', edgecolor='#ff633c', linewidth=1.5))
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=300, pad_inches=0)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return rgb, used

EIGHT_MAP_NAMES = (
    "01_edge.png", "02_orientation.png", "03_contrast_hint.png", "04_skeleton.png",
    "05_width.png", "06_tangent.png", "07_polarity.png", "08_contrast_magnitude.png",
)


def check_eight_maps(out):
    folder = Path(out) / "region_architecture_maps" / "eight_native_maps"
    wanted = ("00_eight_maps_overview.png",) + EIGHT_MAP_NAMES
    missing = [name for name in wanted if not (folder / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Incomplete Region diagnostic export in {folder}. Missing: {', '.join(missing)}")
    print(f"[OK] All 8 trained Region maps exported: {folder}", flush=True)
    print(f"[OK] Eight-panel figure: {folder / '00_eight_maps_overview.png'}", flush=True)
    return folder


def export_eight_native_maps(folder, arrays, observed, missing, wavelengths, band):
    """Export eight native learned heads without making up structural labels.

    Geometry orientation is an undirected edge NORMAL in doubled-angle encoding.
    Topology tangent is a distinct two-component LINE direction learned by Stage B.
    The two fields must not be plotted with the same derived angle.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors

    required = ("edge_prob", "orientation", "contrast_hint", "skeleton_prob",
                "width_expected", "tangent", "polarity", "contrast_mag")
    missing_keys = [name for name in required if name not in arrays]
    if missing_keys:
        raise RuntimeError("Trained Region model did not return: " + ", ".join(missing_keys))

    folder.mkdir(parents=True, exist_ok=True)
    raw_dir = folder / "native_arrays"
    raw_dir.mkdir(exist_ok=True)
    names = {
        "edge_prob": "01_edge", "orientation": "02_orientation",
        "contrast_hint": "03_contrast_hint", "skeleton_prob": "04_skeleton",
        "width_expected": "05_width", "tangent": "06_tangent",
        "polarity": "07_polarity", "contrast_mag": "08_contrast_magnitude",
    }
    for source, name in names.items():
        np.save(raw_dir / f"{name}.npy", np.asarray(arrays[source], dtype=np.float32))

    edge = arrays["edge_prob"][0]
    skeleton = arrays["skeleton_prob"][0]
    orient = arrays["orientation"]
    tangent = arrays["tangent"]
    h, w = observed.shape
    for key in ("edge_prob", "skeleton_prob", "width_expected", "contrast_hint",
                "polarity", "contrast_mag"):
        if arrays[key].shape != (1, h, w):
            raise ValueError(f"Unexpected trained {key} shape: {arrays[key].shape}")
    if orient.shape != (2, h, w) or tangent.shape != (2, h, w):
        raise ValueError("Geometry normal and Topology tangent must have native (2,H,W) fields")

    normal_angle = np.mod(0.5 * np.arctan2(orient[1], orient[0]), np.pi)
    tangent_angle = np.mod(np.arctan2(tangent[1], tangent[0]), np.pi)
    np.save(raw_dir / "02_geometry_normal_angle_radians.npy", normal_angle.astype(np.float32))
    np.save(raw_dir / "06_topology_tangent_angle_radians.npy", tangent_angle.astype(np.float32))

    # Orientation is a Geometry-predicted normal. Tangent is a Topology-predicted
    # line direction. Each is visualized with its OWN native angle and confidence.
    def direction_rgb(angle, strength):
        hsv = np.zeros((h, w, 3), dtype=np.float32)
        hsv[..., 0] = angle / np.pi
        hsv[..., 1] = 0.88
        hsv[..., 2] = 0.94
        directional = colors.hsv_to_rgb(hsv)
        alpha = np.clip(strength, 0.0, 1.0)[..., None] * 0.83
        bg = np.repeat(np.clip(observed[..., None], 0, 1), 3, axis=-1)
        return np.clip(bg * (1.0 - alpha) + directional * alpha, 0, 1)

    displays = [
        ("01_edge", "Edge (Geometry probability)", edge, "gray", 0, 1),
        ("02_orientation", "Orientation (Geometry NORMAL)",
         direction_rgb(normal_angle, edge), None, None, None),
        ("03_contrast_hint", "Contrast Hint (Geometry, signed)",
         arrays["contrast_hint"][0], "coolwarm", -1, 1),
        ("04_skeleton", "Skeleton (Topology probability)", skeleton, "magma", 0, 1),
        ("05_width", "Width (Topology scale)", arrays["width_expected"][0], "viridis", 0, 1),
        ("06_tangent", "Tangent (Topology LINE direction)",
         direction_rgb(tangent_angle, skeleton), None, None, None),
        ("07_polarity", "Polarity (dark - / bright +)",
         arrays["polarity"][0], "coolwarm", -1, 1),
        ("08_contrast_magnitude", "Contrast Magnitude (Topology)",
         arrays["contrast_mag"][0], "magma", 0, 4),
    ]

    def draw(ax, data, title, cmap, vmin, vmax):
        # Image only: no titles, borders, axes, or margins in the saved figures.
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
        ax.axis("off")

    for name, title, data, cmap, vmin, vmax in displays:
        fig = plt.figure(figsize=(5, 5), dpi=300)
        ax = fig.add_axes([0, 0, 1, 1])
        draw(ax, data, title, cmap, vmin, vmax)
        fig.savefig(folder / f"{name}.png", dpi=300, pad_inches=0)
        plt.close(fig)

    # Label-free eight-panel contact sheet, with no whitespace between panels.
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=150)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    for ax, (_name, title, data, cmap, vmin, vmax) in zip(axes.flat, displays):
        draw(ax, data, title, cmap, vmin, vmax)
    fig.savefig(folder / "00_eight_maps_overview.png", dpi=300, pad_inches=0)
    plt.close(fig)
    (folder / "README.txt").write_text(
        "All eight maps are forward outputs of the trained Geometry + Topology models.\n"
        "Geometry Orientation = predicted edge NORMAL from (cos2theta,sin2theta).\n"
        "Topology Tangent = separately learned line-direction vector.\n"
        "Orientation/Tangent PNGs colorize directions, blended by native edge/skeleton confidence.\n"
        "Raw native tensors and decoded angles are in native_arrays/.\n"
        "The gap is NOT blanked: learned maps predict features INSIDE missing pixels.\n"
        "All eight saved PNGs and their overview are title-free and border-free.\n"
        "Missing-region outlines are omitted from the exported maps; no hidden target values enter the model.\n"
        "Width is a learned normalized scale, not measured physical thickness in pixels.\n",
        encoding="utf-8",
    )
    return folder / "00_eight_maps_overview.png"


def save_region_architecture_maps(out, band, wavelengths, preset):
    """Export native intermediate features of the trained Region branch.

    This diagnostic forward pass is independent of the learned router's chosen
    expert. It uses exactly the same input ROI and 3-D corruption mask as the
    router evaluation, rather than an artificially substituted 2-D mask.
    """
    import torch
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from src import spatial_inference

    work = out / "_work"
    roi = np.asarray(np.load(work / "roi_cube.npy", mmap_mode="r"), dtype=np.float32)
    mask = np.asarray(np.load(work / "model_mask.npy", mmap_mode="r"), dtype=bool)
    targets = np.asarray(np.load(work / "synthetic_mask.npy", mmap_mode="r"), dtype=bool)
    invalid = np.asarray(np.load(work / "roi_invalid.npy", mmap_mode="r"), dtype=bool)
    affected = targets[:, :, band] & (~invalid[:, :, band]) & np.isfinite(roi[:, :, band])
    if not np.any(affected):
        present = np.flatnonzero(np.any(targets & ~invalid & np.isfinite(roi), axis=(0, 1)))
        if not len(present):
            print("Region figure skipped: no valid target voxels", flush=True)
            return
        original_band = band
        band = int(present[np.argmin(np.abs(present - band))])
        print(f"Region figure: display band {original_band} not masked; using affected band {band}", flush=True)
        affected = targets[:, :, band] & (~invalid[:, :, band]) & np.isfinite(roi[:, :, band])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # This diagnostic uses the same final Region as the three-expert pipeline.
    from src.lineaware_region import load_lineaware_region
    topo, ck, geom, _gck, topo_path, geom_path = load_lineaware_region(
        device, wavelengths, full_split=True)
    ps = int(ck.get("patch_size", 65))
    if ps > min(roi.shape[:2]):
        raise ValueError(f"Model patch {ps} exceeds ROI dimensions")
    yy, xx = np.where(affected)
    cy, cx = int(np.median(yy)), int(np.median(xx))
    half = ps // 2
    cy = int(np.clip(cy, half, roi.shape[0] - half - 1))
    cx = int(np.clip(cx, half, roi.shape[1] - half - 1))
    area = np.column_stack((yy, xx)).astype(np.int64)
    batch, meta = spatial_inference.prepare_patch(roi, mask, band, area, ck, center_override=(cy, cx))
    sy0, sx0, dy0, dx0, _ps, _mean, _std = meta
    with torch.inference_mode():
        batch = {k: v.to(device) for k, v in batch.items()}
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                geometry = geom.structure_net(**batch)
                topology = topo(structure=geometry, **batch)
        else:
            geometry = geom.structure_net(**batch)
            topology = topo(structure=geometry, **batch)
    combined = dict(geometry)
    combined.update(topology)
    expected = ("edge_prob", "orientation", "contrast_hint", "skeleton_prob",
                "width_expected", "tangent", "polarity", "contrast_mag", "support_prob",
                "transported", "transport_confidence", "radiometry", "detail", "prediction")
    absent = [key for key in expected if key not in combined]
    if absent:
        raise RuntimeError(f"Region checkpoint does not return required maps: {absent}")
    arrays = {k: combined[k][0].detach().float().cpu().numpy() for k in expected}

    y_indices = np.clip(np.arange(ps) + sy0 - dy0, 0, roi.shape[0]-1)
    x_indices = np.clip(np.arange(ps) + sx0 - dx0, 0, roi.shape[1]-1)
    truth = roi[np.ix_(y_indices, x_indices, [band])][:, :, 0]
    masked = mask[np.ix_(y_indices, x_indices, [band])][:, :, 0]
    synthetic = affected[np.ix_(y_indices, x_indices)]
    measured = np.isfinite(truth) & (~masked)
    if not np.any(measured):
        raise RuntimeError("No observed pixels in this patch")
    lo, hi = np.percentile(truth[measured], [1, 99.5])
    hi = max(float(hi), float(lo) + 1e-6)
    def display(v):
        return np.clip((v - lo) / (hi - lo), 0., 1.)

    observed = display(truth)
    observed[masked] = 0.0
    norm_mean = float(np.asarray(ck["band_mean"])[band])
    norm_std = float(np.asarray(ck["band_std"])[band])
    # These four network heads predict standardized target-band intensity.
    def physical(key, centered=True):
        t = arrays[key][0] * norm_std
        return t + norm_mean if centered else t

    region_only = display(truth)
    native_region = physical("prediction")
    region_only[synthetic] = display(native_region)[synthetic]
    folder = out / "region_architecture_maps"
    folder.mkdir(parents=True, exist_ok=True)
    orient_visual, smooth_angle, angle_confidence = make_geometry_orientation_views(
        arrays["orientation"], arrays["edge_prob"][0], observed
    )
    # Retain the unmodified two-channel model output for auditing.
    np.save(folder / "05_geometry_orientation_native_cos2_sin2.npy", arrays["orientation"])
    orientation_lines, line_count = save_orientation_line_view(
        folder / "05_geometry_orientation_lines.png",
        observed, smooth_angle, angle_confidence, mask=None
    )
    eight_figure = export_eight_native_maps(
        folder / "eight_native_maps", arrays, observed, synthetic, wavelengths, band
    )
    np.save(folder / "05_geometry_orientation_line_confidence.npy", angle_confidence.astype(np.float32))
    print(f"Geometry orientation: drawn {line_count} confidence-filtered direction segments.", flush=True)

    data = [
        ("01_original", display(truth), "gray", 0, 1),
        ("02_corrupted", observed, "gray", 0, 1),
        ("03_mask", synthetic.astype(float), "gray", 0, 1),
        ("04_geometry_edges", arrays["edge_prob"][0], "gray", 0, 1),
        ("05_geometry_orientation", orient_visual, None, None, None),
        ("06_topology_skeleton", arrays["skeleton_prob"][0], "magma", 0, 1),
        ("07_topology_support", arrays["support_prob"][0], "magma", 0, 1),
        ("08_transported_reflectance", display(physical("transported")), "gray", 0, 1),
        ("09_transport_confidence", arrays["transport_confidence"][0], "viridis", 0, 1),
        ("10_radiometry", display(physical("radiometry")), "gray", 0, 1),
        ("11_detail_residual", physical("detail", centered=False), "coolwarm", None, None),
        ("12_region_estimate", region_only, "gray", 0, 1),
    ]
    for name, a, cmap, vmin, vmax in data:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
        ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.axis("off")
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        fig.savefig(folder / (name + ".png"), dpi=300, pad_inches=0)
        plt.close(fig)
        np.save(folder / (name + ".npy"), a)

    panels = [
        ("Corrupted patch", observed, "gray"),
        ("Geometry: edges", arrays["edge_prob"][0], "gray"),
        ("Geometry: orientation (lines)", orientation_lines, None),
        ("Topology: skeleton", arrays["skeleton_prob"][0], "magma"),
        ("Topology: support", arrays["support_prob"][0], "magma"),
        ("Correspondence transport", display(physical("transported")), "gray"),
        ("Radiometry", display(physical("radiometry")), "gray"),
        ("Region-only estimate", region_only, "gray"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for ax, (name, panel, cm) in zip(axes.flat, panels):
        ax.imshow(panel, cmap=cm, interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Region expert native intermediate maps | {wavelengths[band]:.0f} nm",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(folder / "13_region_architecture_overview.png", dpi=300)
    fig.savefig(folder / "13_region_architecture_overview.svg")
    plt.close(fig)

    (folder / "metadata.json").write_text(json.dumps({
        "source": "Trained Geometry + Topology checkpoints",
        "model_preset": preset,
        "eight_maps_overview": str(eight_figure),
        "geometry_checkpoint": str(geom_path),
        "topology_checkpoint": str(topo_path),
        "band": int(band),
        "wavelength_nm": float(wavelengths[band]),
        "patch_size": int(ps),
        "patch_center": [int(cy), int(cx)],
        "orientation_display": "Decoded from [cos(2theta), sin(2theta)] with half atan2; overlay restricted by edge confidence; line plot shows geological tangents. Model predictions were not altered.",
        "orientation_line_segments_drawn": int(line_count),
        "note": "Region expert independently evaluated with the same 3-D ROI corruption mask. "
                "This output is NOT necessarily the final learned-router TPSSR estimate. "
                "Display normalization is for visualization only."
    }, indent=2), encoding="utf-8")
    print(f"Saved trained Region maps: {folder}", flush=True)
    print(f"Saved eight native maps: {eight_figure}", flush=True)



def restore_region_work_from_output(out, cube, invalid):
    """Restore the exact ROI corruption input saved by earlier visualizer runs.

    We use the original calibrated cube and the already-saved synthetic mask,
    rather than the TPSSR reconstruction, to avoid leaking reconstructed values
    or the ground-truth values of missing pixels into the network input.
    """
    required = ("roi_ground_truth.npy", "roi_corrupted.npy", "roi_synthetic_mask.npy")
    absent = [name for name in required if not (out / name).is_file()]
    if absent:
        raise FileNotFoundError(
            f"Cannot regenerate the Region maps from {out}: missing {absent}. "
            "The previous ROI reconstruction must have saved its output arrays.")
    roi = np.asarray(np.load(out / "roi_ground_truth.npy"), dtype=np.float32)
    corrupted = np.asarray(np.load(out / "roi_corrupted.npy"), dtype=np.float32)
    synthetic = np.asarray(np.load(out / "roi_synthetic_mask.npy"), dtype=bool)
    invalid_roi = np.asarray(invalid, dtype=bool)
    if roi.shape != corrupted.shape or roi.shape != synthetic.shape or roi.shape != invalid_roi.shape:
        raise ValueError("Saved ROI arrays and original invalid mask have incompatible shapes")
    if roi.ndim != 3 or roi.shape[0:2] != (ROI_SIZE, ROI_SIZE):
        raise ValueError("Expected original fixed 128x128 ROI with spectral bands")
    model_mask = synthetic | invalid_roi | ~np.isfinite(roi)
    if not np.array_equal(~np.isfinite(corrupted), model_mask):
        raise RuntimeError(
            "Saved corrupted ROI is inconsistent with the synthetic/invalid masks; "
            "refusing to generate maps from an altered input")
    work = out / "_work"
    existed = work.exists()
    work.mkdir(exist_ok=True)
    outputs = {
        "roi_cube.npy": roi,
        "roi_invalid.npy": invalid_roi,
        "synthetic_mask.npy": synthetic,
        "model_mask.npy": model_mask,
    }
    for filename, values in outputs.items():
        dest = work / filename
        if dest.exists():
            previous = np.load(dest, mmap_mode="r")
            if previous.shape != values.shape or not np.array_equal(previous, values, equal_nan=True):
                raise RuntimeError(f"Existing {dest} differs from the saved ROI")
        else:
            np.save(dest, values)
    return work, not existed


def regenerate_region_maps(existing_output):
    """Re-run only the split-specific trained Geometry/Topology diagnostics."""
    from experiment_config import resolve_experiment
    from _launcher_common import env_for
    out = Path(existing_output).expanduser().resolve()
    metadata_path = out / "selection.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Could not find {metadata_path}. Pass the ROI run's output directory.")
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    preset = meta.get("model_preset", meta.get("preset"))
    if preset not in DEFAULT_DATASETS:
        raise ValueError(f"Unknown saved preset for Region map recovery: {preset!r}")
    ds = str(meta["dataset"]).upper()
    if (ds.startswith("VNIR")) != (preset == "vnir_8x2"):
        raise ValueError(f"Dataset {ds} does not belong to the sensor used by {preset}")
    if meta.get("router") not in ("learned_router3_lineaware_rankcv", "learned_router3_lineaware"):
        raise ValueError("Saved ROI was not made with the Line-aware Region model")
    band = int(meta["display_band"])

    # Reproduce the very same experiment configuration/checkpoint identity.
    args = argparse.Namespace(preset=preset, dataset=None, active_test=None,
                              train_datasets=None, test_datasets=None, experiment_name=None)
    spec = resolve_experiment(args)
    os.environ.update(env_for(spec, spec.active_test))
    if str(CORE) not in sys.path:
        sys.path.insert(0, str(CORE))
    from src.evaluation_utils import load_dataset
    cube, invalid, _split, wavelengths = load_dataset(ds)
    y = int(meta["roi_y_in_full_image"])
    x = int(meta["roi_x_in_full_image"])
    invalid_roi = invalid[y:y + ROI_SIZE, x:x + ROI_SIZE]
    if invalid_roi.shape != (ROI_SIZE, ROI_SIZE, len(wavelengths)):
        raise RuntimeError("The original source ROI is no longer accessible at its saved coordinates")
    saved_truth = np.load(out / "roi_ground_truth.npy", mmap_mode="r")
    source_truth = cube[y:y+ROI_SIZE, x:x+ROI_SIZE]
    if not np.array_equal(saved_truth, source_truth, equal_nan=True):
        raise RuntimeError("Calibrated source data changed since the original ROI visualization")
    del cube, source_truth

    work, created = restore_region_work_from_output(out, saved_truth, invalid_roi)
    print("[MAPS ONLY] Reusing the saved crop and missing-data mask; no TPSSR rerun.", flush=True)
    try:
        save_region_architecture_maps(out, band, wavelengths, preset)
        check_eight_maps(out)
    finally:
        if created:
            shutil.rmtree(work)
    return out


def main():
    args = arguments()
    if args.band_wide_eval is None:
        args.band_wide_eval = ("all" if args.scenario == "band_wide_saturation"
                               and args.band_wide_mode == "full-roi" else "benchmark")
    if args.spatial_consistency and args.router != "learned_router3_lineaware_rankcv":
        raise SystemExit("--spatial-consistency requires --router learned_router3_lineaware_rankcv")
    if args.global_minority_percent is None and args.minority_policy in ('hybrid', 'hybrid_gap'):
        raise SystemExit("Hybrid minority policies require --global-minority-percent")
    if args.global_minority_percent is not None:
        os.environ["TPSSR_FUSION_DONOR_MATCH"] = "robust"
        if args.router != "learned_router3_lineaware_rankcv":
            raise SystemExit("Global Minority requires the RankCV Line-aware Router")
        if not 0 < args.global_minority_percent < 50:
            raise SystemExit("--global-minority-percent must be between 0 and 50")
        if not 0 <= args.three_expert_min_gap_percent < 100:
            raise SystemExit("--three-expert-min-gap-percent must be between 0 and 100")
        if args.spatial_consistency or args.fusion_guard:
            raise SystemExit("Global Minority is incompatible with Spatial Consistency and Fusion Guard")
    if args.fusion_guard and (args.router != "learned_router3_lineaware_rankcv"
                              or not args.spatial_consistency or args.preset != "swir_8x2"):
        raise SystemExit("--fusion-guard requires swir_8x2 RankCV with --spatial-consistency")
    if args.maps_from_output is not None:
        regenerate_region_maps(args.maps_from_output)
        return
    if args.scenario != "band_wide_saturation" and args.band_wide_mode != "local":
        raise SystemExit("--band-wide-mode full-roi is only supported for band_wide_saturation")
    if args.scenario != "band_wide_saturation" and args.band_wide_eval != "benchmark":
        raise SystemExit("--band-wide-eval is only supported for band_wide_saturation")
    if args.band_wide_mode == "local" and args.band_wide_eval != "benchmark":
        raise SystemExit("--band-wide-eval all requires --band-wide-mode full-roi")
    from experiment_config import PRESETS, resolve_experiment, sensor_of, SUPPORTED_DATASETS
    from _launcher_common import env_for

    # The chosen image never changes the experiment's checkpoint family.
    # Default to a held-out acquisition rather than a training image.
    display_dataset = str(args.dataset or DEFAULT_DATASETS[args.preset]).upper()
    if args.flip_x is None:
        args.flip_x = display_dataset.startswith("VNIR")
    if display_dataset not in SUPPORTED_DATASETS:
        raise SystemExit(f"Unsupported dataset {display_dataset}. Choose SWIR1..SWIR10 (or VNIR1..VNIR10 with vnir_8x2).")
    args.dataset = None
    args.active_test = None
    args.train_datasets = None
    args.test_datasets = None
    args.experiment_name = None
    spec = resolve_experiment(args)
    args.dataset = display_dataset
    if sensor_of(display_dataset) != spec.sensor:
        raise SystemExit(f"Cannot visualize {display_dataset} with {args.preset}: sensor mismatch.")
    if display_dataset not in (*spec.train, *spec.tests):
        raise SystemExit(f"{display_dataset} is not part of the selected model experiment {args.preset}.")
    if args.region_figure and args.router == "learned_router3_aligned":
        raise SystemExit("Eight native maps require a Line-aware Router checkpoint; use --router learned_router3_lineaware_rankcv")
    role = "held-out test" if display_dataset in spec.tests else "model training/validation"
    print(f"Visualizing {display_dataset} ({role}) with the trained {args.preset} model.", flush=True)
    if display_dataset in spec.train:
        print("NOTE: This acquisition was used to train/validate this model. Visual results on it "
              "are not independent test results.", flush=True)

    # The active test acquisition stays within the experiment test set. This
    # ensures config.MODELS_DIR and the original model checkpoint identity are
    # unchanged even when the visualized image is a training acquisition.
    model_env = env_for(spec, spec.active_test)
    os.environ.update(model_env)
    if str(CORE) not in sys.path:
        sys.path.insert(0, str(CORE))
    if args.worker:
        run_worker(args)
        return

    import config
    if not args.select_only:
        # Detect an incomplete split before the user selects an ROI or runs two GPU workers.
        from training.neighbor_fusion import checkpoint_path as fusion_checkpoint_path
        from training.three_expert_router import path_for as router_checkpoint_path
        required = {
            "Learned Neighbor Fusion": fusion_checkpoint_path(Path(config.MODELS_DIR)),
            "RankCV Router": router_checkpoint_path(
                Path(config.MODELS_DIR), objective="rank_cv", region_expert="lineaware"),
        }
        missing = [(name, path) for name, path in required.items() if not path.is_file()]
        if missing:
            lines = "\n".join(f"  {name}: {path}" for name, path in missing)
            raise SystemExit(
                f"Missing trained checkpoints for {args.preset}:\n{lines}\n"
                "Run preflight.py for this preset before reconstruction. "
                "Do not substitute checkpoints from another data split.")
    from src.evaluation_utils import load_dataset, split_blocks
    cube, invalid, split, waves = load_dataset(display_dataset)
    if args.image_scope == "test-block" and display_dataset not in spec.tests:
        raise SystemExit("--image-scope test-block is only valid for held-out test images. Use --image-scope full.")
    use_test_block = (args.image_scope == "test-block" or
                      (args.image_scope == "auto" and display_dataset in spec.tests))
    if use_test_block:
        _, _, test_cube = split_blocks(cube, split)
        _, _, test_invalid = split_blocks(invalid, split)
        image_offset = int(split["val_end"])
        area_label = "held-out test block"
    else:
        test_cube, test_invalid = cube, invalid
        image_offset = 0
        area_label = "full image"
    height, width, bands = test_cube.shape
    band = bands // 2 if args.band is None else args.band
    if not (0 <= band < bands):
        raise SystemExit(f"--band must be between 0 and {bands-1}")
    print(f"{args.dataset} {area_label}: {height} x {width} x {bands} | fixed ROI: {ROI_SIZE} x {ROI_SIZE}", flush=True)
    preview, sy, sx = preview_image(test_cube, test_invalid, band, args.preview_max_height, args.preview_max_width)
    if (args.roi_y is None) != (args.roi_x is None):
        raise SystemExit("Pass both --roi-y and --roi-x, or neither for interactive selection")
    if args.roi_y is None:
        y, x = select_box(preview, height, width, sy, sx, args.dataset, band, area_label,
                          flip_x=args.flip_x)
    else:
        y, x = boundaries(args.roi_y, args.roi_x, height, width)
    roi = np.array(test_cube[y:y + ROI_SIZE, x:x + ROI_SIZE], dtype=np.float32, copy=True)
    invalid_roi = np.asarray(test_invalid[y:y + ROI_SIZE, x:x + ROI_SIZE], dtype=bool).copy()
    full_band_wide = args.scenario == "band_wide_saturation" and args.band_wide_mode == "full-roi"
    if full_band_wide:
        if args.missing_y is not None or args.missing_x is not None:
            raise SystemExit("--missing-y/--missing-x do not apply to full-roi Band-wide Saturation")
        # The corrupted band spans the entire ROI. No second draggable window.
        missing_y, missing_x = 0, 0
        print("[BAND-WIDE FULL ROI] No missing-location window: entire 128x128 tile will lose selected bands.", flush=True)
    else:
        if (args.missing_y is None) != (args.missing_x is None):
            raise SystemExit("Pass both --missing-y and --missing-x, or neither for the second interactive window")
        if args.missing_y is None:
            missing_y, missing_x = select_missing_window(
                roi, invalid_roi, args.scenario, args.severity, band, args.seed, args.whole_spectrum_size,
                flip_x=args.flip_x)
        else:
            missing_y, missing_x = int(args.missing_y), int(args.missing_x)
    synthetic = make_local_mask(
        roi.shape, args.scenario, args.severity, band, args.seed,
        missing_y=missing_y, missing_x=missing_x,
        whole_spectrum_size=args.whole_spectrum_size, band_wide_mode=args.band_wide_mode)
    if args.scenario == "whole_spectrum_pixels":
        # All pixels inside a displayed whole-spectrum square must have valid
        # reference spectra. Reject a bad location instead of silently shrinking
        # the artificial mask (which would leave bright lines/holes inside it).
        side = missing_window_size(args.scenario, args.severity, args.whole_spectrum_size)
        bad = count_invalid_whole_spectrum_pixels(roi, invalid_roi, missing_y, missing_x, side)
        if bad:
            raise SystemExit(
                f"Whole-spectrum selection rejected: {bad}/{side * side} pixels in the "
                f"{side}x{side} square already contain invalid spectral bands. "
                "Move the missing window to an area with fully valid spectra. "
                "No inference was launched and no output files were written.")
    target = synthetic & ~invalid_roi & np.isfinite(roi)
    model_mask = synthetic | invalid_roi | ~np.isfinite(roi)
    if full_band_wide and args.band_wide_eval == "benchmark":
        # This is the EXACT official coordinate-sampling function (margin,
        # spatial cap, voxel cap and seeded selection), but the selected tile
        # is user-chosen rather than drawn by the official image-wide generator.
        from src.evaluation_utils import select_eval_coords
        coords = select_eval_coords(target, args.seed).astype(np.int32)
    else:
        coords = np.argwhere(target).astype(np.int32)
    if len(coords) < 1:
        raise SystemExit("No valid masked pixels in this ROI. Move the ROI onto valid sediment.")
    if not full_band_wide and (len(coords) > 250000 or np.count_nonzero(np.any(target, axis=2)) > 2500):
        raise SystemExit("The requested mask is too large for this visual test. Try another scenario or severity.")
    if full_band_wide:
        if args.band_wide_eval == "all" and len(coords) > 250000:
            print(f"[WARNING] All-target full-roi run: {len(coords):,} voxels. "
                  "TPSSR inference can require considerable GPU memory and time.", flush=True)
        selected_band_count = np.flatnonzero(np.any(synthetic, axis=(0, 1)))
        print(f"[BAND-WIDE FULL ROI] selected bands: {int(selected_band_count[0])}.."
              f"{int(selected_band_count[-1])} ({len(selected_band_count)} consecutive); "
              f"masked valid voxels={int(target.sum()):,}; "
              f"evaluated voxels={len(coords):,}; "
              f"evaluated spatial pixels={len(np.unique(coords[:, :2], axis=0)):,}; "
              f"eval={args.band_wide_eval}", flush=True)
    elif args.scenario == "band_wide_saturation":
        print("NOTE: LOCAL Band-wide example only; use --band-wide-mode full-roi for benchmark-like masking.", flush=True)
    elif args.scenario == "mixed_saturation":
        print("NOTE: mixed_saturation uses a LOCAL, centered composite for visual comparison; "
              "it does not replicate the independently placed full-image benchmark components.", flush=True)
    mode_suffix = f"_{args.band_wide_mode}_{args.band_wide_eval}" if args.scenario == "band_wide_saturation" else ""
    out = (Path(args.outdir) if args.outdir else ROOT / "roi_visuals" / args.preset / args.dataset /
           f"{args.scenario}_S{args.severity}{mode_suffix}_hybridgap10_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output directory is not empty: {out}. Choose another --outdir.")
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)

    np.save(work / "roi_cube.npy", roi)
    np.save(work / "roi_invalid.npy", invalid_roi)
    np.save(work / "synthetic_mask.npy", synthetic)
    np.save(work / "model_mask.npy", model_mask)
    np.save(work / "coords.npy", coords)
    save_selection(out, preview, y, x, sy, sx, image_offset, roi, synthetic, model_mask, band,
                   flip_x=args.flip_x)
    params = {
        "dataset": args.dataset, "preset": args.preset,
        "spatial_consistency": bool(args.spatial_consistency),
        "global_minority_percent": args.global_minority_percent,
        "global_minority_policy": (args.minority_policy if args.global_minority_percent is not None else 'off'),
        "three_expert_min_gap_percent": args.three_expert_min_gap_percent,
        "fusion_donor_match": os.environ.get("TPSSR_FUSION_DONOR_MATCH", "legacy"),
        "fusion_guard": bool(args.fusion_guard),
        "display_flip_x": bool(args.flip_x),
        "scenario": args.scenario, "severity": args.severity,
        "band_wide_mode": args.band_wide_mode,
        "band_wide_eval": args.band_wide_eval,
        "band_wide_selected_bands": (np.flatnonzero(np.any(synthetic, axis=(0, 1))).astype(int).tolist()
                                      if args.scenario == "band_wide_saturation" else None),
        "full_mask_valid_target_voxels": int(target.sum()),
        "evaluated_target_voxels": int(len(coords)),
        "visual_mask_source": ("benchmark_style_full_band_on_roi" if full_band_wide else
                               "single_benchmark_size_component" if args.scenario in
                               {"whole_spectrum_pixels", "spatial_patch", "spatio_spectral_block"}
                               else "compact_benchmark_S5_components" if args.scenario == "mixed_saturation" and args.severity == 5
                               else "src.training_corruptions.generate_training_mask"),
        "visual_mask_note": ("Full-band loss across a user-selected 128x128 tile, not across the "
                             "entire official test acquisition. Width matches the benchmark. "
                             "Official coordinate sampling is used when eval=benchmark. "
                             "Band location/ROI and seed are user-selected, not the official case."
                             if full_band_wide else
                             "One local representative corruption; NOT a full standardized evaluation mask. "
                             "Compound/line masks use the local training-style generator; "
                             "the placement square is not necessarily the missing area."),
        "whole_spectrum_size": (int((1, 3, 5, 9, 15)[args.severity-1])
                                if args.whole_spectrum_size is None else int(args.whole_spectrum_size))
                               if args.scenario == "whole_spectrum_pixels" else None,
        "roi_size": ROI_SIZE, "defect_window_size": (ROI_SIZE if full_band_wide else missing_window_size(
            args.scenario, args.severity, args.whole_spectrum_size)),
        "missing_y_in_roi": int(missing_y), "missing_x_in_roi": int(missing_x),
        "missing_y_in_full_image": int(image_offset + y + missing_y),
        "missing_x_in_full_image": int(x + missing_x),
        "whole_spectrum_mask_rect_full_image": (
            [int(x + missing_x), int(image_offset + y + missing_y),
             int(x + missing_x + missing_window_size(args.scenario, args.severity, args.whole_spectrum_size)),
             int(image_offset + y + missing_y + missing_window_size(args.scenario, args.severity, args.whole_spectrum_size))]
            if args.scenario == "whole_spectrum_pixels" else None),
        "image_role_in_model_experiment": role,
        "image_scope": "test-block" if use_test_block else "full",
        "model_preset": args.preset,
        "region_backend": "lineaware" if args.router != "learned_router3_aligned" else "aligned_default",
        "router": args.router,
        "model_primary_test": spec.primary_test,
        "model_training_datasets": list(spec.train),
        "model_held_out_datasets": list(spec.tests),
        "independent_held_out_evaluation": bool(display_dataset in spec.tests),
        "roi_y_in_selected_area": y, "roi_x_in_selected_area": x,
        "roi_y_in_full_image": image_offset + y, "roi_x_in_full_image": x,
        "display_band": band, "display_wavelength_nm": float(waves[band]),
        "seed": args.seed, "requested_gpus": args.gpus,
    }
    (out / "selection.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"Selected ROI: full-image x={x}:{x + ROI_SIZE}, y={image_offset+y}:{image_offset+y+ROI_SIZE} | mask targets: {len(coords):,} voxels", flush=True)
    side = missing_window_size(args.scenario, args.severity, args.whole_spectrum_size)
    if not full_band_wide:
        print(f"Moved Missing window inside ROI: x={missing_x}:{missing_x+side}, y={missing_y}:{missing_y+side}", flush=True)
    if args.scenario == "whole_spectrum_pixels":
        effective_pixels = int(np.count_nonzero(np.any(target, axis=2)))
        print(f"Effective whole-spectrum masked pixels: {effective_pixels} "
              f"(nominal square {params['whole_spectrum_size']}x{params['whole_spectrum_size']})", flush=True)
        expected_pixels = int(params['whole_spectrum_size']) ** 2
        if effective_pixels != expected_pixels:
            raise RuntimeError("Whole-spectrum mask is not a complete square. "
                               "This indicates an inconsistent selection or target mask.")
    # Region uses 65x65 patches; a moved mask near the ROI boundary may provide
    # less real context than the historical full-image benchmark.
    affected_yx = np.argwhere(np.any(target, axis=2))
    if len(affected_yx):
        if (affected_yx[:, 0].min() < 32 or affected_yx[:, 1].min() < 32 or
                affected_yx[:, 0].max() > ROI_SIZE - 33 or
                affected_yx[:, 1].max() > ROI_SIZE - 33):
            print("WARNING: The Region 65x65 context around this mask may reach beyond the "
                  "128x128 ROI. Move the fixed window inward for a more faithful "
                  "comparison with full-image inference.", flush=True)
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
            "--router", args.router, "--scenario", args.scenario, "--severity", str(args.severity),
            "--outdir", str(out), "--region-batch", str(args.region_batch), "--worker"]
    if args.spatial_consistency:
        cmd.append("--spatial-consistency")
    if args.global_minority_percent is not None:
        cmd.extend(["--global-minority-percent", str(args.global_minority_percent),
                    "--minority-policy", args.minority_policy,
                    "--three-expert-min-gap-percent", str(args.three_expert_min_gap_percent)])
    print(f"Running trained {args.router} on {args.gpus} GPU(s)...", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=model_env, check=True)
    coords, values, routes = merge_outputs(out, args.gpus)
    stats = save_reconstruction(out, band, waves, coords, values, routes, params)
    if args.region_figure:
        save_region_architecture_maps(out, band, waves, args.preset)
        check_eight_maps(out)
    shutil.rmtree(work)
    print(f"Complete | RMSE={stats['RMSE']:.6f} | MAE={stats['MAE']:.6f} | coverage=100%", flush=True)
    print(f"Comparison figure: {out / '03_tpssr_comparison.png'}", flush=True)
    print(f"Saved all outputs to: {out}", flush=True)


if __name__ == "__main__":
    main()
