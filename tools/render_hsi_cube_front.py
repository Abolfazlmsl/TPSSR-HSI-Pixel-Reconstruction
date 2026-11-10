#!/usr/bin/env python3
"""Create a paper-ready HSI cube with the sediment image on the FRONT face.

New in this version:
- Optional 4-point interactive quadrilateral selection (`--select-quad`)
- Perspective rectification of the chosen ROI before 3D rendering
- Keeps the older rectangle crop selection (`--select-crop`)
- Brighter grayscale control for SWIR display

Typical usage
-------------
python render_hsi_cube_front.py --dataset SWIR5 --select-quad --front-mode gray
python render_hsi_cube_front.py --dataset SWIR5 --select-crop --front-mode gray
python render_hsi_cube_front.py --dataset SWIR5 --full-image --front-mode gray
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import Affine2D
from matplotlib.widgets import RectangleSelector


# This display path matches export_swir_middle_band.py. It is only used to
# construct the front face. The two spectral faces still use the TPSSR cube.
ENVI_DTYPES = {
    1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32,
    5: np.float64, 6: np.complex64, 9: np.complex128,
    12: np.uint16, 13: np.uint32, 14: np.int64, 15: np.uint64,
}


def parse_envi_header(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    result = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.upper() == "ENVI" or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip().lower(), value.strip()
        if value.startswith("{") and "}" not in value:
            parts = [value]
            while i < len(lines):
                parts.append(lines[i].strip())
                if "}" in lines[i]:
                    i += 1
                    break
                i += 1
            value = " ".join(parts)
        result[key] = value
    return result


def find_envi_binary(hdr: Path) -> Path:
    base = hdr.with_suffix("")
    for candidate in [base.with_suffix(ext) for ext in (".raw", ".RAW", ".dat", ".DAT")] + [base]:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"ENVI binary not found: {hdr}")


class EnviCube:
    def __init__(self, hdr: Path):
        head = parse_envi_header(hdr)
        self.lines = int(float(head["lines"]))
        self.samples = int(float(head["samples"]))
        self.bands = int(float(head["bands"]))
        self.interleave = head.get("interleave", "bil").strip().lower().strip("{}")
        code = int(float(head["data type"]))
        dtype = np.dtype(ENVI_DTYPES[code])
        byte_order = int(float(head.get("byte order", "0").strip().strip("{}")))
        dtype = dtype.newbyteorder("<" if byte_order == 0 else ">")
        offset = int(float(head.get("header offset", "0").strip().strip("{}")))
        if self.interleave == "bil":
            shape = (self.lines, self.bands, self.samples)
        elif self.interleave == "bip":
            shape = (self.lines, self.samples, self.bands)
        elif self.interleave == "bsq":
            shape = (self.bands, self.lines, self.samples)
        else:
            raise ValueError(f"Unsupported ENVI interleave: {self.interleave}")
        self.mm = np.memmap(find_envi_binary(hdr), dtype=dtype, mode="r", offset=offset, shape=shape)
        raw_wl = re.split(r"[\s,]+", head.get("wavelength", "").replace("{", " ").replace("}", " ").strip())
        try:
            wl = np.asarray([float(v) for v in raw_wl if v], dtype=np.float64)
        except ValueError:
            wl = np.array([])
        self.wavelengths = wl if len(wl) == self.bands else np.arange(self.bands, dtype=np.float64)

    def read_lines(self, y0: int, y1: int) -> np.ndarray:
        if self.interleave == "bil":
            return np.asarray(self.mm[y0:y1]).transpose(0, 2, 1)
        if self.interleave == "bip":
            return np.asarray(self.mm[y0:y1])
        return np.asarray(self.mm[:, y0:y1, :]).transpose(1, 2, 0)

    def read_band(self, band: int) -> np.ndarray:
        if self.interleave == "bil":
            return np.asarray(self.mm[:, band, :])
        if self.interleave == "bip":
            return np.asarray(self.mm[:, :, band])
        return np.asarray(self.mm[band, :, :])


def mean_envi_reference(cube: EnviCube) -> np.ndarray:
    total = np.zeros((cube.samples, cube.bands), dtype=np.float64)
    for y0 in range(0, cube.lines, 128):
        y1 = min(cube.lines, y0 + 128)
        total += np.sum(cube.read_lines(y0, y1), axis=0, dtype=np.float64)
    return total / max(1, cube.lines)


def load_exporter_style_front(raw_dir: Path, cube_shape: tuple[int, int, int],
                              args) -> np.ndarray:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw SWIR folder not found: {raw_dir}")
    headers = sorted(raw_dir.rglob("*.hdr")) + sorted(raw_dir.rglob("*.HDR"))
    main_hdr = [p for p in headers if "darkref" not in p.name.lower() and "whiteref" not in p.name.lower()]
    white_hdr = [p for p in headers if "whiteref" in p.name.lower()]
    dark_hdr = [p for p in headers if "darkref" in p.name.lower()]
    if len(main_hdr) != 1 or len(white_hdr) != 1 or len(dark_hdr) != 1:
        raise RuntimeError(f"Expected one main, WHITEREF, DARKREF .hdr in {raw_dir}; got {len(main_hdr)}, {len(white_hdr)}, {len(dark_hdr)}")
    main, white, dark = [EnviCube(p) for p in (main_hdr[0], white_hdr[0], dark_hdr[0])]
    H, W, B = cube_shape
    if (main.lines, main.samples, main.bands) != (H, W, B):
        raise ValueError(f"Raw and processed SWIR cubes have different dimensions: "
                         f"raw={(main.lines, main.samples, main.bands)} vs processed={cube_shape}. "
                         "Cannot align the spectral faces to the rock image.")
    for name, ref in (("white", white), ("dark", dark)):
        if (ref.samples, ref.bands) != (main.samples, main.bands):
            raise ValueError(f"{name} reference does not match main cube")

    if args.front_band_index is not None:
        band = args.front_band_index
    elif args.front_gray_nm is not None or args.front_band_nm is not None:
        target = args.front_gray_nm if args.front_gray_nm is not None else (args.front_band_nm or 1500.)
        if not np.all(np.diff(main.wavelengths) > 0):
            raise ValueError("Wavelength selection requested, but raw ENVI wavelengths are unavailable")
        band = int(np.argmin(np.abs(main.wavelengths - target)))
    else:
        band = main.bands // 2
    if band < 0 or band >= main.bands:
        raise ValueError(f"--front-band-index must be within 0..{main.bands - 1}")

    print(f"Reading raw ENVI band {band} ({main.wavelengths[band]:.2f} nm); calibrating with white/dark references...")
    white_mean = mean_envi_reference(white)
    dark_mean = mean_envi_reference(dark)
    raw = main.read_band(band).astype(np.float32)
    wref = white_mean[:, band].astype(np.float32)[None, :]
    dref = dark_mean[:, band].astype(np.float32)[None, :]
    denom = np.where(np.abs(wref - dref) > 1e-12, wref - dref, np.nan)
    reflectance = (raw - dref) / denom
    reflectance = np.nan_to_num(reflectance, nan=0.0, posinf=1.0, neginf=0.0)
    gamma = args.front_gamma if args.front_gamma is not None else args.export_gamma
    brightness = args.front_brightness if args.front_brightness is not None else 1.0
    finite = np.isfinite(reflectance)
    if not np.any(finite):
        gray = np.zeros(reflectance.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(reflectance[finite],
                               [args.front_low_pct, args.front_high_pct])
        if hi <= lo:
            lo, hi = np.min(reflectance[finite]), np.max(reflectance[finite])
        if hi <= lo:
            gray = np.zeros(reflectance.shape, dtype=np.uint8)
        else:
            display = np.clip((reflectance - lo) / (hi - lo), 0., 1.)
            display = np.clip(np.power(display, gamma) * brightness, 0., 1.)
            gray = np.round(display * 255).astype(np.uint8)
    print(f"Front face: raw-calibrated display, gamma={gamma:g}, "
          f"brightness={brightness:g}, full-image "
          f"{args.front_low_pct:g}/{args.front_high_pct:g} percentiles")
    return np.repeat(gray[:, :, None], 3, axis=2)


def load_front_override(args, cube_shape) -> np.ndarray | None:
    if args.front_source == "processed":
        return None
    if args.front_source == "raw-calibrated":
        root = args.raw_dir or (args.project_root / "input_raw")
        if not root.is_dir() and args.raw_dir is None:
            root = args.project_root / "Input_raw"
        return load_exporter_style_front(root / args.dataset, cube_shape, args)
    if args.front_source == "png":
        if args.front_png is None:
            raise ValueError("--front-source png requires --front-png PATH")
        with Image.open(args.front_png) as im:
            rgb = np.asarray(im.convert("RGB"))
        if rgb.shape[:2] != cube_shape[:2]:
            raise ValueError(f"PNG shape {rgb.shape[:2]} does not match processed cube shape {cube_shape[:2]}")
        print(f"Front face: using exactly the exported PNG pixels from {args.front_png}")
        return rgb
    raise ValueError(f"Unknown front source: {args.front_source}")


def robust_limits(values: np.ndarray, lo: float = 2, hi: float = 98) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    low, high = np.percentile(finite, [lo, hi])
    if high <= low:
        return (float(low), float(low + 1.0))
    return (float(low), float(high))


def rescale(values: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    low, high = limits
    return np.clip(np.nan_to_num((values - low) / (high - low), nan=0.0,
                                 posinf=1.0, neginf=0.0), 0.0, 1.0)


def enhanced_intensity(values: np.ndarray, args) -> np.ndarray:
    limits = robust_limits(values, args.front_low_pct, args.front_high_pct)
    normalized = rescale(values, limits)
    gamma = args.front_gamma if args.front_gamma is not None else 0.72
    brightness = args.front_brightness if args.front_brightness is not None else 1.08
    return np.clip(np.power(normalized, gamma) * brightness, 0, 1)


def sampled_positions(start: int, stop: int, max_samples: int) -> np.ndarray:
    if max_samples < 2:
        raise ValueError("Sample limits must be >= 2")
    count = min(stop - start, max_samples)
    return np.unique(np.linspace(start, stop - 1, count).round().astype(int))


def spectral_rgb_band_indices(wavelengths_nm: np.ndarray, dataset_name: str, explicit=None):
    if explicit is not None:
        targets = explicit
    elif dataset_name.upper().startswith("VNIR"):
        targets = (650.0, 550.0, 450.0)
    else:
        targets = (2200.0, 1650.0, 1200.0)
    return [int(np.argmin(np.abs(wavelengths_nm - t))) for t in targets]


def preview_image(cube: np.ndarray, wavelengths_nm: np.ndarray, args, front_override=None):
    height, width, _ = cube.shape
    yy = sampled_positions(0, height, args.preview_max_y)
    xx = sampled_positions(0, width, args.preview_max_x)
    if front_override is not None:
        result = front_override[np.ix_(yy, xx)].astype(np.float32) / 255.0
        return np.flip(result, axis=1).copy() if args.flip_x else result
    mode = args.front_mode
    if mode == "auto":
        mode = "false-color" if args.dataset.upper().startswith("VNIR") else "gray"
    if args.front_gray_nm is not None:
        mode = "gray"

    if mode == "false-color":
        indices = spectral_rgb_band_indices(wavelengths_nm, args.dataset, args.rgb_wavelengths)
        channels = []
        for idx in indices:
            band = np.asarray(cube[:, :, idx][np.ix_(yy, xx)], dtype=np.float32)
            channels.append(enhanced_intensity(band, args))
        result = np.stack(channels, axis=-1)
    else:
        target = args.front_gray_nm if args.front_gray_nm is not None else (args.front_band_nm or 1500.)
        idx = int(np.argmin(np.abs(wavelengths_nm - target)))
        band = np.asarray(cube[:, :, idx][np.ix_(yy, xx)], dtype=np.float32)
        intensity = enhanced_intensity(band, args)
        if mode == "warm":
            result = np.clip(intensity[..., None] * np.array([1., .90, .78]) +
                             (1.-intensity[..., None]) * np.array([.025, .018, .012]), 0, 1)
        else:
            result = intensity
    return np.flip(result, axis=1).copy() if args.flip_x else result


def check_gui_backend():
    backend = matplotlib.get_backend().lower()
    if backend in ("agg", "pdf", "svg", "ps", "cairo", "template") or "inline" in backend:
        raise RuntimeError(
            "Interactive selection needs a desktop display with a Matplotlib GUI backend. "
            "Run the command in your Linux graphical desktop/RDP session. "
            "If necessary install python3-tk or PyQt5 and set MPLBACKEND=TkAgg or QtAgg."
        )


def select_crop_interactively(cube: np.ndarray, wavelengths_nm: np.ndarray, args, front_override=None) -> tuple[int, int, int, int]:
    check_gui_backend()
    H, W, _ = cube.shape
    overview = preview_image(cube, wavelengths_nm, args, front_override=front_override)
    # Display-only preview compression; pixel coordinates remain in the source frame.
    fig_h = args.preview_window_height
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.subplots_adjust(bottom=.13, left=.08, right=.96, top=.90)
    ax.imshow(overview, origin="upper", extent=(0, W, H, 0),
              interpolation="nearest", aspect="auto", cmap="gray", vmin=0, vmax=1)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xlabel("X (original pixels)")
    ax.set_ylabel("Y (original pixels)")
    ax.set_title(f"{args.dataset} | Drag to select an area, then press Enter")
    fig.text(.5, .055, "Drag: select     Enter: confirm     R: reset     Esc / Q: cancel",
             ha="center", va="center", fontsize=20)

    state = {"bounds": None, "confirmed": False}
    outline = Rectangle((0, 0), 1, 1, fill=False, lw=1.8, ec="yellow", visible=False)
    ax.add_patch(outline)

    def on_select(first, last):
        if first.xdata is None or last.xdata is None or first.ydata is None or last.ydata is None:
            return
        left = int(np.clip(np.floor(min(first.xdata, last.xdata)), 0, W - 1))
        right = int(np.clip(np.ceil(max(first.xdata, last.xdata)), 1, W))
        top = int(np.clip(np.floor(min(first.ydata, last.ydata)), 0, H - 1))
        bottom = int(np.clip(np.ceil(max(first.ydata, last.ydata)), 1, H))
        if (right - left) < 2 or (bottom - top) < 2:
            print("Selection too small; drag a larger rectangle.")
            return
        state["bounds"] = (top, bottom, left, right)
        outline.set_bounds(left, top, right - left, bottom - top)
        outline.set_visible(True)
        ax.set_title(f"Selected: x={left}:{right}, y={top}:{bottom}  |  Enter to confirm")
        fig.canvas.draw_idle()

    selector = RectangleSelector(
        ax, on_select, useblit=True, button=[1],
        minspanx=4, minspany=4, spancoords="data", interactive=True,
    )

    def on_key(event):
        key = (event.key or "").lower()
        if key in ("enter", "return"):
            if state["bounds"] is None:
                print("No area selected yet.")
                return
            state["confirmed"] = True
            plt.close(fig)
        elif key == "r":
            selector.set_active(False)
            selector.set_active(True)
            state["bounds"] = None
            outline.set_visible(False)
            ax.set_title(f"{args.dataset} | Drag to select an area, then press Enter")
            fig.canvas.draw_idle()
        elif key in ("escape", "q"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    if not state["confirmed"] or state["bounds"] is None:
        raise RuntimeError("Crop selection cancelled.")

    top, bottom, left, right = state["bounds"]
    if args.flip_x:
        left, right = W - right, W - left
    return top, bottom, left, right


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Four points are required")
    by_y = pts[np.argsort(pts[:, 1])]
    top = by_y[:2][np.argsort(by_y[:2, 0])]
    bottom = by_y[2:][np.argsort(by_y[2:, 0])]
    ordered = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
    if cv2.contourArea(ordered) < 4 or not cv2.isContourConvex(ordered):
        raise ValueError("Invalid selection: choose the four distinct corners around the core")
    return ordered



def select_quad_interactively(cube: np.ndarray, wavelengths_nm: np.ndarray, args, front_override=None) -> np.ndarray:
    check_gui_backend()
    H, W, _ = cube.shape
    overview = preview_image(cube, wavelengths_nm, args, front_override=front_override)
    # Display-only preview compression; pixel coordinates remain in the source frame.
    fig_h = args.preview_window_height
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.subplots_adjust(bottom=.13, left=.08, right=.96, top=.90)
    ax.imshow(overview, origin="upper", extent=(0, W, H, 0),
              interpolation="nearest", aspect="auto", cmap="gray", vmin=0, vmax=1)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xlabel("X (original pixels)")
    ax.set_ylabel("Y (original pixels)")
    ax.set_title(f"{args.dataset} | Click 4 points around the ROI, then press Enter")
    fig.text(.5, .055,
             "Click order can be arbitrary. Enter: confirm   Backspace: remove last point   R: reset   Esc/Q: cancel",
             ha="center", va="center", fontsize=20)

    state = {"points": [], "confirmed": False}
    markers = []
    poly = Polygon(np.zeros((0, 2)), closed=True, fill=False, lw=2.0, ec="yellow", visible=False)
    ax.add_patch(poly)

    def redraw():
        if len(state["points"]) >= 2:
            poly.set_xy(np.asarray(state["points"]))
            poly.set_visible(True)
        else:
            poly.set_visible(False)
        ax.set_title(f"{args.dataset} | Selected {len(state['points'])}/4 points")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        if len(state["points"]) >= 4:
            print("Already have 4 points. Press Enter, or Backspace/R to edit.")
            return
        p = [float(np.clip(event.xdata, 0, W - 1)), float(np.clip(event.ydata, 0, H - 1))]
        state["points"].append(p)
        m = ax.plot(p[0], p[1], marker='o', markersize=6, color='yellow', mec='black', mew=.6)[0]
        label = ax.text(p[0] + W * 0.008, p[1], str(len(state["points"])), color='yellow', fontsize=20,
                        path_effects=[patheffects.withStroke(linewidth=2.0, foreground='black')])
        markers.append((m, label))
        redraw()

    def on_key(event):
        key = (event.key or "").lower()
        if key in ("enter", "return"):
            if len(state["points"]) != 4:
                print("Please click exactly 4 points first.")
                return
            state["confirmed"] = True
            plt.close(fig)
        elif key in ("backspace", "delete"):
            if state["points"]:
                state["points"].pop()
                artist, label = markers.pop()
                artist.remove()
                label.remove()
                redraw()
        elif key == "r":
            state["points"].clear()
            while markers:
                artist, label = markers.pop()
                artist.remove()
                label.remove()
            redraw()
        elif key in ("escape", "q"):
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    if not state["confirmed"] or len(state["points"]) != 4:
        raise RuntimeError("4-point selection cancelled.")

    pts = np.asarray(state["points"], dtype=np.float32)
    if args.flip_x:
        pts[:, 0] = (W - 1) - pts[:, 0]
    return order_quad_points(pts)


def get_crop_bounds(shape, args):
    H, W, _ = shape
    if args.full_image or (args.height is None and args.width is None and args.x_start is None and args.y_start is None):
        return 0, H, 0, W
    crop_h = H if args.height is None else min(args.height, H)
    crop_w = W if args.width is None else min(args.width, W)
    y0 = 0 if args.y_start is None else int(np.clip(args.y_start, 0, max(0, H - crop_h)))
    x0 = 0 if args.x_start is None else int(np.clip(args.x_start, 0, max(0, W - crop_w)))
    return y0, y0 + crop_h, x0, x0 + crop_w


def make_front_image_from_cube(cube_small: np.ndarray, wavelengths_nm: np.ndarray, args) -> np.ndarray:
    mode = args.front_mode
    if mode == "auto":
        mode = "false-color" if args.dataset.upper().startswith("VNIR") else "gray"
    if args.front_gray_nm is not None:
        mode = "gray"
    target = args.front_gray_nm if args.front_gray_nm is not None else (args.front_band_nm or 1500.)

    if mode in ("gray", "warm"):
        band = int(np.argmin(np.abs(wavelengths_nm - target)))
        channel = np.asarray(cube_small[:, :, band], dtype=np.float32)
        intensity = enhanced_intensity(channel, args)
        if mode == "gray":
            front = np.repeat(intensity[..., None], 3, axis=2)
        else:
            neutral = intensity[..., None]
            tint = np.array([1.0, .90, .78], dtype=np.float32)
            front = np.clip(neutral * tint[None, None, :] +
                            (1.0 - neutral) * np.array([.025, .018, .012]), 0., 1.)
        print(f"Front face: {mode} display at {wavelengths_nm[band]:.1f} nm (visualization only)")
        return front

    if mode == "false-color":
        selected = spectral_rgb_band_indices(wavelengths_nm, args.dataset, args.rgb_wavelengths)
        rgb = []
        for band in selected:
            channel = np.asarray(cube_small[:, :, band], dtype=np.float32)
            rgb.append(enhanced_intensity(channel, args))
        front = np.stack(rgb, axis=-1)
        print("Front face: three-band RGB composite (false-color for SWIR)")
        return front

    raise ValueError(f"Unknown front mode: {mode}")


def colorize_sides(top: np.ndarray, right: np.ndarray, cmap_name: str):
    cmap = plt.get_cmap(cmap_name)
    lims = robust_limits(np.concatenate([top.ravel(), right.ravel()]), 1, 99)
    top_rgba = cmap(rescale(top, lims))
    right_rgba = cmap(rescale(right, lims))
    return top_rgba, right_rgba


def maybe_flip_views(front: np.ndarray, top: np.ndarray, right: np.ndarray, args):
    if not args.flip_x:
        return front, top, right
    return np.flip(front, axis=1).copy(), np.flip(top, axis=1).copy(), right


def load_small_views_standard(cube: np.ndarray, wavelengths_nm: np.ndarray, args, interactive_bounds=None, front_override=None):
    y0, y1, x0, x1 = (interactive_bounds if interactive_bounds is not None else get_crop_bounds(cube.shape, args))
    y_front = sampled_positions(y0, y1, args.max_front_y)
    x_front = sampled_positions(x0, x1, args.max_front_x)
    y_side = sampled_positions(y0, y1, args.max_side_y)
    x_side = sampled_positions(x0, x1, args.max_side_x)
    b_side = sampled_positions(0, cube.shape[2], args.max_bands)

    if front_override is not None:
        front = front_override[np.ix_(y_front, x_front)].astype(np.float32) / 255.0
    else:
        mode = args.front_mode
        if mode == "auto":
            mode = "false-color" if args.dataset.upper().startswith("VNIR") else "gray"
        if args.front_gray_nm is not None:
            mode = "gray"
        target = args.front_gray_nm if args.front_gray_nm is not None else (args.front_band_nm or 1500.)

        if mode in ("gray", "warm"):
            band = int(np.argmin(np.abs(wavelengths_nm - target)))
            channel = np.asarray(cube[:, :, band][np.ix_(y_front, x_front)], dtype=np.float32)
            intensity = enhanced_intensity(channel, args)
            if mode == "gray":
                front = np.repeat(intensity[..., None], 3, axis=2)
            else:
                neutral = intensity[..., None]
                tint = np.array([1.0, .90, .78], dtype=np.float32)
                front = np.clip(neutral * tint[None, None, :] +
                                (1.0 - neutral) * np.array([.025, .018, .012]), 0., 1.)
        else:
            selected = spectral_rgb_band_indices(wavelengths_nm, args.dataset, args.rgb_wavelengths)
            rgb = []
            for band in selected:
                channel = np.asarray(cube[:, :, band][np.ix_(y_front, x_front)], dtype=np.float32)
                rgb.append(enhanced_intensity(channel, args))
            front = np.stack(rgb, axis=-1)

    top = np.asarray(cube[y0, :, :][np.ix_(x_side, b_side)].T, dtype=np.float32)
    right_x = x0 if args.flip_x else x1 - 1
    right = np.asarray(cube[:, right_x, :][np.ix_(y_side, b_side)], dtype=np.float32)
    front, top, right = maybe_flip_views(front, top, right, args)

    print(f"Data shape: {cube.shape}; crop x={x0}:{x1}, y={y0}:{y1}")
    print(f"Front resolution: {front.shape[1]} x {front.shape[0]}")
    print(f"Spectral surface: {len(b_side)} sampled of {cube.shape[2]} bands")
    return front, top, right, wavelengths_nm[b_side], (y0, y1, x0, x1)


def rectify_quad_subcube(cube: np.ndarray, wavelengths_nm: np.ndarray, args, quad_points: np.ndarray, front_override=None):
    quad = order_quad_points(quad_points)
    x_coords = quad[:, 0]
    y_coords = quad[:, 1]
    x0 = max(0, int(np.floor(x_coords.min())) - 1)
    x1 = min(cube.shape[1], int(np.ceil(x_coords.max())) + 2)
    y0 = max(0, int(np.floor(y_coords.min())) - 1)
    y1 = min(cube.shape[0], int(np.ceil(y_coords.max())) + 2)

    local_quad = quad.copy()
    local_quad[:, 0] -= x0
    local_quad[:, 1] -= y0

    width_top = np.linalg.norm(quad[1] - quad[0])
    width_bottom = np.linalg.norm(quad[2] - quad[3])
    height_left = np.linalg.norm(quad[3] - quad[0])
    height_right = np.linalg.norm(quad[2] - quad[1])
    width_px = max(2, int(round(max(width_top, width_bottom))))
    height_px = max(2, int(round(max(height_left, height_right))))

    # The older VNIR display sampled height and width independently. It is
    # available only as an explicit visual option; SWIR retains uniform scaling.
    if args.vnir_legacy_sampling and args.dataset.upper().startswith("VNIR"):
        out_w = min(width_px, args.max_front_x)
        out_h = min(height_px, args.max_front_y)
        scale = None
        print(f"VNIR legacy display sampling: {width_px} x {height_px} -> "
              f"{out_w} x {out_h}. This may distort pore shapes.")
    else:
        scale = min(1., args.max_front_x / width_px, args.max_front_y / height_px)
        out_w = max(2, int(round(width_px * scale)))
        out_h = max(2, int(round(height_px * scale)))
        print(f"Selected ROI: {width_px} x {height_px} pixels. "
              f"Rectified output: {out_w} x {out_h} pixels (uniform scale {scale:.4f}).")

    b_side = sampled_positions(0, cube.shape[2], args.max_bands)

    mode = args.front_mode
    if mode == "auto":
        mode = "false-color" if args.dataset.upper().startswith("VNIR") else "gray"
    if args.front_gray_nm is not None:
        mode = "gray"

    needed = set(int(i) for i in b_side)
    target = args.front_gray_nm if args.front_gray_nm is not None else (args.front_band_nm or 1500.)
    if front_override is None:
        if mode in ("gray", "warm"):
            needed.add(int(np.argmin(np.abs(wavelengths_nm - target))))
        elif mode == "false-color":
            needed.update(spectral_rgb_band_indices(wavelengths_nm, args.dataset, args.rgb_wavelengths))

    needed = np.array(sorted(needed), dtype=int)

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)
    if args.rectify_method == "perspective":
        transform = cv2.getPerspectiveTransform(local_quad.astype(np.float32), dst)
        def warp(image):
            return cv2.warpPerspective(image, transform, (out_w, out_h),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        top_center = .5 * (local_quad[0] + local_quad[1])
        bottom_center = .5 * (local_quad[2] + local_quad[3])
        direction = bottom_center - top_center
        rotation_deg = -np.degrees(np.arctan2(float(direction[0]), float(direction[1])))
        center = np.mean(local_quad, axis=0)
        transform = cv2.getRotationMatrix2D(tuple(center.astype(float)), float(rotation_deg), float(scale))
        transform[0, 2] += (out_w - 1) * .5 - center[0]
        transform[1, 2] += (out_h - 1) * .5 - center[1]
        def warp(image):
            return cv2.warpAffine(image, transform, (out_w, out_h),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Warp one band at a time to avoid loading an entire hyperspectral cube.
    rectified = np.empty((out_h, out_w, needed.size), dtype=np.float32)
    for j, band in enumerate(needed):
        source_band = np.asarray(cube[y0:y1, x0:x1, int(band)], dtype=np.float32)
        rectified[:, :, j] = warp(source_band)

    if front_override is not None:
        front_crop = np.asarray(front_override[y0:y1, x0:x1], dtype=np.uint8)
        front = warp(front_crop)
        front = front.astype(np.float32) / 255.0
    else:
        wav_rect = wavelengths_nm[needed]
        front = make_front_image_from_cube(rectified, wav_rect, args)

    # top face: top row of rectified cube, shape (bands, x)
    x_side = sampled_positions(0, out_w, min(args.max_side_x, out_w))
    y_side = sampled_positions(0, out_h, min(args.max_side_y, out_h))
    # map requested sampled bands to indices inside rectified 'needed' set
    local_b = np.array([np.where(needed == b)[0][0] for b in b_side], dtype=int)
    top = np.asarray(rectified[0, x_side[:, None], local_b[None, :]].T, dtype=np.float32)
    side_x = 0 if args.flip_x else out_w - 1
    right = np.asarray(rectified[y_side[:, None], side_x, local_b[None, :]], dtype=np.float32)
    right = right[:, :, 0] if right.ndim == 3 else right
    if right.ndim == 1:
        right = right[:, None]

    front, top, right = maybe_flip_views(front, top, right, args)
    print(f"Rectified ROI bbox: x={x0}:{x1}, y={y0}:{y1} | output size: {out_w} x {out_h}")
    # Source geometry defines the figure's aspect ratio; displayed sampling does not.
    return front, top, right, wavelengths_nm[b_side], (0, height_px, 0, width_px)


def render(front, top, right, sampled_wavelengths, args, bounds):
    physical_h = bounds[1] - bounds[0]
    physical_w = bounds[3] - bounds[2]
    w = 1.0
    natural_ratio = (physical_h * args.pixel_size_y_um) / max(physical_w * args.pixel_size_x_um, 1e-9)
    if args.face_aspect is not None:
        h = args.face_aspect
    elif args.legacy_aspect:
        h = max(.40, min(args.max_face_aspect, natural_ratio)) * args.display_stretch
    else:
        h = natural_ratio * args.display_stretch
    print(f"Figure aspect H/W = {h:.4f}; native = {natural_ratio:.4f}; "
          f"display stretch = {args.display_stretch:g}")
    if abs(h - natural_ratio) > 0.02:
        print("Note: display aspect differs from pixel aspect; apparent pore geometry is not to scale.")
    dx = args.depth
    dy = args.depth * args.rise

    front_bl = np.array([0., 0.])
    front_br = np.array([w, 0.])
    front_tl = np.array([0., h])
    front_tr = np.array([w, h])
    depth_offset = np.array([dx, dy])

    figure_h = max(6.7, min(18., 8.5 + 0.55 * max(h - 3., 0.)))
    fig = plt.figure(figsize=(7.5, figure_h), facecolor="white")
    ax = fig.add_axes([.08, .10, .84, .82])
    ax.set_aspect("equal")
    ax.set_axis_off()

    top_rgba, right_rgba = colorize_sides(top, right, args.cmap)

    transform_top = Affine2D.from_values(w, 0, dx, dy, 0, h) + ax.transData
    ax.imshow(top_rgba, origin="lower", extent=(0, 1, 0, 1), interpolation="bilinear",
              transform=transform_top, zorder=2)

    transform_right = Affine2D.from_values(dx, dy, 0, h, w, 0) + ax.transData
    ax.imshow(right_rgba, origin="upper", extent=(0, 1, 0, 1), interpolation="bilinear",
              transform=transform_right, zorder=3)

    if args.band_lines > 0:
        for t in np.linspace(0, 1, args.band_lines + 2)[1:-1]:
            c = (0.1, 0.14, 0.18, .24)
            ax.plot([w + dx*t, w + dx*t], [dy*t, h + dy*t], color=c, linewidth=.50, zorder=4)
            ax.plot([dx*t, w + dx*t], [h + dy*t, h + dy*t], color=c, linewidth=.5, zorder=4)

    ax.imshow(front, origin="upper", extent=(0, w, 0, h), interpolation="nearest", zorder=10)

    edge_color = "#27323a"
    for a, b in [
        (front_bl, front_br), (front_br, front_tr), (front_tr, front_tl), (front_tl, front_bl),
        (front_tl, front_tl+depth_offset),
        (front_tr, front_tr+depth_offset),
        (front_br, front_br+depth_offset),
        (front_tl+depth_offset, front_tr+depth_offset),
        (front_tr+depth_offset, front_br+depth_offset),
    ]:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=edge_color,
                linewidth=1.0 if tuple(a) in (tuple(front_tl), tuple(front_tr)) else .85,
                solid_capstyle="round", zorder=12)

    # label_font = 18
    # if not args.hide_labels:
    #     ax.text(w*.5, -.11*max(w, .5), "Spatial X", ha="center", va="top", fontsize=label_font, color=edge_color)
    #     ax.text(-.30, h*.50, "Spatial Y", ha="center", va="center", fontsize=label_font, color=edge_color, rotation=90)
    #     ax.annotate("", xy=tuple(front_tr+depth_offset*1.04), xytext=tuple(front_tr+depth_offset*.04),
    #                 arrowprops={"arrowstyle": "->", "lw": 1.25, "color": "#374b66"}, zorder=13)
    #     # ax.text(w + dx*.04, h + dy + .12, "Spectral Z (λ, nm)", va="bottom", ha="center",
    #     #         fontsize=label_font-1, color=edge_color, zorder=13)
    #     ax.text(w + dx*.04 - 0.6, h + dy + .3, "Spectral Z (nm)", va="bottom", ha="center",
    #             fontsize=label_font-1, color=edge_color, zorder=13)
    #     # Annotate wavelengths along the upper-left spectral edge.
    #     if args.z_ticks:
    #         for idx in np.linspace(0, len(sampled_wavelengths) - 1,
    #                                min(args.z_ticks, len(sampled_wavelengths))).round().astype(int):
    #             t = idx / max(len(sampled_wavelengths) - 1, 1)
    #             pos = front_tl + depth_offset * t
    #             ax.plot([pos[0] - .014, pos[0] + .014], [pos[1] + .009, pos[1] - .009],
    #                     color=edge_color, linewidth=.8, zorder=15)
    #             ax.text(pos[0] - .15, pos[1], f"{sampled_wavelengths[idx]:.0f}",
    #                     fontsize=18, ha="right", va="center", color=edge_color, zorder=16,
    #                     path_effects=[patheffects.withStroke(linewidth=1.6, foreground="white")])

    if args.scale_bar_mm is not None:
        width_mm = physical_w * args.pixel_size_x_um / 1000.
        bar_w = args.scale_bar_mm / max(width_mm, 1e-9) * w
        if bar_w <= .7 * w:
            xbar, ybar = .07 * w, .055 * h
            ax.plot([xbar, xbar + bar_w], [ybar, ybar], color="black", lw=3.5,
                    zorder=20, path_effects=[patheffects.withStroke(linewidth=5, foreground="white")])
            ax.text(xbar, ybar + .018 * h, f"{args.scale_bar_mm:g} mm", color="white", fontsize=8.5,
                    va="bottom", zorder=21,
                    path_effects=[patheffects.withStroke(linewidth=1.8, foreground="black")])
        else:
            print("Scale bar not drawn: requested bar is wider than 70% of ROI")

    ax.set_xlim(-.30, w + dx + .40)
    ax.set_ylim(-.22, h + dy + .32)

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = args.output_name or f"{args.dataset.lower()}_hsi_cube_front"
    path = output / f"{stem}.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", pad_inches=.16, facecolor="white")
    plt.close(fig)
    return [path]


def parse_args():
    parser = argparse.ArgumentParser(description="HSI 3D cube: sediment on FRONT face, wavelength in Z depth")
    parser.add_argument("--dataset", default="SWIR5", help="e.g. SWIR5 or VNIR5")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--processed-dir", type=Path, default=None, help="Override root of processed_data")
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--y-start", type=int, default=None)
    parser.add_argument("--x-start", type=int, default=None)
    parser.add_argument("--height", type=int, default=None, help="Crop height; no crop unless a crop argument is given")
    parser.add_argument("--width", type=int, default=None, help="Crop width; no crop unless a crop argument is given")
    parser.add_argument("--full-image", action="store_true", help="Show the entire image (this is already the default)")
    parser.add_argument("--select-crop", action="store_true", help="Open a 2-D image and let you draw a rectangle")
    parser.add_argument("--select-quad", action="store_true", help="Open a 2-D image and let you click 4 points for rectification")
    parser.add_argument("--legacy-aspect", action="store_true",
                        help="Allow the old --max-face-aspect cap (may deform geological structures).")
    parser.add_argument("--pixel-size-x-um", type=float, default=1.0,
                        help="Calibrated X pixel spacing, when known. Default assumes square pixels.")
    parser.add_argument("--pixel-size-y-um", type=float, default=1.0,
                        help="Calibrated Y pixel spacing, when known. Default assumes square pixels.")
    parser.add_argument("--rectify-method", choices=("perspective", "rigid"), default="perspective",
                        help="Default: original perspective rectification. 'rigid' is optional rotation-only.")
    parser.add_argument("--preserve-aspect", action="store_true",
                        help="Compatibility flag: true spatial aspect is already preserved by default.")
    parser.add_argument("--face-aspect", type=float, default=None,
                        help="Optional fixed final figure height/width; overrides display stretch.")
    parser.add_argument("--display-stretch", type=float, default=1.5,
                        help="Final figure vertical stretch (1.5 default; 1.0 native pixel aspect)")
    parser.add_argument("--scale-bar-mm", type=float, default=None,
                        help="Optional physical scale bar in mm; requires calibrated pixel sizes.")
    parser.add_argument("--preview-window-height", type=float, default=6.5,
                        help="Compact selection window height in inches; does not affect output")
    parser.add_argument("--preview-max-y", type=int, default=1400, help="Number of sampled rows in the crop-selection preview")
    parser.add_argument("--preview-max-x", type=int, default=800, help="Number of sampled columns in the crop-selection preview")
    parser.add_argument("--front-gamma", type=float, default=None,
                        help="Front gamma override. Default: 0.35 for raw-calibrated, 0.72 for processed")
    parser.add_argument("--front-brightness", type=float, default=None,
                        help="Front brightness override. Default: 1.0 for raw-calibrated, 1.08 for processed")
    parser.add_argument("--front-low-pct", type=float, default=1., help="Lower display contrast percentile")
    parser.add_argument("--front-high-pct", type=float, default=99., help="Upper display contrast percentile")
    parser.add_argument("--max-front-y", type=int, default=1800)
    parser.add_argument("--max-front-x", type=int, default=800)
    parser.add_argument("--vnir-legacy-sampling", action="store_true",
                        help="Optional earlier VNIR quad-sampling behavior; may alter apparent shapes. No effect on SWIR.")
    parser.add_argument("--max-side-y", type=int, default=240)
    parser.add_argument("--max-side-x", type=int, default=160)
    parser.add_argument("--max-bands", type=int, default=120)
    parser.add_argument("--front-source", choices=("processed", "raw-calibrated", "png"),
                        default="processed",
                        help="Front image: TPSSR processed cube (default), original ENVI calibrated like exporter, or an existing exported PNG")
    parser.add_argument("--front-png", type=Path, default=None,
                        help="Exact existing exported PNG to use with --front-source png")
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Override input_raw root (parent containing SWIR6, SWIR7, etc.)")
    flip = parser.add_mutually_exclusive_group()
    flip.add_argument("--flip-x", dest="flip_x", action="store_true", default=None,
                      help="Mirror preview and final figure in X; default enabled for VNIR.")
    flip.add_argument("--no-flip-x", dest="flip_x", action="store_false",
                      help="Disable mirroring in X.")
    parser.add_argument("--front-band-index", type=int, default=None,
                        help="Raw-calibrated front band index; default is exact middle band")
    parser.add_argument("--export-gamma", type=float, default=0.35,
                        help="Gamma for --front-source raw-calibrated (exporter's default: 0.35)")
    parser.add_argument("--front-mode", choices=("auto", "gray", "warm", "false-color"), default="auto",
                        help="Front image display. Default: gray for SWIR; RGB for VNIR")
    parser.add_argument("--front-band-nm", type=float, default=None, help="Explicit wavelength; processed default 1500 nm, raw-calibrated default middle band")
    parser.add_argument("--rgb-wavelengths", nargs=3, type=float, metavar=("R", "G", "B"), help="Wavelengths for 3-band composite")
    parser.add_argument("--front-gray-nm", type=float, default=None, help="Instead of RGB, show one grayscale reflectance band on the front")
    parser.add_argument("--max-face-aspect", type=float, default=3.2, help="Maximum displayed height/width ratio of front face")
    parser.add_argument("--depth", type=float, default=.43, help="Projected depth relative to front-face width")
    parser.add_argument("--rise", type=float, default=.68, help="Depth-line slope")
    parser.add_argument("--cmap", type=str, default="cividis")
    parser.add_argument("--band-lines", type=int, default=11)
    parser.add_argument("--z-ticks", type=int, default=2,
                        help="Number of wavelength labels on upper-left spectral edge")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--hide-labels", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.flip_x is None:
        args.flip_x = args.dataset.upper().startswith("VNIR")
    print(f"X flip: {'ON' if args.flip_x else 'OFF'}")
    if args.max_front_x < 2 or args.max_front_y < 2:
        raise ValueError("Display sample limits must be >= 2")
    if args.pixel_size_x_um <= 0 or args.pixel_size_y_um <= 0:
        raise ValueError("Pixel spacing must be positive")
    if args.face_aspect is not None and args.face_aspect <= 0:
        raise ValueError("--face-aspect must be positive")
    if not np.isfinite(args.display_stretch) or args.display_stretch <= 0:
        raise ValueError("--display-stretch must be positive and finite")
    if not np.isfinite(args.preview_window_height) or args.preview_window_height < 3:
        raise ValueError("--preview-window-height must be at least 3 inches")
    if args.scale_bar_mm is not None:
        if args.scale_bar_mm <= 0:
            raise ValueError("--scale-bar-mm must be positive")
        if args.pixel_size_x_um == 1. and args.pixel_size_y_um == 1.:
            raise ValueError("Scale bar needs measured --pixel-size-x-um and --pixel-size-y-um")
    if args.z_ticks < 0:
        raise ValueError("--z-ticks must be >= 0")
    if args.select_crop and args.select_quad:
        raise ValueError("Use only one of --select-crop or --select-quad")
    if args.depth <= 0 or args.rise <= 0 or args.max_face_aspect <= 0:
        raise ValueError("--depth, --rise and --max-face-aspect must be positive")
    if ((args.front_gamma is not None and
         (not np.isfinite(args.front_gamma) or args.front_gamma <= 0)) or
        (args.front_brightness is not None and
         (not np.isfinite(args.front_brightness) or args.front_brightness <= 0)) or
        not (0 <= args.front_low_pct < args.front_high_pct <= 100)):
        raise ValueError("Invalid brightness/gamma or contrast percentiles")

    data_root = args.processed_dir or (args.project_root / "processed_data")
    folder = data_root / args.dataset
    cube_path, wav_path = folder / "calibrated_reflectance.npy", folder / "wavelengths_nm.npy"
    if not cube_path.is_file() or not wav_path.is_file():
        raise FileNotFoundError(
            f"Expected {cube_path} and {wav_path}. Run data preparation first or set --processed-dir."
        )
    cube = np.load(cube_path, mmap_mode="r")
    wavelengths = np.asarray(np.load(wav_path), dtype=np.float64).ravel()
    if cube.ndim != 3 or cube.shape[2] != wavelengths.size:
        raise ValueError(f"Unexpected cube shape {cube.shape} for {wavelengths.size} wavelengths; expected H x W x B")
    if float(wavelengths.max()) < 100:
        wavelengths = wavelengths * 1000.0

    if args.export_gamma <= 0:
        raise ValueError("--export-gamma must be positive")
    front_override = load_front_override(args, cube.shape)

    if args.select_quad:
        quad = select_quad_interactively(cube, wavelengths, args, front_override=front_override)
        front, top, right, wav_ds, bounds = rectify_quad_subcube(
            cube, wavelengths, args, quad, front_override=front_override)
    else:
        interactive_bounds = (select_crop_interactively(cube, wavelengths, args, front_override=front_override)
                              if args.select_crop else None)
        front, top, right, wav_ds, bounds = load_small_views_standard(
            cube, wavelengths, args, interactive_bounds, front_override=front_override)

    for path in render(front, top, right, wav_ds, args, bounds):
        print("Saved:", path.resolve())


if __name__ == "__main__":
    main()
