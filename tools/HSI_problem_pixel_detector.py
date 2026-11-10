#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
import tifffile


# ============================================================
# USER SETTINGS
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent
INPUT_ROOT = PROJECT_DIR / "input_raw"

PROCESS_ALL_SWIR_DATASETS = False
PRIMARY_DATASET = "SWIR8"  

AUTO_COMPARE_ALL_SWIR_FOLDERS = True
COMPARISON_DATASETS: list[str] = []

# Cube IO
CHUNK_LINES = 128

CLIPPING_CODE_OVERRIDE: int | None = None
MIN_REPEATED_TOP_CODE_COUNT = 20
MIN_REPEATED_TOP_CODE_FRACTION = 1e-8

USE_ENVI_DATA_IGNORE_VALUE_AS_CONFIRMED_MISSING = False

EXTRA_CONFIRMED_MISSING_VALUES: list[float] = []

STRONG_LINE_Y_FRACTION = 0.80
PARTIAL_LINE_MIN_Y_FRACTION = 0.30

MIN_RECURRENT_CLIPPED_STRONG_DATASETS = 4
MIN_RECURRENT_CLIPPED_PARTIAL_DATASETS = 4
REFERENCE_ROBUST_Z = 10.0

WIDESPREAD_BAND_X_FRACTION = 0.50

BAND_WIDE_SPATIAL_FRACTION = 0.95
WHOLE_SPECTRUM_BAND_FRACTION = 0.95
PARTIAL_LINE_LONGEST_RUN_FRACTION = 0.25
PATCH_COMPONENT_DENSITY = 0.50
PATCH_BAND_SUPPORT_FRACTION = 0.60
PATCH_SQUARENESS_MIN = 0.40
CONSECUTIVE_SPECTRAL_MIN_WIDTH = 3


OUTPUT_MASK_MODE = "all_detected"

WRITE_SELECTED_VOXEL_CSV = False
MAX_SELECTED_VOXEL_CSV_ROWS = 2_000_000

SAVE_PROBLEM_BAND_RGB_IMAGES = True
SAVE_ONLY_BANDS_WITH_PROBLEMS = True
RGB_OUTPUT_FOLDER_NAME = "problem_band_rgb_calibrated_side_by_side"
OVERLAY_HIGHLIGHT_RGB = (255, 0, 0)

DISPLAY_GAMMA = 0.35


COLLAPSE_NEARBY_VERTICAL_DUPLICATES = True
VERTICAL_DUPLICATE_MIN_COVERAGE = 0.30
VERTICAL_DUPLICATE_MIN_Y_OVERLAP = 0.80
VERTICAL_DUPLICATE_MAX_SEPARATION = 2

VERTICAL_DUPLICATE_ADJACENT_WEAK_RATIO = 0.60

SAVE_ALL_PROBLEM_BAND_OVERLAYS = False
WRITE_MASK_STACK_TIFF = False
WRITE_OVERLAY_STACK_TIFF = False
MASK_STACK_FOREGROUND_VALUE = 255


SCENARIOS = [
    "isolated_voxels",
    "single_detector_line",
    "multiple_detector_lines",
    "partial_vertical_lines",
    "spatial_patch",
    "consecutive_spectral_bands",
    "spatio_spectral_block",
    "whole_spectrum_pixels",
    "band_wide_saturation",
    "mixed_saturation",
]


SPATIAL_PATCH_TEMPLATES = [(5, 1), (9, 3), (15, 7), (25, 15), (41, 31)]
SPATIO_SPECTRAL_TEMPLATES = [(5, 5), (9, 11), (15, 21), (25, 41), (41, 63)]


ENVI_DTYPES = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    6: np.complex64,
    9: np.complex128,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


# ============================================================
# ENVI IO
# ============================================================
def parse_envi_header(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    result: dict[str, str] = {}
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line or line.upper() == "ENVI" or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip().lower()
        value = value.strip()

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


def parse_number_list(value: str | None) -> np.ndarray:
    if not value:
        return np.array([], dtype=np.float64)

    cleaned = value.replace("{", " ").replace("}", " ")
    tokens = re.split(r"[\s,]+", cleaned.strip())
    values: list[float] = []

    for token in tokens:
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue

    return np.asarray(values, dtype=np.float64)


def parse_single_number(value: str | None) -> float | None:
    values = parse_number_list(value)
    if values.size == 0:
        return None
    return float(values[0])


def header_int(header: dict[str, str], key: str) -> int:
    if key not in header:
        raise KeyError(f"Missing ENVI header field: {key}")
    return int(float(header[key].strip().strip("{}")))


def find_binary(hdr_path: Path) -> Path:
    base = hdr_path.with_suffix("")
    candidates = [
        base.with_suffix(".raw"),
        base.with_suffix(".RAW"),
        base.with_suffix(".dat"),
        base.with_suffix(".DAT"),
        base,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No RAW/DAT binary file found for:\n{hdr_path}")


def find_headers_in_folder(folder: Path) -> tuple[Path, Path, Path]:
    headers = sorted(folder.rglob("*.hdr")) + sorted(folder.rglob("*.HDR"))
    if not headers:
        raise FileNotFoundError(f"No ENVI .hdr files found under:\n{folder}")

    dark = [p for p in headers if "darkref" in p.name.lower()]
    white = [p for p in headers if "whiteref" in p.name.lower()]
    main = [
        p for p in headers
        if "darkref" not in p.name.lower()
        and "whiteref" not in p.name.lower()
    ]

    if len(main) != 1 or len(white) != 1 or len(dark) != 1:
        msg = [
            f"Could not uniquely identify main/WHITEREF/DARKREF in {folder}",
            "Headers found:",
        ]
        msg.extend(f"  {p}" for p in headers)
        raise RuntimeError("\n".join(msg))

    return main[0], white[0], dark[0]


class EnviCube:
    def __init__(self, hdr_path: Path):
        self.hdr_path = hdr_path
        self.binary_path = find_binary(hdr_path)
        self.header = parse_envi_header(hdr_path)

        self.lines = header_int(self.header, "lines")
        self.samples = header_int(self.header, "samples")
        self.bands = header_int(self.header, "bands")
        self.data_type_code = header_int(self.header, "data type")
        self.header_offset = int(
            float(self.header.get("header offset", "0").strip().strip("{}"))
        )
        self.byte_order = int(
            float(self.header.get("byte order", "0").strip().strip("{}"))
        )
        self.interleave = self.header.get("interleave", "bil").strip().lower().strip("{}")

        if self.data_type_code not in ENVI_DTYPES:
            raise ValueError(f"Unsupported ENVI data type: {self.data_type_code}")

        dtype = np.dtype(ENVI_DTYPES[self.data_type_code])
        self.dtype = dtype.newbyteorder("<" if self.byte_order == 0 else ">")

        if self.interleave == "bil":
            shape = (self.lines, self.bands, self.samples)
        elif self.interleave == "bip":
            shape = (self.lines, self.samples, self.bands)
        elif self.interleave == "bsq":
            shape = (self.bands, self.lines, self.samples)
        else:
            raise ValueError(f"Unsupported interleave: {self.interleave}")

        self.memmap = np.memmap(
            self.binary_path,
            dtype=self.dtype,
            mode="r",
            offset=self.header_offset,
            shape=shape,
        )

        wavelengths = parse_number_list(self.header.get("wavelength"))
        if wavelengths.size != self.bands:
            wavelengths = np.arange(self.bands, dtype=np.float64)
        self.wavelengths = wavelengths

        self.data_ignore_value = parse_single_number(self.header.get("data ignore value"))
        try:
            self.x_start = int(float(self.header.get("x start", "0").strip().strip("{}")))
        except ValueError:
            self.x_start = 0

    def read_lines(self, y0: int, y1: int) -> np.ndarray:
        # Returns (y, x, band)
        if self.interleave == "bil":
            return np.asarray(self.memmap[y0:y1]).transpose(0, 2, 1)
        if self.interleave == "bip":
            return np.asarray(self.memmap[y0:y1])
        return np.asarray(self.memmap[:, y0:y1, :]).transpose(1, 2, 0)

    def read_band(self, band: int) -> np.ndarray:
        # Returns (y, x)
        if self.interleave == "bil":
            return np.asarray(self.memmap[:, band, :])
        if self.interleave == "bip":
            return np.asarray(self.memmap[:, :, band])
        return np.asarray(self.memmap[band, :, :])


def validate_reference_compatibility(main: EnviCube, white: EnviCube, dark: EnviCube) -> None:
    for cube, label in [(white, "WHITEREF"), (dark, "DARKREF")]:
        if cube.samples != main.samples:
            raise ValueError(f"{label}: samples differ from main cube.")
        if cube.bands != main.bands:
            raise ValueError(f"{label}: bands differ from main cube.")


# ============================================================
# CLIPPING + REFERENCE DIAGNOSTICS
# ============================================================
def full_integer_histogram(cube: EnviCube) -> np.ndarray:
    if cube.dtype.kind not in "ui" or cube.dtype.itemsize > 2:
        raise ValueError(
            "Automatic clipping detection supports integer raw data up to 16 bits. "
            "Set CLIPPING_CODE_OVERRIDE manually otherwise."
        )

    dtype_max = int(np.iinfo(cube.dtype).max)
    histogram = np.zeros(dtype_max + 1, dtype=np.int64)
    total_chunks = math.ceil(cube.lines / CHUNK_LINES)

    for ci, y0 in enumerate(range(0, cube.lines, CHUNK_LINES), start=1):
        y1 = min(cube.lines, y0 + CHUNK_LINES)
        raw = cube.read_lines(y0, y1)
        histogram += np.bincount(
            raw.reshape(-1).astype(np.int64),
            minlength=dtype_max + 1,
        )
        if ci % 20 == 0 or ci == total_chunks:
            print(f"    histogram {ci}/{total_chunks}")

    return histogram


def infer_clipping_code(cube: EnviCube) -> tuple[int | None, str]:
    # Robust fallback: if the user copied an older settings block that does not
    # define CLIPPING_CODE_OVERRIDE, automatic clipping detection still works.
    clipping_code_override = globals().get("CLIPPING_CODE_OVERRIDE", None)
    if clipping_code_override is not None:
        return int(clipping_code_override), "manual override"

    if cube.dtype.kind not in "ui" or cube.dtype.itemsize > 2:
        return None, "automatic clipping code unavailable for this dtype"

    histogram = full_integer_histogram(cube)
    observed = np.flatnonzero(histogram)
    if observed.size == 0:
        return None, "empty dataset"

    observed_max = int(observed[-1])
    count = int(histogram[observed_max])
    total = int(histogram.sum())
    required = max(
        MIN_REPEATED_TOP_CODE_COUNT,
        math.ceil(total * MIN_REPEATED_TOP_CODE_FRACTION),
    )
    dtype_max = int(np.iinfo(cube.dtype).max)

    if observed_max == dtype_max and count >= required:
        return observed_max, "confirmed repeated storage maximum"
    if count >= required:
        return observed_max, "repeated observed maximum; treated only as a clipping candidate"
    return None, "no repeated fixed upper code detected"


def robust_z_across_x(values_x_band: np.ndarray) -> np.ndarray:
    med = np.median(values_x_band, axis=0, keepdims=True)
    mad = np.median(np.abs(values_x_band - med), axis=0, keepdims=True)
    scale = 1.4826 * mad
    std = np.std(values_x_band, axis=0, keepdims=True)
    scale = np.where(scale > 0, scale, std)
    scale = np.where(scale > 0, scale, 1.0)
    return np.abs(values_x_band - med) / scale


def reference_bad_detector_mask(white: EnviCube, dark: EnviCube) -> np.ndarray:
    dark_sum = np.zeros((dark.samples, dark.bands), dtype=np.float64)
    white_sum = np.zeros((white.samples, white.bands), dtype=np.float64)

    for y0 in range(0, dark.lines, CHUNK_LINES):
        y1 = min(dark.lines, y0 + CHUNK_LINES)
        dark_sum += np.sum(dark.read_lines(y0, y1), axis=0, dtype=np.float64)

    for y0 in range(0, white.lines, CHUNK_LINES):
        y1 = min(white.lines, y0 + CHUNK_LINES)
        white_sum += np.sum(white.read_lines(y0, y1), axis=0, dtype=np.float64)

    dark_mean = dark_sum / max(1, dark.lines)
    white_mean = white_sum / max(1, white.lines)
    gain = white_mean - dark_mean

    dark_z = robust_z_across_x(dark_mean)
    gain_z = robust_z_across_x(gain)

    return (
        (dark_z >= REFERENCE_ROBUST_Z)
        | (gain_z >= REFERENCE_ROBUST_Z)
        | (gain <= 0)
    )


def two_sided_vertical_residual(raw: np.ndarray) -> np.ndarray:
    """
    Robust vertical-stripe residual with NO immediate-neighbor halo.

    A detector sample is considered spatially inconsistent only when it differs
    from BOTH the left-side local baseline and the right-side local baseline.
    Therefore, if x=141 is a bright defective stripe, normal x=140 and x=142
    are not flagged merely because one of their neighbors is defective.

    Input:
        raw: (y, x, band) or (x, band)

    Output:
        residual score with the same shape.
    """
    arr = np.asarray(raw, dtype=np.float64)
    squeeze_y = False
    if arr.ndim == 2:
        arr = arr[None, ...]
        squeeze_y = True

    score = np.zeros(arr.shape, dtype=np.float64)
    xdim = arr.shape[1]

    if xdim >= 5:
        center = arr[:, 2:-2, :]
        left_baseline = 0.5 * (arr[:, :-4, :] + arr[:, 1:-3, :])
        right_baseline = 0.5 * (arr[:, 3:-1, :] + arr[:, 4:, :])

        left_diff = np.abs(center - left_baseline)
        right_diff = np.abs(center - right_baseline)

        # min(diff_left, diff_right) is high only if BOTH sides disagree.
        score[:, 2:-2, :] = np.minimum(left_diff, right_diff)

    elif xdim >= 3:
        center = arr[:, 1:-1, :]
        left_diff = np.abs(center - arr[:, :-2, :])
        right_diff = np.abs(center - arr[:, 2:, :])
        score[:, 1:-1, :] = np.minimum(left_diff, right_diff)

    return score[0] if squeeze_y else score


def estimate_vertical_residual_threshold(cube: EnviCube) -> np.ndarray:
    # Threshold is estimated from the SAME two-sided residual used during the
    # scan. This avoids the one-pixel -> three-pixel expansion caused by the old
    # immediate-neighbor-average residual.
    n = min(128, cube.lines)
    ys = np.unique(np.linspace(0, cube.lines - 1, n, dtype=int))
    pieces = []

    for y in ys:
        line = cube.read_lines(y, y + 1)[0]
        score = two_sided_vertical_residual(line)
        if cube.samples >= 5:
            pieces.append(score[2:-2])
        elif cube.samples >= 3:
            pieces.append(score[1:-1])

    if not pieces:
        return np.full(cube.bands, np.inf, dtype=np.float64)

    arr = np.stack(pieces, axis=0)
    med = np.median(arr, axis=(0, 1))
    mad = np.median(np.abs(arr - med[None, None, :]), axis=(0, 1))
    scale = 1.4826 * mad
    std = np.std(arr, axis=(0, 1))
    scale = np.where(scale > 0, scale, std)
    scale = np.where(scale > 0, scale, 1.0)
    return med + REFERENCE_ROBUST_Z * scale


def detector_persistence_maps(
    cube: EnviCube,
    clipping_code: int | None,
    residual_threshold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    clipped_count = np.zeros((cube.samples, cube.bands), dtype=np.uint32)
    residual_count = np.zeros((cube.samples, cube.bands), dtype=np.uint32)

    for y0 in range(0, cube.lines, CHUNK_LINES):
        y1 = min(cube.lines, y0 + CHUNK_LINES)
        raw = cube.read_lines(y0, y1)

        if clipping_code is not None:
            clipped_count += np.sum(raw == clipping_code, axis=0, dtype=np.uint32)

        residual = two_sided_vertical_residual(raw)
        bad = residual > residual_threshold[None, None, :]
        residual_count += np.sum(bad, axis=0, dtype=np.uint32)

    clip_fraction = clipped_count.astype(np.float32) / float(cube.lines)
    residual_fraction = residual_count.astype(np.float32) / float(cube.lines)
    return clip_fraction, residual_fraction


@dataclass
class DatasetInfo:
    name: str
    main: EnviCube
    white: EnviCube
    dark: EnviCube
    clipping_code: int | None
    clipping_status: str
    reference_bad: np.ndarray
    residual_threshold: np.ndarray
    clip_fraction: np.ndarray
    residual_fraction: np.ndarray
    strong_detector: np.ndarray
    weak_detector: np.ndarray
    white_mean: np.ndarray
    dark_mean: np.ndarray


def load_dataset_info(name: str) -> DatasetInfo:
    print(f"\nPreparing diagnostics for {name}")
    folder = next((p for p in (INPUT_ROOT / "test" / name,
                                INPUT_ROOT / "train" / name,
                                INPUT_ROOT / name) if p.is_dir()), INPUT_ROOT / name)
    main_hdr, white_hdr, dark_hdr = find_headers_in_folder(folder)

    main = EnviCube(main_hdr)
    white = EnviCube(white_hdr)
    dark = EnviCube(dark_hdr)
    validate_reference_compatibility(main, white, dark)

    clipping_code, clipping_status = infer_clipping_code(main)
    print(f"  clipping code: {clipping_code} ({clipping_status})")

    reference_bad = reference_bad_detector_mask(white, dark)
    residual_threshold = estimate_vertical_residual_threshold(main)
    clip_fraction, residual_fraction = detector_persistence_maps(
        main, clipping_code, residual_threshold
    )

    strong = (
        (clip_fraction >= STRONG_LINE_Y_FRACTION)
        | (residual_fraction >= STRONG_LINE_Y_FRACTION)
    )
    weak = (
        (clip_fraction >= PARTIAL_LINE_MIN_Y_FRACTION)
        | (residual_fraction >= PARTIAL_LINE_MIN_Y_FRACTION)
    )

    # Cache mean WHITE/DARK references for calibrated visualization output.
    white_sum = np.zeros((white.samples, white.bands), dtype=np.float64)
    dark_sum = np.zeros((dark.samples, dark.bands), dtype=np.float64)

    for y0 in range(0, white.lines, CHUNK_LINES):
        y1 = min(white.lines, y0 + CHUNK_LINES)
        white_sum += np.sum(white.read_lines(y0, y1), axis=0, dtype=np.float64)

    for y0 in range(0, dark.lines, CHUNK_LINES):
        y1 = min(dark.lines, y0 + CHUNK_LINES)
        dark_sum += np.sum(dark.read_lines(y0, y1), axis=0, dtype=np.float64)

    white_mean = white_sum / max(1, white.lines)
    dark_mean = dark_sum / max(1, dark.lines)

    return DatasetInfo(
        name=name,
        main=main,
        white=white,
        dark=dark,
        clipping_code=clipping_code,
        clipping_status=clipping_status,
        reference_bad=reference_bad,
        residual_threshold=residual_threshold,
        clip_fraction=clip_fraction,
        residual_fraction=residual_fraction,
        strong_detector=strong,
        weak_detector=weak,
        white_mean=white_mean,
        dark_mean=dark_mean,
    )


# ============================================================
# CONFIRMED PROBLEM MASK
# ============================================================
def confirmed_missing_values(cube: EnviCube) -> list[float]:
    values: list[float] = []
    if USE_ENVI_DATA_IGNORE_VALUE_AS_CONFIRMED_MISSING and cube.data_ignore_value is not None:
        values.append(float(cube.data_ignore_value))
    for value in EXTRA_CONFIRMED_MISSING_VALUES:
        fv = float(value)
        if fv not in values:
            values.append(fv)
    return values


def value_matches(raw: np.ndarray, values: list[float]) -> np.ndarray:
    if not values:
        return np.zeros(raw.shape, dtype=bool)

    out = np.zeros(raw.shape, dtype=bool)
    for value in values:
        if np.issubdtype(raw.dtype, np.integer):
            if float(value).is_integer():
                out |= raw == int(value)
        else:
            out |= np.isclose(raw, value, rtol=0.0, atol=0.0, equal_nan=False)
    return out


def current_vertical_anomaly_chunk(
    raw: np.ndarray,
    clipping_code: int | None,
    residual_threshold: np.ndarray,
) -> np.ndarray:
    bad = np.zeros(raw.shape, dtype=bool)

    if clipping_code is not None:
        bad |= raw == clipping_code

    residual = two_sided_vertical_residual(raw)
    bad |= residual > residual_threshold[None, None, :]

    return bad


def build_problem_masks(
    info: DatasetInfo,
    strong_clip_recurrence: np.ndarray,
    weak_clip_recurrence: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path, np.ndarray, np.ndarray, int, int, int]:
    cube = info.main

    # Recurrence is strong evidence for sparse fixed detector elements, but not
    # when the same entire band is abnormal over most x positions. That case may
    # still be true radiometric saturation and therefore needs independent
    # reference/missing-value evidence.
    widespread_band = (
        np.mean(info.strong_detector, axis=0) >= WIDESPREAD_BAND_X_FRACTION
    )

    recurrent_sparse_clipped_strong = (
        strong_clip_recurrence >= MIN_RECURRENT_CLIPPED_STRONG_DATASETS
    ) & ~widespread_band[None, :]

    recurrent_sparse_clipped_partial = (
        weak_clip_recurrence >= MIN_RECURRENT_CLIPPED_PARTIAL_DATASETS
    ) & ~widespread_band[None, :]

    # CONFIRMED means:
    #   A) WHITE/DARK reference independently supports a bad detector coordinate,
    #      OR
    #   B) exact clipping at the same sparse detector coordinate recurs in many
    #      independent acquisitions.
    #
    # Residual recurrence by itself is NOT enough because a real vertical scene
    # feature (tray edge, sample edge, etc.) can recur at the same x position.
    confirmed_full_detector = info.strong_detector & (
        info.reference_bad | recurrent_sparse_clipped_strong
    )

    confirmed_partial_detector = (
        info.weak_detector
        & ~info.strong_detector
        & (info.reference_bad | recurrent_sparse_clipped_partial)
    )

    missing_values = confirmed_missing_values(cube)

    confirmed_mask_path = output_dir / "confirmed_problem_mask.npy"
    selected_mask_path = output_dir / f"{OUTPUT_MASK_MODE}_problem_mask.npy"

    confirmed_mask = np.lib.format.open_memmap(
        confirmed_mask_path,
        mode="w+",
        dtype=np.bool_,
        shape=(cube.lines, cube.samples, cube.bands),
    )
    selected_mask = np.lib.format.open_memmap(
        selected_mask_path,
        mode="w+",
        dtype=np.bool_,
        shape=(cube.lines, cube.samples, cube.bands),
    )

    csv_file = None
    csv_writer = None
    csv_rows = 0
    csv_truncated = False

    if WRITE_SELECTED_VOXEL_CSV:
        csv_file = (output_dir / f"{OUTPUT_MASK_MODE}_problem_voxels.csv").open(
            "w", newline="", encoding="utf-8"
        )
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "y",
            "x",
            "band_index",
            "wavelength_nm",
            "raw_value",
            "reason",
        ])

    excluded_clipped_count = 0
    confirmed_count = 0
    selected_count = 0
    total_chunks = math.ceil(cube.lines / CHUNK_LINES)

    for ci, y0 in enumerate(range(0, cube.lines, CHUNK_LINES), start=1):
        y1 = min(cube.lines, y0 + CHUNK_LINES)
        raw = cube.read_lines(y0, y1)

        explicit_missing = value_matches(raw, missing_values)
        if np.issubdtype(raw.dtype, np.floating):
            explicit_missing |= ~np.isfinite(raw)

        clipped = np.zeros(raw.shape, dtype=bool)
        if info.clipping_code is not None:
            clipped = raw == info.clipping_code

        residual = two_sided_vertical_residual(raw)
        residual_bad = residual > info.residual_threshold[None, None, :]
        current_anomaly = clipped | residual_bad

        # Even for a confirmed detector coordinate, mark ONLY y-values that are
        # actually abnormal in the current acquisition. This prevents a detector
        # coordinate with 80-90% persistence from forcing the remaining normal
        # y-values into reconstruction.
        full_detector_mask = (
            current_anomaly
            & confirmed_full_detector[None, :, :]
        )

        partial_detector_mask = (
            current_anomaly
            & confirmed_partial_detector[None, :, :]
        )

        confirmed_chunk = explicit_missing | full_detector_mask | partial_detector_mask
        all_detected_chunk = explicit_missing | current_anomaly

        if OUTPUT_MASK_MODE == "confirmed_only":
            selected_chunk = confirmed_chunk
        elif OUTPUT_MASK_MODE == "all_detected":
            selected_chunk = all_detected_chunk
        else:
            raise ValueError(
                "OUTPUT_MASK_MODE must be 'confirmed_only' or 'all_detected'"
            )

        confirmed_mask[y0:y1] = confirmed_chunk
        selected_mask[y0:y1] = selected_chunk

        confirmed_count += int(np.count_nonzero(confirmed_chunk))
        selected_count += int(np.count_nonzero(selected_chunk))

        if info.clipping_code is not None:
            # Anything clipped without independent artifactual/missing evidence
            # is deliberately excluded from the strict confirmed mask.
            excluded_clipped_count += int(np.count_nonzero(clipped & ~confirmed_chunk))

        if csv_writer is not None and not csv_truncated:
            ys, xs, bs = np.where(selected_chunk)
            remaining = MAX_SELECTED_VOXEL_CSV_ROWS - csv_rows
            n_write = min(len(ys), max(0, remaining))

            for j in range(n_write):
                yi = int(ys[j])
                xi = int(xs[j])
                bi = int(bs[j])
                reasons: list[str] = []

                if explicit_missing[yi, xi, bi]:
                    reasons.append("explicit_missing_or_ignore_value")
                if full_detector_mask[yi, xi, bi]:
                    reasons.append("confirmed_fixed_detector_artifact")
                elif partial_detector_mask[yi, xi, bi]:
                    reasons.append("confirmed_partial_detector_artifact")
                else:
                    if clipped[yi, xi, bi]:
                        reasons.append("clipped_value")
                    if residual_bad[yi, xi, bi]:
                        reasons.append("vertical_residual_anomaly")

                if not reasons:
                    reasons.append("detected_problem")

                csv_writer.writerow([
                    y0 + yi,
                    xi,
                    bi,
                    float(cube.wavelengths[bi]),
                    float(raw[yi, xi, bi]),
                    ";".join(reasons),
                ])

            csv_rows += n_write
            if len(ys) > n_write:
                csv_truncated = True

        if ci % 10 == 0 or ci == total_chunks:
            print(f"    mask scan {ci}/{total_chunks}")

    confirmed_mask.flush()
    selected_mask.flush()
    if csv_file is not None:
        csv_file.close()

    return (
        confirmed_mask_path,
        selected_mask_path,
        confirmed_full_detector,
        confirmed_partial_detector,
        confirmed_count,
        selected_count,
        excluded_clipped_count,
    )


def collapse_nearby_vertical_duplicate_lines(info: DatasetInfo, mask_path: Path, output_dir: Path) -> int:
    """
    Collapse near-duplicate vertical-line responses without dilating the mask.

    The common false-double pattern is two almost identical line masks at x and
    x+2 (one clean column between them) generated by one physical stripe.  For
    adjacent columns (x and x+1), we are conservative and collapse only when one
    response is clearly weaker, because a real detector defect may genuinely be
    two pixels wide.

    Only the overlapping y-positions are removed from the weaker duplicate.
    Unique non-overlapping portions are preserved.
    """
    if not COLLAPSE_NEARBY_VERTICAL_DUPLICATES:
        return 0

    cube = info.main
    mask = np.load(mask_path, mmap_mode="r+")
    changes: list[dict] = []
    removed_voxels = 0

    # Work band-by-band.  Only columns with long vertical coverage are eligible,
    # so isolated/patch anomalies are untouched.
    for b in range(cube.bands):
        plane = np.asarray(mask[:, :, b], dtype=bool)
        counts = np.sum(plane, axis=0, dtype=np.int64)
        coverage = counts.astype(np.float64) / float(cube.lines)
        candidates = np.flatnonzero(coverage >= VERTICAL_DUPLICATE_MIN_COVERAGE)
        if candidates.size < 2:
            continue

        # Residual strength is computed from the raw band only for candidate bands.
        raw_band = cube.read_band(b).astype(np.float64)
        residual_band = two_sided_vertical_residual(raw_band[:, :, None])[:, :, 0]

        # Recompute pairs iteratively because one suppression can affect the next.
        for i in range(len(candidates) - 1):
            x1 = int(candidates[i])
            for j in range(i + 1, len(candidates)):
                x2 = int(candidates[j])
                sep = x2 - x1
                if sep > VERTICAL_DUPLICATE_MAX_SEPARATION:
                    break
                if sep <= 0:
                    continue

                v1 = np.asarray(mask[:, x1, b], dtype=bool)
                v2 = np.asarray(mask[:, x2, b], dtype=bool)
                n1 = int(np.count_nonzero(v1))
                n2 = int(np.count_nonzero(v2))
                if n1 == 0 or n2 == 0:
                    continue

                overlap = v1 & v2
                n_overlap = int(np.count_nonzero(overlap))
                overlap_fraction = n_overlap / float(min(n1, n2))
                if overlap_fraction < VERTICAL_DUPLICATE_MIN_Y_OVERLAP:
                    continue

                # If both coordinates have independent strong evidence of being
                # real detector defects, keep both even when they are close.
                both_reference_bad = bool(info.reference_bad[x1, b] and info.reference_bad[x2, b])
                both_clipped = bool(
                    info.clip_fraction[x1, b] >= PARTIAL_LINE_MIN_Y_FRACTION
                    and info.clip_fraction[x2, b] >= PARTIAL_LINE_MIN_Y_FRACTION
                )
                if both_reference_bad or both_clipped:
                    continue

                # Score only the y positions currently marked at each x.  Median
                # residual is robust to a few extreme values.
                s1_vals = residual_band[v1, x1]
                s2_vals = residual_band[v2, x2]
                s1 = float(np.median(s1_vals)) if s1_vals.size else 0.0
                s2 = float(np.median(s2_vals)) if s2_vals.size else 0.0

                # Persistence is useful as a tie-breaker.
                p1 = float(max(info.residual_fraction[x1, b], info.clip_fraction[x1, b]))
                p2 = float(max(info.residual_fraction[x2, b], info.clip_fraction[x2, b]))
                score1 = s1 * (0.5 + p1)
                score2 = s2 * (0.5 + p2)

                stronger = max(score1, score2)
                weaker = min(score1, score2)
                weak_ratio = weaker / stronger if stronger > 0 else 1.0

                # For adjacent columns, preserve a plausible true 2-pixel-wide
                # stripe unless one response is clearly weaker.
                if sep == 1 and weak_ratio > VERTICAL_DUPLICATE_ADJACENT_WEAK_RATIO:
                    continue

                if score1 >= score2:
                    keep_x, remove_x = x1, x2
                else:
                    keep_x, remove_x = x2, x1

                remove_vec = np.asarray(mask[:, remove_x, b], dtype=bool)
                keep_vec = np.asarray(mask[:, keep_x, b], dtype=bool)
                duplicate_y = remove_vec & keep_vec
                n_remove = int(np.count_nonzero(duplicate_y))
                if n_remove == 0:
                    continue

                mask[duplicate_y, remove_x, b] = False
                removed_voxels += n_remove
                changes.append({
                    "band_index": b,
                    "wavelength_nm": float(cube.wavelengths[b]),
                    "kept_x": keep_x,
                    "removed_duplicate_x": remove_x,
                    "separation_px": sep,
                    "y_overlap_fraction": overlap_fraction,
                    "kept_score": max(score1, score2),
                    "removed_score": min(score1, score2),
                    "removed_voxels": n_remove,
                })

    mask.flush()

    if changes:
        csv_path = output_dir / "collapsed_vertical_duplicate_lines.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(changes[0].keys()))
            writer.writeheader()
            writer.writerows(changes)
        print(
            f"  collapsed {len(changes)} near-duplicate vertical-line pairs "
            f"({removed_voxels} overlapping voxels removed)"
        )
    else:
        print("  no near-duplicate vertical-line responses needed collapsing")

    return removed_voxels


# ============================================================
# SCENARIO GEOMETRY DETECTION
# ============================================================
def true_ranges_1d(values: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(values)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    return [(int(g[0]), int(g[-1])) for g in groups]


def longest_true_run(values: np.ndarray) -> int:
    ranges = true_ranges_1d(values)
    if not ranges:
        return 0
    return max(b - a + 1 for a, b in ranges)


def template_score(spatial_size: int, band_width: int, templates: list[tuple[int, int]]) -> float:
    scores = []
    for s, b in templates:
        score = abs(math.log((spatial_size + 1) / (s + 1))) + abs(
            math.log((band_width + 1) / (b + 1))
        )
        scores.append(score)
    return min(scores)


def append_instance(
    instances: list[dict],
    scenario: str,
    location: str,
    details: str,
) -> None:
    instances.append({
        "scenario": scenario,
        "location": location,
        "details": details,
    })


def detect_scenarios(
    info: DatasetInfo,
    mask_path: Path,
    confirmed_full_detector: np.ndarray,
    confirmed_partial_detector: np.ndarray,
) -> tuple[list[dict], dict[str, int]]:
    cube = info.main
    mask = np.load(mask_path, mmap_mode="r")

    # Projections.
    detector_y_count = np.zeros((cube.samples, cube.bands), dtype=np.int64)
    band_spatial_count = np.zeros(cube.bands, dtype=np.int64)
    spatial_band_count = np.zeros((cube.lines, cube.samples), dtype=np.uint16)

    for y0 in range(0, cube.lines, CHUNK_LINES):
        y1 = min(cube.lines, y0 + CHUNK_LINES)
        m = np.asarray(mask[y0:y1], dtype=bool)
        detector_y_count += np.sum(m, axis=0, dtype=np.int64)
        band_spatial_count += np.sum(m, axis=(0, 1), dtype=np.int64)
        spatial_band_count[y0:y1] = np.sum(m, axis=2, dtype=np.uint16)

    instances: list[dict] = []
    counts = {scenario: 0 for scenario in SCENARIOS}

    # --------------------------------------------------------
    # 1) band_wide_saturation
    # --------------------------------------------------------
    band_fraction = band_spatial_count / float(cube.lines * cube.samples)
    band_wide = band_fraction >= BAND_WIDE_SPATIAL_FRACTION
    band_wide_ranges = true_ranges_1d(band_wide)

    for b0, b1 in band_wide_ranges:
        counts["band_wide_saturation"] += 1
        append_instance(
            instances,
            "band_wide_saturation",
            f"bands={b0}-{b1}",
            (
                f"wavelength={cube.wavelengths[b0]:.2f}-{cube.wavelengths[b1]:.2f} nm; "
                f"width={b1-b0+1}; minimum spatial coverage="
                f"{float(np.min(band_fraction[b0:b1+1])):.4f}"
            ),
        )

    # --------------------------------------------------------
    # 2) whole_spectrum_pixels
    # --------------------------------------------------------
    whole_spectrum = (
        spatial_band_count / float(cube.bands)
        >= WHOLE_SPECTRUM_BAND_FRACTION
    )
    labels, n_labels = ndimage.label(whole_spectrum, structure=np.ones((3, 3), dtype=np.uint8))

    for label_id in range(1, n_labels + 1):
        ys, xs = np.where(labels == label_id)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        counts["whole_spectrum_pixels"] += 1
        append_instance(
            instances,
            "whole_spectrum_pixels",
            f"y={y0}-{y1}, x={x0}-{x1}",
            f"spatial_pixels={ys.size}; >= {WHOLE_SPECTRUM_BAND_FRACTION:.0%} of spectrum confirmed bad",
        )

    # --------------------------------------------------------
    # 3-5) detector-line scenarios
    # --------------------------------------------------------
    detector_fraction = detector_y_count / float(cube.lines)

    # Do not reinterpret an already detected full bad band as thousands of
    # detector lines.
    eligible_detector = ~band_wide[None, :]

    full_lines = (
        (detector_fraction >= STRONG_LINE_Y_FRACTION)
        & eligible_detector
    )
    if OUTPUT_MASK_MODE == "confirmed_only":
        full_lines &= confirmed_full_detector

    # Count each contiguous x-range in one band as ONE visible detector stripe.
    # This prevents a genuinely 2-3-pixel-wide stripe from being reported as
    # several separate line scenarios.
    full_line_instances: list[tuple[int, int, int]] = []  # (band, x0, x1)
    for b in range(cube.bands):
        xs = np.flatnonzero(full_lines[:, b])
        for x0, x1 in true_ranges_1d(np.isin(np.arange(cube.samples), xs)):
            full_line_instances.append((b, x0, x1))

    if len(full_line_instances) == 1:
        b, x0, x1 = full_line_instances[0]
        counts["single_detector_line"] = 1
        x_label = f"{x0}" if x0 == x1 else f"{x0}-{x1}"
        coverage = float(np.min(detector_fraction[x0:x1+1, b]))
        append_instance(
            instances,
            "single_detector_line",
            f"x={x_label}, band={b}",
            (
                f"wavelength={cube.wavelengths[b]:.2f} nm; "
                f"width={x1-x0+1}; minimum y coverage={coverage:.4f}"
            ),
        )
    elif len(full_line_instances) >= 2:
        counts["multiple_detector_lines"] = len(full_line_instances)
        examples = "; ".join(
            (
                f"(x={x0 if x0 == x1 else f'{x0}-{x1}'},"
                f"b={b},wl={cube.wavelengths[b]:.2f},width={x1-x0+1})"
            )
            for b, x0, x1 in full_line_instances[:20]
        )
        append_instance(
            instances,
            "multiple_detector_lines",
            f"count={len(full_line_instances)}",
            f"examples: {examples}",
        )

    partial_map = (
        (detector_fraction >= PARTIAL_LINE_MIN_Y_FRACTION)
        & (detector_fraction < STRONG_LINE_Y_FRACTION)
        & eligible_detector
    )
    if OUTPUT_MASK_MODE == "confirmed_only":
        partial_map &= confirmed_partial_detector

    partial_instances: list[tuple[int, int, int, int]] = []  # b,x0,x1,longest
    for b in range(cube.bands):
        bool_x = partial_map[:, b]
        for x0, x1 in true_ranges_1d(bool_x):
            longest = 0
            for x in range(x0, x1 + 1):
                vector = np.asarray(mask[:, x, b], dtype=bool)
                longest = max(longest, longest_true_run(vector))
            if longest / float(cube.lines) < PARTIAL_LINE_LONGEST_RUN_FRACTION:
                continue
            partial_instances.append((b, x0, x1, longest))

    if partial_instances:
        counts["partial_vertical_lines"] = len(partial_instances)
        examples = "; ".join(
            (
                f"(x={x0 if x0 == x1 else f'{x0}-{x1}'},"
                f"b={b},wl={cube.wavelengths[b]:.2f},run={longest})"
            )
            for b, x0, x1, longest in partial_instances[:20]
        )
        append_instance(
            instances,
            "partial_vertical_lines",
            f"count={len(partial_instances)}",
            "examples: " + examples,
        )

    # --------------------------------------------------------
    # 6-7) spatial_patch / spatio_spectral_block
    # --------------------------------------------------------
    # Start from the spatial projection, but remove whole-spectrum regions.
    spatial_any = spatial_band_count > 0
    spatial_for_blocks = spatial_any & ~whole_spectrum

    labels2d, n2d = ndimage.label(
        spatial_for_blocks,
        structure=np.ones((3, 3), dtype=np.uint8),
    )

    patch_spatial_mask = np.zeros((cube.lines, cube.samples), dtype=bool)
    block_instances: list[tuple[int, int, int, int, int, int, str]] = []

    for label_id in range(1, n2d + 1):
        ys, xs = np.where(labels2d == label_id)
        if ys.size < 2:
            continue

        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        sy = y1 - y0 + 1
        sx = x1 - x0 + 1
        bbox_area = sy * sx
        density = ys.size / float(bbox_area)
        squareness = min(sy, sx) / float(max(sy, sx))

        # Lines are handled separately.
        if sx <= 4 and sy / float(cube.lines) >= PARTIAL_LINE_MIN_Y_FRACTION:
            continue

        if density < PATCH_COMPONENT_DENSITY or squareness < PATCH_SQUARENESS_MIN:
            continue

        component_mask_2d = labels2d[y0:y1+1, x0:x1+1] == label_id
        sub = np.asarray(mask[y0:y1+1, x0:x1+1, :], dtype=bool)
        component_voxels = sub[component_mask_2d]
        if component_voxels.size == 0:
            continue

        band_support = np.mean(component_voxels, axis=0)
        supported_bands = band_support >= PATCH_BAND_SUPPORT_FRACTION
        # Do not classify full bad bands as local blocks.
        supported_bands &= ~band_wide

        for b0, b1 in true_ranges_1d(supported_bands):
            bw = b1 - b0 + 1
            if bw <= 0:
                continue

            spatial_size = max(sy, sx)
            patch_score = template_score(
                spatial_size, bw, SPATIAL_PATCH_TEMPLATES
            )
            block_score = template_score(
                spatial_size, bw, SPATIO_SPECTRAL_TEMPLATES
            )

            if patch_score <= block_score:
                scenario = "spatial_patch"
            else:
                scenario = "spatio_spectral_block"

            counts[scenario] += 1
            append_instance(
                instances,
                scenario,
                f"y={y0}-{y1}, x={x0}-{x1}, bands={b0}-{b1}",
                (
                    f"spatial_size={sy}x{sx}; band_width={bw}; "
                    f"component_density={density:.3f}; "
                    f"band_support_min={float(np.min(band_support[b0:b1+1])):.3f}"
                ),
            )
            patch_spatial_mask[ys, xs] = True
            block_instances.append((y0, y1, x0, x1, b0, b1, scenario))

    # --------------------------------------------------------
    # 8) consecutive_spectral_bands
    # --------------------------------------------------------
    consecutive_count = 0
    consecutive_examples: list[str] = []

    detector_any = confirmed_full_detector | confirmed_partial_detector

    for y0 in range(0, cube.lines, CHUNK_LINES):
        y1 = min(cube.lines, y0 + CHUNK_LINES)
        m = np.asarray(mask[y0:y1], dtype=bool).copy()

        # Remove structures that have already been assigned to another scenario.
        if np.any(band_wide):
            m[:, :, band_wide] = False

        whole_local = whole_spectrum[y0:y1]
        if np.any(whole_local):
            m[whole_local, :] = False

        # Remove fixed detector coordinates.
        if np.any(detector_any):
            m &= ~detector_any[None, :, :]

        # Remove compact patch/block spatial regions. These would otherwise look
        # like many per-pixel spectral gaps.
        patch_local = patch_spatial_mask[y0:y1]
        if np.any(patch_local):
            m[patch_local, :] = False

        band_counts_at_pixel = np.sum(m, axis=2)
        candidates = np.argwhere(band_counts_at_pixel >= CONSECUTIVE_SPECTRAL_MIN_WIDTH)

        for yi, x in candidates:
            vec = m[int(yi), int(x)]
            for b0, b1 in true_ranges_1d(vec):
                width = b1 - b0 + 1
                if width < CONSECUTIVE_SPECTRAL_MIN_WIDTH:
                    continue
                consecutive_count += 1
                if len(consecutive_examples) < 30:
                    consecutive_examples.append(
                        f"(y={y0+int(yi)},x={int(x)},bands={b0}-{b1},width={width})"
                    )

    if consecutive_count:
        counts["consecutive_spectral_bands"] = consecutive_count
        append_instance(
            instances,
            "consecutive_spectral_bands",
            f"count={consecutive_count}",
            "examples: " + "; ".join(consecutive_examples),
        )

    # --------------------------------------------------------
    # 9) isolated_voxels
    # --------------------------------------------------------
    isolated_count = 0
    isolated_examples: list[str] = []
    structure = np.zeros((3, 3, 3), dtype=np.uint8)
    structure[1, 1, 1] = 1
    structure[0, 1, 1] = 1
    structure[2, 1, 1] = 1
    structure[1, 0, 1] = 1
    structure[1, 2, 1] = 1
    structure[1, 1, 0] = 1
    structure[1, 1, 2] = 1

    for y0 in range(0, cube.lines, CHUNK_LINES):
        y1 = min(cube.lines, y0 + CHUNK_LINES)
        halo0 = max(0, y0 - 1)
        halo1 = min(cube.lines, y1 + 1)
        block = np.asarray(mask[halo0:halo1], dtype=np.uint8)
        neighbor_count = ndimage.convolve(block, structure, mode="constant", cval=0)

        local0 = y0 - halo0
        local1 = local0 + (y1 - y0)
        center_block = block[local0:local1].astype(bool)
        center_neighbor_count = neighbor_count[local0:local1]
        isolated = center_block & (center_neighbor_count == 1)

        ys, xs, bs = np.where(isolated)
        isolated_count += len(ys)
        for j in range(min(len(ys), max(0, 30 - len(isolated_examples)))):
            isolated_examples.append(
                f"(y={y0+int(ys[j])},x={int(xs[j])},b={int(bs[j])})"
            )

    if isolated_count:
        counts["isolated_voxels"] = isolated_count
        append_instance(
            instances,
            "isolated_voxels",
            f"count={isolated_count}",
            "examples: " + "; ".join(isolated_examples),
        )

    # --------------------------------------------------------
    # 10) mixed_saturation
    # --------------------------------------------------------
    base_present = [
        s for s in SCENARIOS
        if s != "mixed_saturation" and counts[s] > 0
    ]
    if len(base_present) >= 2:
        counts["mixed_saturation"] = 1
        append_instance(
            instances,
            "mixed_saturation",
            "dataset-level mixture",
            "confirmed component families present: " + ", ".join(base_present),
        )

    return instances, counts


# ============================================================
# OUTPUTS
# ============================================================
def calibrated_band(info: DatasetInfo, band: int) -> np.ndarray:
    """Return calibrated reflectance for one band using (raw-dark)/(white-dark)."""
    raw = info.main.read_band(band).astype(np.float32)
    white = info.white_mean[:, band].astype(np.float32)[None, :]
    dark = info.dark_mean[:, band].astype(np.float32)[None, :]

    denom = white - dark
    denom = np.where(np.abs(denom) > 1e-12, denom, np.nan)
    refl = (raw - dark) / denom
    refl = np.nan_to_num(refl, nan=0.0, posinf=1.0, neginf=0.0)
    return refl


def normalize_band_to_u8(array: np.ndarray) -> np.ndarray:
    arr = array.astype(np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)

    lo, hi = np.percentile(arr[finite], [1.0, 99.0])
    if hi <= lo:
        lo = float(np.min(arr[finite]))
        hi = float(np.max(arr[finite]))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    # Brighten the saved visualization only (gamma < 1 -> brighter).
    gamma = float(DISPLAY_GAMMA) if DISPLAY_GAMMA is not None else 1.0
    if gamma > 0 and gamma != 1.0:
        out = np.power(out, gamma)

    return np.round(out * 255).astype(np.uint8)


def make_original_band_image(cube: EnviCube, band: int) -> Image.Image:
    raw = cube.read_band(band)
    gray = normalize_band_to_u8(raw)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)

    # Clean original image: no lines, no text, no markers.
    return Image.fromarray(rgb, mode="RGB")


def make_problem_overlay(cube: EnviCube, mask: np.ndarray, band: int) -> Image.Image:
    raw = cube.read_band(band)
    gray = normalize_band_to_u8(raw)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)

    problem = np.asarray(mask[:, :, band], dtype=bool)
    rgb[problem] = np.array(OVERLAY_HIGHLIGHT_RGB, dtype=np.uint8)

    return Image.fromarray(rgb, mode="RGB")


def save_mask_stack_tiff(info: DatasetInfo, mask_path: Path, output_dir: Path) -> None:
    if not WRITE_MASK_STACK_TIFF:
        return

    cube = info.main
    mask = np.load(mask_path, mmap_mode="r")
    out_path = output_dir / f"{OUTPUT_MASK_MODE}_problem_mask_stack.tiff"

    with tifffile.TiffWriter(out_path, bigtiff=True) as tif:
        for b in range(cube.bands):
            plane = np.asarray(mask[:, :, b], dtype=np.uint8) * int(MASK_STACK_FOREGROUND_VALUE)
            tif.write(
                plane,
                photometric="minisblack",
                metadata={"axes": "YX", "band_index": int(b), "wavelength_nm": float(cube.wavelengths[b])},
            )


def save_overlay_stack_tiff(info: DatasetInfo, mask_path: Path, output_dir: Path) -> None:
    if not WRITE_OVERLAY_STACK_TIFF:
        return

    cube = info.main
    mask = np.load(mask_path, mmap_mode="r")
    out_path = output_dir / f"{OUTPUT_MASK_MODE}_problem_overlay_stack.tiff"

    with tifffile.TiffWriter(out_path, bigtiff=True) as tif:
        for b in range(cube.bands):
            raw = cube.read_band(b)
            gray = normalize_band_to_u8(raw)
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
            problem = np.asarray(mask[:, :, b], dtype=bool)
            rgb[problem] = np.array(OVERLAY_HIGHLIGHT_RGB, dtype=np.uint8)
            tif.write(
                rgb,
                photometric="rgb",
                metadata={"axes": "YXS", "band_index": int(b), "wavelength_nm": float(cube.wavelengths[b])},
            )


def save_problem_band_rgb_images(info: DatasetInfo, mask_path: Path, output_dir: Path) -> None:
    """
    Save one independent 2-D RGB PNG per selected spectral band.

    Each output image is side-by-side:
      LEFT  = calibrated original band
      RIGHT = same calibrated band with detected problem pixels painted red

    No dilation or thickening is applied to the detected pixels.
    """
    if not SAVE_PROBLEM_BAND_RGB_IMAGES:
        return

    cube = info.main
    mask = np.load(mask_path, mmap_mode="r")

    counts = np.zeros(cube.bands, dtype=np.int64)
    for y0 in range(0, cube.lines, CHUNK_LINES):
        y1 = min(cube.lines, y0 + CHUNK_LINES)
        counts += np.sum(mask[y0:y1], axis=(0, 1), dtype=np.int64)

    if SAVE_ONLY_BANDS_WITH_PROBLEMS:
        bands = np.flatnonzero(counts > 0)
    else:
        bands = np.arange(cube.bands)

    if bands.size == 0:
        print("  no problem bands found; no RGB visualization images written")
        return

    out_dir = output_dir / RGB_OUTPUT_FOLDER_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    for b in bands:
        b = int(b)
        wl = float(cube.wavelengths[b])

        # Calibrated reflectance for visualization
        refl = calibrated_band(info, b)
        gray = normalize_band_to_u8(refl)

        # LEFT: clean calibrated image
        original_rgb = np.repeat(gray[:, :, None], 3, axis=2)

        # RIGHT: same calibrated image with problem pixels in red
        overlay_rgb = original_rgb.copy()
        problem = np.asarray(mask[:, :, b], dtype=bool)
        overlay_rgb[problem] = np.array(OVERLAY_HIGHLIGHT_RGB, dtype=np.uint8)

        # Join horizontally: original | overlay
        combined = np.concatenate([original_rgb, overlay_rgb], axis=1)

        Image.fromarray(combined, mode="RGB").save(
            out_dir / f"band_{b:04d}_{wl:.2f}nm_calibrated_original_PLUS_problem.png"
        )

    print(f"  saved {len(bands)} side-by-side 2-D RGB band images to: {out_dir}")


def save_problem_overlays(info: DatasetInfo, mask_path: Path, output_dir: Path) -> None:
    """
    Save ONLY full-resolution band images for bands that contain confirmed
    bad/missing voxels.

    For each band, produce:
      1) the clean original image
      2) the overlay image
      3) a side-by-side comparison: original | overlay
    """
    cube = info.main
    mask = np.load(mask_path, mmap_mode="r")

    counts = np.zeros(cube.bands, dtype=np.int64)
    for y0 in range(0, cube.lines, CHUNK_LINES):
        y1 = min(cube.lines, y0 + CHUNK_LINES)
        counts += np.sum(mask[y0:y1], axis=(0, 1), dtype=np.int64)

    problem_bands = np.flatnonzero(counts > 0)
    if problem_bands.size == 0:
        return

    original_dir = output_dir / "original_band_images"
    overlay_dir = output_dir / "problem_band_overlays"
    side_by_side_dir = output_dir / "comparison_side_by_side"

    original_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    side_by_side_dir.mkdir(parents=True, exist_ok=True)

    for b in problem_bands:
        b = int(b)
        wl = float(cube.wavelengths[b])

        original_img = make_original_band_image(cube, b)
        overlay_img = make_problem_overlay(cube, mask, b)

        original_img.save(
            original_dir / f"band_{b:04d}_{wl:.2f}nm_original_FULL.png"
        )
        overlay_img.save(
            overlay_dir / f"band_{b:04d}_{wl:.2f}nm_{OUTPUT_MASK_MODE}_problem_FULL.png"
        )

        combined = Image.new(
            "RGB",
            (original_img.width + overlay_img.width, max(original_img.height, overlay_img.height)),
            (0, 0, 0),
        )
        combined.paste(original_img, (0, 0))
        combined.paste(overlay_img, (original_img.width, 0))
        combined.save(
            side_by_side_dir / f"band_{b:04d}_{wl:.2f}nm_original_PLUS_overlay.png"
        )

def write_saturation_summary(info: DatasetInfo, output_dir: Path) -> None:
    """
    Report exact raw clipping/saturation separately from the general problem mask.

    Saturation here means: raw voxel value == the inferred/manual clipping code.
    This report is independent of residual anomalies, detector-line collapsing,
    and the 10 scenario labels.
    """
    cube = info.main
    per_band = np.zeros(cube.bands, dtype=np.int64)

    if info.clipping_code is not None:
        for y0 in range(0, cube.lines, CHUNK_LINES):
            y1 = min(cube.lines, y0 + CHUNK_LINES)
            raw = cube.read_lines(y0, y1)
            per_band += np.sum(
                raw == info.clipping_code,
                axis=(0, 1),
                dtype=np.int64,
            )

    total_saturated = int(per_band.sum())
    saturated_bands = np.flatnonzero(per_band > 0)
    pixels_per_band = int(cube.lines * cube.samples)

    # Detailed per-band table: only bands that actually contain clipped voxels.
    csv_path = output_dir / "saturated_pixels_per_band.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "band_index",
            "wavelength_nm",
            "saturated_voxel_count",
            "spatial_fraction",
        ])
        for b in saturated_bands:
            b = int(b)
            writer.writerow([
                b,
                float(cube.wavelengths[b]),
                int(per_band[b]),
                float(per_band[b] / pixels_per_band),
            ])

    # Small human-readable summary.
    txt_path = output_dir / "saturation_summary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"dataset: {info.name}\n")
        f.write(f"clipping_code: {info.clipping_code}\n")
        f.write(f"clipping_status: {info.clipping_status}\n")
        f.write(f"total_saturated_clipped_voxels: {total_saturated}\n")
        f.write(f"bands_with_saturation: {len(saturated_bands)}\n")
        f.write(f"total_bands: {cube.bands}\n")
        if saturated_bands.size:
            f.write("saturated_band_indices: " + ", ".join(str(int(b)) for b in saturated_bands) + "\n")
        else:
            f.write("saturated_band_indices: none\n")

    print("  saturation/clipping summary:")
    print(f"    clipping code: {info.clipping_code} ({info.clipping_status})")
    print(f"    total saturated/clipped voxels: {total_saturated}")
    print(f"    bands containing saturation: {len(saturated_bands)}")
    if saturated_bands.size:
        preview = ", ".join(str(int(b)) for b in saturated_bands[:20])
        if saturated_bands.size > 20:
            preview += ", ..."
        print(f"    saturated band indices: {preview}")


def write_scenario_outputs(
    info: DatasetInfo,
    instances: list[dict],
    counts: dict[str, int],
    confirmed_count: int,
    excluded_clipped_count: int,
    output_dir: Path,
) -> None:
    found = [s for s in SCENARIOS if counts[s] > 0]

    # Only detected scenarios are written. Absent/suspicious scenarios are not
    # added as rows.
    with (output_dir / "detected_scenarios.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "confirmed_instance_count"])
        for scenario in found:
            writer.writerow([scenario, counts[scenario]])

    with (output_dir / "detected_scenario_instances.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "location", "details"])
        for row in instances:
            writer.writerow([row["scenario"], row["location"], row["details"]])


# ============================================================
# MAIN
# ============================================================
def discover_datasets() -> list[str]:
    if PROCESS_ALL_SWIR_DATASETS:
        names = sorted({p.name for folder in (INPUT_ROOT / "train", INPUT_ROOT / "test", INPUT_ROOT)
                        for p in folder.glob("SWIR*") if p.is_dir()})
        if not names:
            raise FileNotFoundError(f"No SWIR* folders found under:\n{INPUT_ROOT}")
        return names
    return [PRIMARY_DATASET]


def main() -> None:
    dataset_names = discover_datasets()
    print("Datasets:", ", ".join(dataset_names))

    # Build the detector diagnostics once for each acquisition, so SWIR1-SWIR5
    # can support each other without rescanning every cube for every output.
    infos: dict[str, DatasetInfo] = {}
    for name in dataset_names:
        infos[name] = load_dataset_info(name)

    # Group only geometrically compatible acquisitions for recurrence.
    for primary_name in dataset_names:
        info = infos[primary_name]
        compatible = []
        for other in dataset_names:
            other_cube = infos[other].main
            same_geometry = (
                other_cube.samples == info.main.samples
                and other_cube.bands == info.main.bands
                and other_cube.x_start == info.main.x_start
            )
            same_spectral_axis = (
                same_geometry
                and np.allclose(
                    other_cube.wavelengths,
                    info.main.wavelengths,
                    rtol=0.0,
                    atol=0.5,
                )
            )
            if same_spectral_axis:
                compatible.append(other)

        strong_clip_recurrence = np.zeros(
            (info.main.samples, info.main.bands), dtype=np.uint16
        )
        weak_clip_recurrence = np.zeros_like(strong_clip_recurrence)

        for other in compatible:
            other_info = infos[other]
            strong_clip_recurrence += (
                other_info.clip_fraction >= STRONG_LINE_Y_FRACTION
            ).astype(np.uint16)
            weak_clip_recurrence += (
                other_info.clip_fraction >= PARTIAL_LINE_MIN_Y_FRACTION
            ).astype(np.uint16)

        output_dir = PROJECT_DIR / "results" / f"{primary_name}_confirmed_10_scenarios"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 88)
        print(f"CONFIRMED PROBLEM DETECTION: {primary_name}")
        print("=" * 88)

        (
            confirmed_mask_path,
            selected_mask_path,
            confirmed_full_detector,
            confirmed_partial_detector,
            confirmed_count,
            selected_count,
            excluded_clipped_count,
        ) = build_problem_masks(
            info,
            strong_clip_recurrence,
            weak_clip_recurrence,
            output_dir,
        )

        removed_duplicate_voxels = collapse_nearby_vertical_duplicate_lines(
            info, selected_mask_path, output_dir
        )
        selected_count -= removed_duplicate_voxels

        print(f"  selected mask mode: {OUTPUT_MASK_MODE}; selected voxel count = {selected_count}")
        print("  detecting which of the 10 benchmark geometries are actually present...")
        instances, counts = detect_scenarios(
            info,
            selected_mask_path,
            confirmed_full_detector,
            confirmed_partial_detector,
        )

        write_scenario_outputs(
            info,
            instances,
            counts,
            confirmed_count,
            excluded_clipped_count,
            output_dir,
        )

        # Keep exact clipping/saturation statistics separate from the general
        # anomaly/scenario summary so it is always clear how much true clipping
        # was found in the raw cube.
        write_saturation_summary(info, output_dir)

        if SAVE_PROBLEM_BAND_RGB_IMAGES:
            print("  creating calibrated 2-D RGB problem-band images...")
            save_problem_band_rgb_images(info, selected_mask_path, output_dir)

        if SAVE_ALL_PROBLEM_BAND_OVERLAYS:
            print("  creating legacy full-resolution problem band overlays...")
            save_problem_overlays(info, selected_mask_path, output_dir)

        if WRITE_MASK_STACK_TIFF:
            print("  writing 3-D mask TIFF stack...")
            save_mask_stack_tiff(info, selected_mask_path, output_dir)

        if WRITE_OVERLAY_STACK_TIFF:
            print("  writing 3-D overlay TIFF stack...")
            save_overlay_stack_tiff(info, selected_mask_path, output_dir)

        # The NPY masks are needed only internally while the script is running.
        # Remove them after all outputs are made.
        for temp_mask_path in [confirmed_mask_path, selected_mask_path]:
            try:
                Path(temp_mask_path).unlink()
            except FileNotFoundError:
                pass

        print("  finished:", output_dir)
        print("  found scenarios:", [s for s in SCENARIOS if counts[s] > 0])

    print("\nAll datasets finished successfully.")


if __name__ == "__main__":
    main()
