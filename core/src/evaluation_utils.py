import csv
import math
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

import config


def processed_dataset_dir(dataset_name):
    return (
        config.PROCESSED_DATA_ROOT
        / str(dataset_name)
    )


def _dynamic_split(dataset_name, total_y):
    """Return the experiment-specific split without storing role in processed_data.

    This is important because the same acquisition can be training data in one
    experiment and held-out test data in another (e.g. region A->B vs B->A).
    """
    dataset_name = str(dataset_name).upper()
    total_y = int(total_y)

    if dataset_name in config.TRAIN_DATASET_FOLDERS:
        train_end = int(round(total_y * config.TRAIN_IMAGE_TRAIN_FRACTION))
        train_end = min(max(train_end, 1), max(total_y - 1, 1))
        val_end = total_y
        role = "train_validation"
    elif dataset_name in config.TEST_DATASET_FOLDERS:
        test_size = max(1, int(round(total_y * config.HELD_OUT_TEST_FRACTION)))
        test_start = max(0, total_y - test_size)
        train_end = 0
        val_end = test_start
        role = "held_out_test"
    else:
        raise ValueError(
            f"Dataset {dataset_name} is not part of the current experiment. "
            f"Train={config.TRAIN_DATASET_FOLDERS}; Test={config.TEST_DATASET_FOLDERS}"
        )

    return {
        "train_end": np.asarray(train_end, dtype=np.int64),
        "val_end": np.asarray(val_end, dtype=np.int64),
        "role": np.asarray(role),
        "dataset_name": np.asarray(dataset_name),
    }


def load_dataset(dataset_name=None):
    if dataset_name is None:
        dataset_name = config.ACTIVE_TEST_DATASET
    dataset_name = str(dataset_name).upper()

    dataset_dir = processed_dataset_dir(dataset_name)
    cube_path = dataset_dir / "calibrated_reflectance.npy"
    invalid_path = dataset_dir / "invalid_mask.npy"
    wavelength_path = dataset_dir / "wavelengths_nm.npy"

    required = [cube_path, invalid_path, wavelength_path]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "A calibrated dataset is missing. Run prepare_dataset.py first.\n"
            + "\n".join(str(path) for path in missing)
        )

    cube = np.load(cube_path, mmap_mode="r")
    invalid = np.load(invalid_path, mmap_mode="r")
    wavelengths = np.load(wavelength_path)
    split = _dynamic_split(dataset_name, cube.shape[0])

    return cube, invalid, split, wavelengths


def validate_wavelength_grid(
    reference,
    candidate,
    reference_name,
    candidate_name,
):
    reference = np.asarray(
        reference,
        dtype=np.float32,
    )
    candidate = np.asarray(
        candidate,
        dtype=np.float32,
    )

    if reference.shape != candidate.shape:
        raise RuntimeError(
            "The SWIR acquisitions do not have the same number of bands: "
            f"{reference_name}={len(reference)}, "
            f"{candidate_name}={len(candidate)}."
        )

    if not np.allclose(
        reference,
        candidate,
        rtol=0.0,
        atol=config.WAVELENGTH_MATCH_ATOL_NM,
    ):
        maximum_difference = float(
            np.max(
                np.abs(
                    reference
                    - candidate
                )
            )
        )
        raise RuntimeError(
            "The wavelength grids are not aligned closely enough for one "
            "global model. "
            f"Maximum difference between {reference_name} and "
            f"{candidate_name}: {maximum_difference:.4f} nm."
        )


def load_global_training_data():
    train_cubes = []
    val_cubes = []
    reference_wavelengths = None
    reference_name = None

    for dataset_name in config.TRAIN_DATASET_FOLDERS:
        cube, _, split, wavelengths = (
            load_dataset(
                dataset_name
            )
        )

        train, val, _ = split_blocks(
            cube,
            split,
        )

        if len(train) == 0 or len(val) == 0:
            raise RuntimeError(
                f"Training/validation split is empty for {dataset_name}. "
                "Run data_preparation/prepare_dataset.py again."
            )

        if reference_wavelengths is None:
            reference_wavelengths = np.asarray(
                wavelengths,
                dtype=np.float32,
            )
            reference_name = dataset_name
        else:
            validate_wavelength_grid(
                reference_wavelengths,
                wavelengths,
                reference_name,
                dataset_name,
            )

        train_cubes.append(
            train
        )
        val_cubes.append(
            val
        )

    if not train_cubes:
        raise RuntimeError(
            "No global training datasets are configured."
        )

    return (
        train_cubes,
        val_cubes,
        reference_wavelengths,
    )


def split_blocks(array, split):
    train_end = int(split["train_end"])
    val_end = int(split["val_end"])
    return array[:train_end], array[train_end:val_end], array[val_end:]


_MASK_NAME_RE = re.compile(r"^(?P<scenario>.+)__s(?P<severity>\d+)__r(?P<repeat>\d+)\.npz$")


def _mask_case_key(path):
    """Return (scenario, severity, repeat) without loading the full mask."""
    match = _MASK_NAME_RE.match(Path(path).name)
    if match:
        return (
            str(match.group("scenario")),
            int(match.group("severity")),
            int(match.group("repeat")),
        )

    # Fallback for non-standard filenames.
    with np.load(path, allow_pickle=False) as data:
        return (
            str(data["scenario"]),
            int(data["severity"]),
            int(data["repeat"]),
        )


def _experiment_total_mask_budget():
    raw = os.environ.get("HSI_TOTAL_TEST_MASKS")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"HSI_TOTAL_TEST_MASKS must be an integer, got {raw!r}."
        ) from exc
    return value


def _balanced_experiment_mask_keys(files, total_budget):
    """Select this acquisition's share of one GLOBAL experiment mask budget.

    The benchmark catalogue is stratified by (scenario, severity).  Candidate
    masks within a stratum are the repeat IDs.  A global budget (normally 200)
    is distributed across *all* configured held-out acquisitions, not applied
    independently to each acquisition.

    The allocation is deterministic and coordinated across acquisitions:
      * 2 test acquisitions + budget 200 -> 100 + 100
      * 5 test acquisitions + budget 200 -> 40 each

    For the standard 10 scenarios x 5 severities x 4 repeats catalogue, budget
    200 keeps exactly four cases per scenario/severity stratum across the whole
    experiment.  Which acquisition receives a stratum/repeat rotates so no test
    core is systematically favoured.
    """
    files = list(files)
    tests = [str(x).upper() for x in config.TEST_DATASET_FOLDERS]
    active = str(config.ACTIVE_TEST_DATASET).upper()
    if active not in tests:
        raise ValueError(
            f"ACTIVE_TEST_DATASET={active} is not in TEST_DATASET_FOLDERS={tests}."
        )
    if total_budget is None or int(total_budget) <= 0:
        return files
    by_stratum = {}
    path_by_key = {}
    for path in files:
        scenario, severity, repeat = _mask_case_key(path)
        by_stratum.setdefault((scenario, severity), set()).add(int(repeat))
        path_by_key[(scenario, severity, int(repeat))] = path

    strata = sorted(by_stratum)
    if not strata:
        return []

    # Maximum number of distinct experiment cases available across all held-out
    # acquisitions.  Each acquisition has its own physical mask realization.
    total_available = len(tests) * len(files)
    budget = min(int(total_budget), int(total_available))
    if budget >= total_available:
        return files

    # Build a deterministic candidate order for each stratum.  The first R
    # candidates (R = number of repeats) rotate over test acquisitions.  Thus
    # with 5 tests and 4 repeats, four different cores contribute one case each
    # to every scenario/severity stratum; the omitted core rotates by stratum.
    candidate_orders = []
    for s_idx, stratum in enumerate(strata):
        repeats = sorted(by_stratum[stratum])
        candidates = []
        seen = set()
        for round_idx in range(len(tests)):
            for rep_pos, repeat in enumerate(repeats):
                dataset_idx = (s_idx + round_idx + rep_pos) % len(tests)
                pair = (dataset_idx, int(repeat))
                if pair not in seen:
                    candidates.append(pair)
                    seen.add(pair)
        candidate_orders.append(candidates)

    # Fill one candidate per stratum at a time.  This keeps scenario/severity
    # representation as even as mathematically possible for any budget.
    selected = [set() for _ in tests]
    remaining = budget
    layer = 0
    while remaining > 0:
        added = 0
        for s_idx, stratum in enumerate(strata):
            candidates = candidate_orders[s_idx]
            if layer >= len(candidates):
                continue
            dataset_idx, repeat = candidates[layer]
            key = (stratum[0], int(stratum[1]), int(repeat))
            selected[dataset_idx].add(key)
            remaining -= 1
            added += 1
            if remaining <= 0:
                break
        if added == 0:
            break
        layer += 1

    active_idx = tests.index(active)
    wanted = selected[active_idx]
    chosen = [
        path for path in files
        if _mask_case_key(path) in wanted
    ]
    return chosen


def list_mask_files():
    """List benchmark masks for the active test acquisition.

    If HSI_TOTAL_TEST_MASKS is set, its value is interpreted as the TOTAL mask
    budget for the complete multi-acquisition experiment.  The returned list is
    this acquisition's deterministic stratified share of that global budget.
    """
    mask_dir = config.BENCHMARK_DIR / "masks"
    files = sorted(mask_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(
            "No saturation simulation masks were found. "
            "Run data_preparation/prepare_dataset.py first."
        )
    return _balanced_experiment_mask_keys(
        files,
        _experiment_total_mask_budget(),
    )


@lru_cache(maxsize=32)
def _full_mask_index_map(mask_dir_text):
    mask_dir = Path(mask_dir_text)
    return {
        path.name: index
        for index, path in enumerate(sorted(mask_dir.glob("*.npz")), start=1)
    }


def benchmark_case_index(mask_path):
    """Stable 1-based index in the FULL local 200-mask catalogue.

    Use this index for select_eval_coords so the sampled voxels for a particular
    scenario/severity/repeat do not change when the experiment-wide mask budget
    changes from 200-per-core to 200-total.
    """
    path = Path(mask_path)
    mapping = _full_mask_index_map(str(path.parent))
    if path.name not in mapping:
        raise KeyError(f"Mask {path.name} was not found in {path.parent}.")
    return int(mapping[path.name])


def experiment_progress(local_mask_index, local_total):
    """Return (global_index, global_total) for user-facing progress messages."""
    offset = int(os.environ.get("HSI_EXPERIMENT_MASK_OFFSET", "0"))
    total = int(
        os.environ.get(
            "HSI_EXPERIMENT_MASK_TOTAL",
            str(local_total),
        )
    )
    return offset + int(local_mask_index), total


def save_mask_file(path, mask, metadata):
    coords = np.argwhere(mask).astype(np.int32)
    payload = {
        "shape": np.asarray(mask.shape, dtype=np.int32),
        "scenario": np.asarray(metadata["scenario"]),
        "severity": np.asarray(int(metadata["severity"]), dtype=np.int16),
        "repeat": np.asarray(int(metadata["repeat"]), dtype=np.int16),
        "seed": np.asarray(int(metadata["seed"]), dtype=np.int64),
        "masked_voxels": np.asarray(int(len(coords)), dtype=np.int64),
    }

    if len(coords) <= config.SPARSE_COORD_THRESHOLD:
        payload["format"] = np.asarray("coords")
        payload["coords"] = coords
    else:
        payload["format"] = np.asarray("packbits")
        payload["packed"] = np.packbits(mask.reshape(-1))

    np.savez_compressed(path, **payload)


def load_test_mask(path):
    data = np.load(path, allow_pickle=False)
    shape = tuple(int(v) for v in data["shape"])
    fmt = str(data["format"])

    if fmt == "coords":
        mask = np.zeros(shape, dtype=bool)
        coords = data["coords"].astype(np.int64)
        if len(coords):
            mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    elif fmt == "packbits":
        packed = data["packed"]
        n = int(np.prod(shape))
        flat = np.unpackbits(packed, count=n).astype(bool)
        mask = flat.reshape(shape)
    else:
        raise ValueError(f"Unknown mask format: {fmt}")

    metadata = {
        "scenario": str(data["scenario"]),
        "severity": int(data["severity"]),
        "repeat": int(data["repeat"]),
        "seed": int(data["seed"]),
    }

    return mask, metadata


def select_eval_coords(mask, seed):
    spatial = np.any(mask, axis=2)

    margin = config.SPARSE_ROUTER_PATCH_SIZE // 2
    if spatial.shape[0] > 2 * margin and spatial.shape[1] > 2 * margin:
        allowed = np.zeros_like(spatial)
        allowed[margin:-margin, margin:-margin] = True
        spatial &= allowed

    pixels = np.argwhere(spatial)

    if len(pixels) == 0:
        return np.empty((0, 3), dtype=np.int64)

    masked_band_counts = np.array(
        [np.count_nonzero(mask[y, x, :]) for y, x in pixels],
        dtype=np.int64,
    )

    mean_bands = max(1.0, float(np.mean(masked_band_counts)))
    voxel_limited_pixels = max(1, int(config.MAX_EVAL_VOXELS / mean_bands))
    max_pixels = min(config.MAX_EVAL_SPATIAL_PIXELS, voxel_limited_pixels)

    if len(pixels) > max_pixels:
        rng = np.random.default_rng(seed)
        selected_idx = rng.choice(len(pixels), max_pixels, replace=False)
        pixels = pixels[selected_idx]

    coord_parts = []
    for y, x in pixels:
        bands = np.flatnonzero(mask[int(y), int(x), :])
        if len(bands):
            y_col = np.full(len(bands), int(y), dtype=np.int64)
            x_col = np.full(len(bands), int(x), dtype=np.int64)
            coord_parts.append(np.column_stack([y_col, x_col, bands]))

    if not coord_parts:
        return np.empty((0, 3), dtype=np.int64)

    return np.concatenate(coord_parts, axis=0)


def regression_metrics(truth, pred):
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    valid = np.isfinite(truth) & np.isfinite(pred)
    target_n = int(len(truth))
    valid_n = int(np.count_nonzero(valid))
    coverage = valid_n / target_n if target_n else 0.0

    if valid_n == 0:
        return {
            "target_n": target_n,
            "valid_n": 0,
            "coverage": coverage,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
        }

    t = truth[valid]
    p = pred[valid]

    return {
        "target_n": target_n,
        "valid_n": valid_n,
        "coverage": coverage,
        "mae": float(np.mean(np.abs(t - p))),
        "rmse": float(np.sqrt(np.mean((t - p) ** 2))),
        "r2": (
            float(r2_score(t, p))
            if len(t) > 1 and np.std(t) > 0
            else np.nan
        ),
    }


def spectral_angle_metrics(gt_cube, coords, preds):
    by_pixel = {}

    for coord, pred in zip(coords, preds):
        y, x, b = map(int, coord)
        by_pixel.setdefault((y, x), []).append((b, float(pred)))

    angles = []
    complete = 0

    for (y, x), items in by_pixel.items():
        if not all(np.isfinite(pred) for _, pred in items):
            continue

        gt = np.asarray(gt_cube[y, x, :], dtype=np.float64)
        if np.count_nonzero(np.isfinite(gt)) < 2:
            continue

        rec = gt.copy()
        for b, pred in items:
            rec[b] = pred

        valid = np.isfinite(gt) & np.isfinite(rec)
        a = gt[valid]
        bvec = rec[valid]

        denom = np.linalg.norm(a) * np.linalg.norm(bvec)
        if denom <= 1e-12:
            continue

        cosine = np.clip(np.dot(a, bvec) / denom, -1.0, 1.0)
        angles.append(math.degrees(math.acos(cosine)))
        complete += 1

    total_pixels = len(by_pixel)
    return {
        "sam_deg": float(np.mean(angles)) if angles else np.nan,
        "sam_pixel_coverage": complete / total_pixels if total_pixels else 0.0,
        "sam_pixels": complete,
        "target_pixels": total_pixels,
    }



def masked_ssim3_sparse(gt, coords_yx, pred_values, target_mask2d=None):
    """Replicate TPSSR's masked 3x3 sparse-window SSIM exactly.

    Only sampled, successfully reconstructed target voxels are substituted.
    Windows containing other masked/unreconstructed pixels are excluded.
    """
    gt = np.asarray(gt, float)
    coords_yx = np.asarray(coords_yx, dtype=np.int64)
    pred_values = np.asarray(pred_values, float)
    if len(coords_yx) == 0:
        return float("nan")

    finite = np.isfinite(pred_values)
    finite &= np.isfinite(gt[coords_yx[:, 0], coords_yx[:, 1]])
    coords_yx = coords_yx[finite]
    pred_values = pred_values[finite]
    if len(coords_yx) == 0:
        return float("nan")

    vals_gt = gt[coords_yx[:, 0], coords_yx[:, 1]]
    dr = max(float(np.nanpercentile(vals_gt, 99) - np.nanpercentile(vals_gt, 1)), 1e-6)
    c1, c2 = (.01 * dr) ** 2, (.03 * dr) ** 2
    lookup = {(int(y), int(x)): float(v) for (y, x), v in zip(coords_yx, pred_values)}
    target_mask2d = None if target_mask2d is None else np.asarray(target_mask2d, dtype=bool)
    vals = []
    h, w = gt.shape
    for y, x in coords_yx:
        y, x = int(y), int(x)
        if y < 1 or x < 1 or y >= h - 1 or x >= w - 1:
            continue
        a = gt[y-1:y+2, x-1:x+2].astype(float, copy=False).ravel()
        b = a.copy()
        k = 0
        window_valid = True
        for yy in range(y - 1, y + 2):
            for xx in range(x - 1, x + 2):
                pv = lookup.get((yy, xx))
                if pv is not None:
                    b[k] = pv
                elif target_mask2d is not None and bool(target_mask2d[yy, xx]):
                    window_valid = False
                    break
                k += 1
            if not window_valid:
                break
        if not window_valid or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        ma, mb = a.mean(), b.mean()
        va, vb = a.var(), b.var()
        cov = np.mean((a-ma) * (b-mb))
        den = (ma*ma + mb*mb + c1) * (va + vb + c2)
        if den > 1e-20:
            vals.append(((2*ma*mb+c1) * (2*cov+c2)) / den)
    return float(np.mean(vals)) if vals else float("nan")


def sparse_ssim_metrics(gt_cube, mask3d, eval_coords, predictions):
    """Evaluate baseline SSIM on precisely TPSSR's sparse sampled windows.

    Bandwise SSIM is aggregated using the number of valid reconstructed
    sampled voxels per band, matching spatial_inference.evaluate_reconstruction.
    If no valid 3x3 windows exist, SSIM is NaN (never artificially zero).
    """
    coords = np.asarray(eval_coords, dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) != len(pred):
        raise ValueError("eval_coords must be Nx3 and match predictions length")
    if len(coords) == 0:
        return {"ssim": float("nan"), "ssim_mean_band": float("nan")}

    scores, weights = [], []
    # Keep identical band ordering and weighting as the TPSSR evaluator.
    for band in np.unique(coords[:, 2]):
        band = int(band)
        in_band = coords[:, 2] == band
        c = coords[in_band, :2]
        pv = pred[in_band]
        truth = np.asarray(gt_cube[c[:, 0], c[:, 1], band])
        valid = (np.asarray(mask3d[c[:, 0], c[:, 1], band], bool)
                 & np.isfinite(truth) & np.isfinite(pv))
        c = c[valid]
        pv = pv[valid]
        n = int(len(c))
        if n == 0:
            continue
        score = masked_ssim3_sparse(
            np.asarray(gt_cube[:, :, band]), c, pv,
            target_mask2d=mask3d[:, :, band],
        )
        if np.isfinite(score):
            scores.append(float(score))
            weights.append(n)

    return {
        "ssim": float(np.average(scores, weights=weights)) if scores else float("nan"),
        "ssim_mean_band": float(np.mean(scores)) if scores else float("nan"),
    }



def save_method_metrics(method_name, rows):
    output_dir = config.RESULTS_DIR / method_name
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metrics.csv"

    if rows and "ssim" in rows[0] and path.exists():
        backup = output_dir / "metrics_before_ssim.csv"
        if not backup.exists():
            import shutil
            shutil.copy2(path, backup)

    if not rows:
        return path

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return path
