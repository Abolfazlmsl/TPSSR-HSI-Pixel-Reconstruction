import csv
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

import config


def processed_dataset_dir(dataset_name):
    return (
        config.PROCESSED_DATA_ROOT
        / str(dataset_name)
    )


def load_dataset(dataset_name=None):
    if dataset_name is None:
        dataset_name = config.TEST_DATASET_FOLDER

    dataset_dir = processed_dataset_dir(
        dataset_name
    )

    cube_path = (
        dataset_dir
        / "calibrated_reflectance.npy"
    )
    invalid_path = (
        dataset_dir
        / "invalid_mask.npy"
    )
    split_path = (
        dataset_dir
        / "split_info.npz"
    )
    wavelength_path = (
        dataset_dir
        / "wavelengths_nm.npy"
    )

    required = [
        cube_path,
        invalid_path,
        split_path,
        wavelength_path,
    ]
    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "A calibrated dataset is missing. "
            "Run data_preparation/prepare_dataset.py first.\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )

    cube = np.load(
        cube_path,
        mmap_mode="r",
    )
    invalid = np.load(
        invalid_path,
        mmap_mode="r",
    )
    split = np.load(
        split_path,
    )
    wavelengths = np.load(
        wavelength_path,
    )

    return (
        cube,
        invalid,
        split,
        wavelengths,
    )


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


def list_mask_files():
    mask_dir = config.BENCHMARK_DIR / "masks"
    files = sorted(mask_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(
            "No saturation simulation masks were found. "
            "Run data_preparation/prepare_dataset.py first."
        )
    return files


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


def save_method_metrics(method_name, rows):
    output_dir = config.RESULTS_DIR / method_name
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metrics.csv"

    if not rows:
        return path

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return path
