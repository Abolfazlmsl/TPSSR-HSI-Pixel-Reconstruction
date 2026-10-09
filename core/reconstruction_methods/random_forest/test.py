from pathlib import Path
import sys

HERE = Path(__file__).resolve()
PROJECT_ROOT = next(
    parent for parent in HERE.parents
    if (parent / "config.py").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from tqdm import tqdm

import config
from src.evaluation_utils import (
    load_dataset,
    split_blocks,
    list_mask_files,
    load_test_mask,
    select_eval_coords,
    regression_metrics,
    spectral_angle_metrics,
    save_method_metrics,
    benchmark_case_index,
    experiment_progress,
)
from src.ml_common import (
    extract_patch,
    rf_feature,
)


def predict(model, band_mean, test, mask, coords):
    predictions = np.full(len(coords), np.nan, dtype=np.float32)

    features = []
    indices = []

    def flush():
        if not features:
            return
        X = np.asarray(features, dtype=np.float32)
        output = model.predict(X)
        predictions[np.asarray(indices, dtype=np.int64)] = output
        features.clear()
        indices.clear()

    for i, (y, x, b) in enumerate(
        tqdm(coords, desc="RF inference", leave=False)
    ):
        y, x, b = map(int, (y, x, b))

        values, bad = extract_patch(
            test,
            y,
            x,
            b,
            config.RF_PATCH_Y,
            config.RF_PATCH_X,
            config.RF_PATCH_B,
            mask=mask,
        )

        bad |= ~np.isfinite(values)

        features.append(
            rf_feature(
                values,
                bad,
                band_mean,
                b,
            )
        )
        indices.append(i)

        if len(features) >= 4096:
            flush()

    flush()
    return predictions


def main():
    path = (
        config.MODELS_DIR
        / "random_forest"
        / "random_forest_model.joblib"
    )
    if not path.exists():
        raise FileNotFoundError(
            "Random Forest model not found. Run RF training first."
        )

    artifact = joblib.load(path)
    model = artifact["model"]
    band_mean = artifact["band_mean"]

    cube, _, split, _ = load_dataset()
    _, _, test = split_blocks(cube, split)

    rows = []
    mask_files = list_mask_files()
    total_masks = len(mask_files)

    for mask_index, mask_file in enumerate(
        tqdm(mask_files, desc="RF masks"),
        start=1,
    ):
        mask, meta = load_test_mask(mask_file)
        global_index, global_total = experiment_progress(mask_index, total_masks)
        print(
            f"[RF] Experiment Mask {global_index}/{global_total} "
            f"({100.0*global_index/max(global_total,1):5.1f}%) | "
            f"local {mask_index}/{total_masks} | {meta['scenario']} "
            f"S{meta['severity']} R{meta['repeat']}",
            flush=True,
        )
        stable_case_index = benchmark_case_index(mask_file)
        coords = select_eval_coords(
            mask,
            config.SEED + stable_case_index,
        )

        if len(coords) == 0:
            continue

        pred = predict(
            model,
            band_mean,
            test,
            mask,
            coords,
        )

        truth = np.asarray(test)[
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
        ]

        reg = regression_metrics(truth, pred)
        sam = spectral_angle_metrics(test, coords, pred)

        rows.append({
            "scenario": meta["scenario"],
            "severity": meta["severity"],
            "repeat": meta["repeat"],
            "method": "random_forest",
            **reg,
            **sam,
        })

    output = save_method_metrics("random_forest", rows)
    print("")
    print("Completed.")
    print(output)


if __name__ == "__main__":
    main()
