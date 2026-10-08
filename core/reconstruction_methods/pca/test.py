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
)


def reconstruct_pca(test, mask, coords, model, band_mean):
    predictions = np.full(len(coords), np.nan, dtype=np.float32)

    by_pixel = {}
    for index, (y, x, b) in enumerate(coords):
        by_pixel.setdefault((int(y), int(x)), []).append(
            (index, int(b))
        )

    for (y, x), items in tqdm(
        by_pixel.items(),
        total=len(by_pixel),
        desc="PCA spectra",
        leave=False,
    ):
        spectrum = np.asarray(test[y, x, :], dtype=np.float32).copy()
        missing = mask[y, x, :] | ~np.isfinite(spectrum)

        filled = spectrum.copy()
        filled[missing] = band_mean[missing]

        for _ in range(config.PCA_ITERATIONS):
            reconstructed = model.inverse_transform(
                model.transform(filled[None, :])
            )[0]
            filled[missing] = reconstructed[missing]

        for index, band in items:
            predictions[index] = filled[band]

    return predictions


def main():
    path = config.MODELS_DIR / "pca" / "pca_model.joblib"
    if not path.exists():
        raise FileNotFoundError("PCA model not found. Run PCA training first.")

    artifact = joblib.load(path)
    model = artifact["model"]
    band_mean = artifact["band_mean"]

    cube, _, split, _ = load_dataset()
    _, _, test = split_blocks(cube, split)

    rows = []
    mask_files = list_mask_files()
    total_masks = len(mask_files)

    for mask_index, mask_file in enumerate(
        tqdm(mask_files, desc="PCA masks"),
        start=1,
    ):
        mask, meta = load_test_mask(mask_file)
        print(f"[PCA] Mask {mask_index}/{total_masks} ({100.0*mask_index/max(total_masks,1):5.1f}%) | {meta['scenario']} S{meta['severity']} R{meta['repeat']}", flush=True)
        coords = select_eval_coords(
            mask,
            config.SEED + mask_index,
        )

        if len(coords) == 0:
            continue

        pred = reconstruct_pca(
            test,
            mask,
            coords,
            model,
            band_mean,
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
            "method": "pca",
            **reg,
            **sam,
        })

    output = save_method_metrics("pca", rows)
    print("")
    print("Completed.")
    print(output)


if __name__ == "__main__":
    main()
