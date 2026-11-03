"""Train an RF regressor with multiscale spatial/spectral context.

Training samples and corruption masks are drawn exclusively from the training
acquisitions. All 10 scenario families and five severity levels are balanced.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'config.py').exists())
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import config
from src.evaluation_utils import load_global_training_data
from src.ml_common import band_stats
from src.enhanced_classical import (PATCH_SPATIAL, PATCH_SPECTRAL, FEATURE_DIM,
                                     RF_VARIANT, rf_multiscale_feature)
from src.training_corruptions import generate_training_mask, SCENARIOS


# Match the pixel dimensions in saturation_simulation.py. A training target
# is randomly positioned anywhere inside its synthetic missing block rather
# than always at the centre (the latter gave very few large-block edge cases).
BLOCK_SPATIAL = (5, 9, 15, 25, 41)
PATCH_BANDS = (1, 3, 7, 15, 31)
BLOCK_BANDS = (5, 11, 21, 41, 63)
WHOLE_SPATIAL = (1, 3, 5, 9, 15)


def rf_training_mask(rng, scenario, severity):
    shape = (PATCH_SPATIAL, PATCH_SPATIAL, PATCH_SPECTRAL)
    if scenario not in ("spatial_patch", "spatio_spectral_block", "whole_spectrum_pixels"):
        mask, _, _ = generate_training_mask(
            shape, rng, scenario=scenario, severity=severity,
            center=(24, 24), band_center=16)
        return mask
    if scenario == "whole_spectrum_pixels":
        side, band_width = WHOLE_SPATIAL[severity - 1], PATCH_SPECTRAL
    elif scenario == "spatial_patch":
        side, band_width = BLOCK_SPATIAL[severity - 1], PATCH_BANDS[severity - 1]
    else:
        side, band_width = BLOCK_SPATIAL[severity - 1], BLOCK_BANDS[severity - 1]
    # Offsets are with respect to the complete degradation block. Cropping at
    # the patch boundary simulates the observation window, not a smaller loss.
    y0 = 24 - int(rng.integers(side))
    x0 = 24 - int(rng.integers(side))
    b0 = 16 - int(rng.integers(band_width))
    mask = np.zeros(shape, dtype=bool)
    mask[max(0, y0):min(PATCH_SPATIAL, y0 + side),
         max(0, x0):min(PATCH_SPATIAL, x0 + side),
         max(0, b0):min(PATCH_SPECTRAL, b0 + band_width)] = True
    if not mask[24, 24, 16]:
        raise RuntimeError("RF training mask missed the supervised target")
    return mask


def main():
    train_cubes, _val_cubes, wavelengths = load_global_training_data()
    band_mean, _ = band_stats(train_cubes)
    rng = np.random.default_rng(config.SEED + 1842)
    samples = int(os.getenv('HSI_RF_MULTISCALE_SAMPLES', '60000'))
    trees = int(os.getenv('HSI_RF_MULTISCALE_TREES', '250'))
    if samples < 500 or trees < 10:
        raise ValueError('Expected >=500 samples and >=10 trees')
    X = np.empty((samples, FEATURE_DIM), dtype=np.float32)
    targets = np.empty(samples, dtype=np.float32)
    scenarios = list(SCENARIOS)
    filled = 0
    attempts = 0
    with tqdm(total=samples, desc='Multiscale RF training features') as progress:
        while filled < samples:
            attempts += 1
            if attempts > samples * 120:
                raise RuntimeError('Not enough finite training targets for RF')
            cube = train_cubes[int(rng.integers(len(train_cubes)))]
            h, w, bands = cube.shape
            if h < PATCH_SPATIAL or w < PATCH_SPATIAL or bands < PATCH_SPECTRAL:
                raise RuntimeError('Training acquisition is smaller than RF context window')
            y = int(rng.integers(24, h-24))
            x = int(rng.integers(24, w-24))
            b = int(rng.integers(16, bands-16))
            target = float(cube[y,x,b])
            if not np.isfinite(target):
                continue
            # Read a patch once; mask is synthetic and never modifies source data.
            yi = np.arange(y-24, y+25)
            xi = np.arange(x-24, x+25)
            bi = np.arange(b-16, b+17)
            values = np.asarray(cube[np.ix_(yi,xi,bi)], np.float32)
            scenario = scenarios[(filled // 5) % len(scenarios)]
            severity = 1 + (filled % 5)
            synthetic = rf_training_mask(rng, scenario, severity)
            bad = synthetic | ~np.isfinite(values)
            if not bad[24,24,16]:
                raise RuntimeError('Training corruption did not mask target voxel')
            X[filled] = rf_multiscale_feature(values,bad,band_mean,b)
            targets[filled] = target
            filled += 1
            progress.update(1)
    print('Multiscale RF training acquisitions:', ', '.join(config.TRAIN_DATASET_FOLDERS),flush=True)
    print(f'Features: {FEATURE_DIM}; samples: {samples}; trees: {trees}',flush=True)
    print('RF target: measured reflectance directly (no pre-inpainting or residual stage).', flush=True)
    model = RandomForestRegressor(
        n_estimators=trees, max_depth=28, min_samples_leaf=2, max_features=0.75,
        n_jobs=config.RF_N_JOBS, random_state=config.SEED + 1843)
    model.fit(X,targets)
    folder = config.MODELS_DIR / 'random_forest'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'random_forest_multiscale_model.joblib'
    payload = dict(model=model, band_mean=band_mean, variant=RF_VARIANT,
        patch_spatial=PATCH_SPATIAL, patch_spectral=PATCH_SPECTRAL,
        feature_dim=FEATURE_DIM, training_samples=samples, trees=trees,
        training_datasets=list(config.TRAIN_DATASET_FOLDERS),
        wavelengths_nm=np.asarray(wavelengths, np.float32))
    # Finish writing before replacing the checkpoint. Retain a copy of the
    # previous model so prior benchmark figures remain reproducible.
    temporary = output.with_name(output.name + '.tmp')
    joblib.dump(payload, temporary, compress=3)
    if output.exists():
        from shutil import copy2
        copy2(output, output.with_name('random_forest_multiscale_previous.joblib'))
    os.replace(temporary, output)
    print('Saved:', output, '| variant:', RF_VARIANT, flush=True)

if __name__ == '__main__':
    main()
