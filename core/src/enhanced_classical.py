"""Mask-aware spatial PCA and multiscale RF features.

Only the observed values of the input cube contribute to either method.
The target values are never used to fill input features or priors.
"""
from __future__ import annotations

import numpy as np

PATCH_SPATIAL = 49
PATCH_SPECTRAL = 33
BAND_OFFSETS = np.array([-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16], dtype=np.int32)
RING_RADII = ((0, 2), (2, 5), (5, 11), (11, 24))
_Y, _X = np.ogrid[-24:25, -24:25]
_DIST = np.maximum(np.abs(_Y), np.abs(_X))
RINGS = tuple((_DIST > low) & (_DIST <= high) for low, high in RING_RADII)
RINGS = (RINGS[0] & (_DIST > 0),) + RINGS[1:]
# Directional context in the outer ring distinguishes a continuing sediment
# layer from four equally bright, but differently arranged, surroundings.
_OUTER = RINGS[-1]
DIRECTIONAL_RINGS = (
    _OUTER & (_Y < 0) & (np.abs(_Y) >= np.abs(_X)),  # north
    _OUTER & (_X > 0) & (np.abs(_X) > np.abs(_Y)),   # east
    _OUTER & (_Y > 0) & (np.abs(_Y) >= np.abs(_X)),  # south
    _OUTER & (_X < 0) & (np.abs(_X) > np.abs(_Y)),   # west
)
# RF receives measured donor values as features; no preliminary estimate of a
# missing voxel is computed. First and second observed samples on each ray
# supply local direction and gradient information, including inside large gaps.
RAY_DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1), (-1, -1))
RAY_LENGTH = PATCH_SPATIAL // 2
RAY_STEPS = np.arange(1, RAY_LENGTH + 1, dtype=np.int32)
RAY_COORDS = tuple((24 + dy * RAY_STEPS, 24 + dx * RAY_STEPS)
                   for dy, dx in RAY_DIRECTIONS)
RAY_FEATURES_PER_DIRECTION = 6  # measured value, distance, presence x 2
RF_VARIANT = "multiscale_rf_v3_observed_donors41"
FEATURE_DIM = (len(BAND_OFFSETS) *
               (2 + len(RINGS) * 3 + len(DIRECTIONAL_RINGS) * 2 +
                len(RAY_DIRECTIONS) * RAY_FEATURES_PER_DIRECTION) + 3)


def spatial_spectral_prior(cube, mask, y, x, band_mean, *, radii=(3, 7, 14, 26)):
    """Use only observed spectra and spatial neighbours for a missing pixel."""
    spectrum = np.asarray(cube[y, x], dtype=np.float32)
    bad = np.asarray(mask[y, x], bool) | ~np.isfinite(spectrum)
    n_bands = len(spectrum)
    prior = np.asarray(band_mean, np.float32).copy()
    if len(prior) != n_bands:
        raise ValueError('PCA model and input have different spectral grids')
    good_bands = np.flatnonzero(~bad)
    if len(good_bands) >= 2:
        spectral = np.interp(np.arange(n_bands), good_bands, spectrum[good_bands]).astype(np.float32)
    elif len(good_bands) == 1:
        spectral = np.full(n_bands, spectrum[good_bands[0]], dtype=np.float32)
    else:
        spectral = None

    need = bad.copy()
    spatial = np.empty(n_bands, dtype=np.float32)
    have = np.zeros(n_bands, dtype=bool)
    H, W, _ = cube.shape
    # Expand context only as far as necessary; do not read hidden truth.
    for radius in radii:
        if not np.any(need):
            break
        y0, y1 = max(0, y-radius), min(H, y+radius+1)
        x0, x1 = max(0, x-radius), min(W, x+radius+1)
        values = np.asarray(cube[y0:y1, x0:x1, :], dtype=np.float32)
        invalid = np.asarray(mask[y0:y1, x0:x1, :], bool) | ~np.isfinite(values)
        chosen = np.flatnonzero(need)
        vals = values[:, :, chosen]
        valid = ~invalid[:, :, chosen]
        count = valid.sum(axis=(0, 1))
        possible = count >= 3
        if np.any(possible):
            idx = chosen[possible]
            means = np.sum(np.where(valid[:, :, possible], vals[:, :, possible], 0.0),
                           axis=(0, 1), dtype=np.float64) / count[possible]
            spatial[idx] = means.astype(np.float32)
            have[idx] = True
            need[idx] = False
    if spectral is not None:
        prior[bad] = spectral[bad]
        both = bad & have
        prior[both] = 0.60 * spectral[both] + 0.40 * spatial[both]
    else:
        prior[have] = spatial[have]
    prior[~bad] = spectrum[~bad]
    return prior, bad, bool(len(good_bands) >= 2)


def reconstruct_spatial_pca(cube, mask, coords, model, band_mean, iterations=10):
    """Project spatially initialized missing spectra onto a fitted PCA subspace."""
    prediction = np.full(len(coords), np.nan, dtype=np.float32)
    by_pixel = {}
    for i, (y, x, b) in enumerate(coords):
        by_pixel.setdefault((int(y), int(x)), []).append((i, int(b)))
    if model.n_features_in_ != cube.shape[2]:
        raise ValueError('PCA checkpoint band count differs from image')
    for (y, x), cases in by_pixel.items():
        prior, bad, has_spectral = spatial_spectral_prior(cube, mask, y, x, band_mean)
        estimate = prior.copy()
        weight = 0.30 if has_spectral else 0.80
        for _ in range(int(iterations)):
            projected = model.inverse_transform(model.transform(estimate[None]))[0]
            estimate[bad] = ((1.0 - weight) * projected[bad] + weight * prior[bad])
        for i, b in cases:
            prediction[i] = estimate[b]
    return prediction


def rf_multiscale_feature(values, mask, band_mean, center_band):
    """Mask-aware multiscale and directional statistics from observed voxels only."""
    values = np.asarray(values, dtype=np.float32)
    invalid = np.asarray(mask, bool) | ~np.isfinite(values)
    if values.shape != (PATCH_SPATIAL, PATCH_SPATIAL, PATCH_SPECTRAL):
        raise ValueError(f'Unexpected RF patch shape {values.shape}')
    band_mean = np.asarray(band_mean, np.float32)
    bands = np.clip(center_band + BAND_OFFSETS, 0, len(band_mean) - 1)
    v = values[:, :, BAND_OFFSETS + PATCH_SPECTRAL // 2]
    m = invalid[:, :, BAND_OFFSETS + PATCH_SPECTRAL // 2].copy()
    m[PATCH_SPATIAL//2, PATCH_SPATIAL//2, BAND_OFFSETS.tolist().index(0)] = True
    # The center voxel must always be hidden, even in a training sample.
    f = []
    for j, b in enumerate(bands):
        center_bad = bool(m[24, 24, j])
        f.extend((float(band_mean[b] if center_bad else v[24,24,j]), float(center_bad)))
        for ring in RINGS:
            selected = ring & ~m[:, :, j]
            vv = v[:, :, j][selected]
            if len(vv):
                f.extend((float(vv.mean()), float(vv.std()), float(len(vv) / int(ring.sum()))))
            else:
                f.extend((float(band_mean[b]), 0., 0.))
        # Long-range, orientation-sensitive support matters for a 41x41 loss:
        # the first three rings can be entirely missing at its centre.
        for sector in DIRECTIONAL_RINGS:
            selected = sector & ~m[:, :, j]
            vv = v[:, :, j][selected]
            if len(vv):
                f.extend((float(vv.mean()), float(len(vv) / int(sector.sum()))))
            else:
                f.extend((float(band_mean[b]), 0.))
        # Raw observed donor readings in eight directions: no interpolation,
        # no pseudo-target, and no prediction from another model. The RF
        # learns how to use these measurements to predict the missing value.
        for ray_y, ray_x in RAY_COORDS:
            observed = ~m[ray_y, ray_x, j]
            donor_steps = np.flatnonzero(observed)[:2]
            for k in range(2):
                if k < len(donor_steps):
                    index = int(donor_steps[k])
                    f.extend((float(v[ray_y[index], ray_x[index], j]),
                              float((index + 1) / RAY_LENGTH), 1.0))
                else:
                    # Placeholder is always paired with presence=0, so the
                    # forest can distinguish absent donors from observations.
                    f.extend((float(band_mean[b]), 1.0, 0.0))
    center_valid = np.isfinite(values[24,24,:]) & ~invalid[24,24,:]
    f.extend((float(center_band / max(1, len(band_mean)-1)),
              float(np.mean(center_valid)), float(np.mean(invalid))))
    feature = np.asarray(f, np.float32)
    if len(feature) != FEATURE_DIM or not np.isfinite(feature).all():
        raise RuntimeError(f'Invalid RF feature vector: {feature.shape}, expected {FEATURE_DIM}')
    return feature


def extract_rf_patch(cube, mask, y, x, b):
    yi = np.clip(np.arange(int(y)-24, int(y)+25), 0, cube.shape[0]-1)
    xi = np.clip(np.arange(int(x)-24, int(x)+25), 0, cube.shape[1]-1)
    bi = np.clip(np.arange(int(b)-16, int(b)+17), 0, cube.shape[2]-1)
    values = np.asarray(cube[np.ix_(yi, xi, bi)], np.float32)
    invalid = np.asarray(mask[np.ix_(yi, xi, bi)], bool)
    return values, invalid


def predict_multiscale_rf(model, band_mean, cube, mask, coords, batch=2048):
    pred = np.empty(len(coords), np.float32)
    X = np.empty((min(batch, max(len(coords), 1)), FEATURE_DIM), np.float32)
    for start in range(0, len(coords), batch):
        end = min(len(coords), start+batch)
        for i, (y,x,b) in enumerate(coords[start:end]):
            v, bad = extract_rf_patch(cube, mask, int(y), int(x), int(b))
            X[i] = rf_multiscale_feature(v, bad, band_mean, int(b))
        pred[start:end] = model.predict(X[:end-start]).astype(np.float32)
    return pred
