import numpy as np


def _nearest_valid_indices(valid):
    n = len(valid)
    idx = np.arange(n, dtype=np.int32)

    left = np.where(valid, idx, -n - 1)
    left = np.maximum.accumulate(left)

    right = np.where(valid, idx, 2 * n + 1)
    right = np.minimum.accumulate(right[::-1])[::-1]

    return left, right


def build_sparse_router_features(
    patch,
    missing_mask,
    band_mean,
    band_std,
    wavelengths_nm,
):
    """
    Build normalized model input plus two analytical reconstruction experts.

    Expert 0: wavelength-aware spectral linear interpolation at the center pixel.
    Expert 1: distance-weighted spatial interpolation within the patch.

    No scenario label is used. Availability, confidence, and mask-topology
    descriptors are derived only from the observed patch and its missing mask.
    """
    patch = np.asarray(patch, dtype=np.float32)
    missing_mask = np.asarray(missing_mask, dtype=bool)
    band_mean = np.asarray(band_mean, dtype=np.float32)
    band_std = np.asarray(band_std, dtype=np.float32)
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float32)

    py, px, n_bands = patch.shape
    cy, cx = py // 2, px // 2

    natural_bad = ~np.isfinite(patch)
    bad = natural_bad | missing_mask

    normalized = (
        patch - band_mean[None, None, :]
    ) / band_std[None, None, :]
    normalized[bad] = 0.0

    center_values = normalized[cy, cx, :].copy()
    center_bad = bad[cy, cx, :].copy()

    experts = np.zeros((n_bands, 2), dtype=np.float32)
    availability = np.zeros((n_bands, 2), dtype=np.float32)
    confidence = np.zeros((n_bands, 2), dtype=np.float32)

    # ------------------------------------------------------------------
    # Spectral analytical expert.
    # ------------------------------------------------------------------
    spectral_valid = ~center_bad
    valid_idx = np.flatnonzero(spectral_valid)

    if len(valid_idx) >= 1:
        experts[:, 0] = np.interp(
            wavelengths_nm,
            wavelengths_nm[valid_idx],
            center_values[valid_idx],
        ).astype(np.float32)
        availability[:, 0] = 1.0

        left, right = _nearest_valid_indices(spectral_valid)
        idx = np.arange(n_bands, dtype=np.int32)

        left_ok = left >= 0
        right_ok = right < n_bands
        both = left_ok & right_ok
        one_side = left_ok ^ right_ok

        dl = np.where(left_ok, idx - left, n_bands).astype(np.float32)
        dr = np.where(right_ok, right - idx, n_bands).astype(np.float32)
        nearest = np.minimum(dl, dr)

        spectral_conf = np.exp(-nearest / 12.0)
        spectral_conf[one_side] *= 0.55
        spectral_conf[~(left_ok | right_ok)] = 0.0
        spectral_conf[spectral_valid] = 1.0
        confidence[:, 0] = spectral_conf.astype(np.float32)
    else:
        left = np.full(n_bands, -n_bands - 1, dtype=np.int32)
        right = np.full(n_bands, 2 * n_bands + 1, dtype=np.int32)

    # ------------------------------------------------------------------
    # Spatial analytical expert.
    # A fixed Gaussian-like radial kernel keeps nearby intact pixels more
    # influential while still reaching the edge of the 31x31 context window.
    # ------------------------------------------------------------------
    yy, xx = np.mgrid[0:py, 0:px]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
    spatial_weights = np.exp(-(dist ** 2) / (2.0 * 7.0 ** 2)).astype(np.float32)
    spatial_weights[cy, cx] = 0.0

    valid_spatial = ~bad
    weighted_valid = spatial_weights[:, :, None] * valid_spatial.astype(np.float32)
    denom = weighted_valid.sum(axis=(0, 1))
    numer = (normalized * weighted_valid).sum(axis=(0, 1))

    spatial_available = denom > 1e-8
    experts[spatial_available, 1] = (
        numer[spatial_available] / denom[spatial_available]
    ).astype(np.float32)
    availability[spatial_available, 1] = 1.0

    flat_dist = dist.reshape(-1, 1)
    valid_flat = valid_spatial.reshape(-1, n_bands)
    nearest_spatial = np.where(valid_flat, flat_dist, np.inf).min(axis=0)
    valid_count = valid_spatial.sum(axis=(0, 1)).astype(np.float32)

    spatial_conf = (
        np.exp(-nearest_spatial / 6.0)
        * np.clip(valid_count / 32.0, 0.0, 1.0)
    )
    spatial_conf[~spatial_available] = 0.0
    spatial_conf[~center_bad] = 1.0
    confidence[:, 1] = spatial_conf.astype(np.float32)

    # Observed center bands are exact analytical anchors.
    experts[~center_bad, 0] = center_values[~center_bad]
    experts[~center_bad, 1] = center_values[~center_bad]
    availability[~center_bad, :] = 1.0
    confidence[~center_bad, :] = 1.0

    # ------------------------------------------------------------------
    # Mask-topology descriptors, one vector per spectral band.
    # ------------------------------------------------------------------
    spatial_missing_fraction = bad.mean(axis=(0, 1)).astype(np.float32)
    center_missing_fraction = np.float32(center_bad.mean())

    if len(valid_idx) >= 1:
        idx = np.arange(n_bands, dtype=np.int32)
        left_ok = left >= 0
        right_ok = right < n_bands

        dl = np.where(left_ok, idx - left, n_bands).astype(np.float32)
        dr = np.where(right_ok, right - idx, n_bands).astype(np.float32)
        nearest_spectral = np.minimum(dl, dr)

        gap_span = np.zeros(n_bands, dtype=np.float32)
        both = left_ok & right_ok
        gap_span[both] = (right[both] - left[both]).astype(np.float32)
        only_left = left_ok & ~right_ok
        gap_span[only_left] = dl[only_left]
        only_right = right_ok & ~left_ok
        gap_span[only_right] = dr[only_right]
        gap_span[~center_bad] = 0.0
        nearest_spectral[~center_bad] = 0.0
    else:
        nearest_spectral = np.full(n_bands, n_bands, dtype=np.float32)
        gap_span = np.full(n_bands, n_bands, dtype=np.float32)

    spatial_nearest_norm = np.where(
        np.isfinite(nearest_spatial),
        nearest_spatial / max(float(np.hypot(py - 1, px - 1)), 1.0),
        1.0,
    ).astype(np.float32)

    geometry = np.stack(
        [
            center_bad.astype(np.float32),
            spatial_missing_fraction,
            np.clip(gap_span / max(float(n_bands), 1.0), 0.0, 1.0),
            np.clip(nearest_spectral / max(float(n_bands), 1.0), 0.0, 1.0),
            np.clip(spatial_nearest_norm, 0.0, 1.0),
            np.full(n_bands, center_missing_fraction, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    value_channels = normalized.transpose(2, 0, 1)
    mask_channels = bad.astype(np.float32).transpose(2, 0, 1)
    model_input = np.concatenate(
        [value_channels, mask_channels],
        axis=0,
    ).astype(np.float32)

    return {
        "input": model_input,
        "experts": experts,
        "availability": availability,
        "confidence": confidence,
        "geometry": geometry,
        "center_missing": center_bad.astype(np.float32),
    }
