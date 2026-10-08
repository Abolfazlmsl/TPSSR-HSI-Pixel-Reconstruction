import numpy as np


def _finite_fill(array):
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    if np.all(finite):
        return array
    fill = float(np.nanmedian(array)) if np.any(finite) else 0.0
    out = array.copy()
    out[~finite] = fill
    return out


def compute_difficulty_maps(test_cube, spatial_quantile=0.75, spectral_quantile=0.70, margin=24):
    # Spatial difficulty: gradient magnitude of a broadband reflectance image.
    sampled = np.asarray(test_cube[:, :, ::8], dtype=np.float32)
    image = np.nanmean(sampled, axis=2)
    image = _finite_fill(image)

    gy, gx = np.gradient(image)
    spatial_score = np.sqrt(gx * gx + gy * gy)

    valid = np.zeros_like(spatial_score, dtype=bool)
    if spatial_score.shape[0] > 2 * margin and spatial_score.shape[1] > 2 * margin:
        valid[margin:-margin, margin:-margin] = True
    else:
        valid[:] = True

    values = spatial_score[valid]
    threshold = np.quantile(values, spatial_quantile) if len(values) else 0.0
    challenging_pixels = np.argwhere(valid & (spatial_score >= threshold)).astype(np.int32)

    # Spectral difficulty: curvature of the mean spectrum.
    subsampled = np.asarray(test_cube[::4, ::4, :], dtype=np.float32)
    mean_spectrum = np.nanmean(subsampled, axis=(0, 1))
    mean_spectrum = _finite_fill(mean_spectrum)

    curvature = np.zeros_like(mean_spectrum, dtype=np.float32)
    curvature[1:-1] = np.abs(
        mean_spectrum[:-2] - 2.0 * mean_spectrum[1:-1] + mean_spectrum[2:]
    )

    band_margin = min(34, max(2, len(mean_spectrum) // 8))
    band_valid = np.zeros(len(mean_spectrum), dtype=bool)
    if len(mean_spectrum) > 2 * band_margin:
        band_valid[band_margin:-band_margin] = True
    else:
        band_valid[1:-1] = True

    band_values = curvature[band_valid]
    band_threshold = np.quantile(band_values, spectral_quantile) if len(band_values) else 0.0
    challenging_bands = np.flatnonzero(band_valid & (curvature >= band_threshold)).astype(np.int32)

    return {
        "spatial_score": spatial_score.astype(np.float32),
        "challenging_pixels": challenging_pixels,
        "mean_spectrum": mean_spectrum.astype(np.float32),
        "spectral_curvature": curvature.astype(np.float32),
        "challenging_bands": challenging_bands,
    }


def _hard_probability(severity):
    return [0.20, 0.40, 0.60, 0.80, 0.90][severity - 1]


def _choose_pixel(rng, shape, challenging_pixels, margin, severity):
    y_size, x_size, _ = shape
    hard_prob = _hard_probability(severity)

    if len(challenging_pixels) and rng.random() < hard_prob:
        for _ in range(50):
            y, x = map(int, challenging_pixels[int(rng.integers(0, len(challenging_pixels)))])
            if margin <= y < y_size - margin and margin <= x < x_size - margin:
                return y, x

    y = int(rng.integers(margin, max(margin + 1, y_size - margin)))
    x = int(rng.integers(margin, max(margin + 1, x_size - margin)))
    return y, x


def _choose_band_center(rng, n_bands, challenging_bands, half_width, severity):
    hard_prob = _hard_probability(severity)
    lo = half_width
    hi = n_bands - half_width

    candidates = challenging_bands[
        (challenging_bands >= lo) & (challenging_bands < hi)
    ]

    if len(candidates) and rng.random() < hard_prob:
        return int(candidates[int(rng.integers(0, len(candidates)))])

    return int(rng.integers(lo, max(lo + 1, hi)))


def _bounds(center, size, limit):
    size = max(1, min(int(size), int(limit)))
    half = size // 2
    start = center - half
    start = max(0, min(start, limit - size))
    return start, start + size


def _band_bounds(center, width, n_bands):
    return _bounds(center, width, n_bands)


def _add_detector_line(mask, rng, challenging_pixels, challenging_bands, severity, thickness=1, partial_fraction=None):
    y_size, x_size, n_bands = mask.shape
    _, x = _choose_pixel(rng, mask.shape, challenging_pixels, 1, severity)
    b = _choose_band_center(rng, n_bands, challenging_bands, 0, severity)

    x0, x1 = _bounds(x, thickness, x_size)

    if partial_fraction is None:
        mask[:, x0:x1, b] = True
        return

    length = max(1, int(round(partial_fraction * y_size)))
    y0 = int(rng.integers(0, max(1, y_size - length + 1)))
    mask[y0:y0 + length, x0:x1, b] = True


def _generate_component(shape, scenario, severity, rng, challenging_pixels, challenging_bands):
    mask = np.zeros(shape, dtype=bool)
    y_size, x_size, n_bands = shape

    if scenario == "isolated_voxels":
        counts = [1000, 5000, 15000, 40000, 100000]
        n = counts[severity - 1]

        hard_prob = _hard_probability(severity)
        for _ in range(n):
            if len(challenging_pixels) and rng.random() < hard_prob:
                y, x = map(int, challenging_pixels[int(rng.integers(0, len(challenging_pixels)))])
            else:
                y = int(rng.integers(0, y_size))
                x = int(rng.integers(0, x_size))

            if len(challenging_bands) and rng.random() < hard_prob:
                b = int(challenging_bands[int(rng.integers(0, len(challenging_bands)))])
            else:
                b = int(rng.integers(0, n_bands))

            mask[y, x, b] = True

    elif scenario == "single_detector_line":
        # A single detector failure must remain exactly one detector element
        # at every severity. Severity changes where the line is placed: higher
        # levels are increasingly biased toward spatially difficult columns and
        # high-curvature spectral regions.
        _add_detector_line(
            mask,
            rng,
            challenging_pixels,
            challenging_bands,
            severity,
            thickness=1,
        )

    elif scenario == "multiple_detector_lines":
        counts = [4, 8, 16, 32, 64]
        thicknesses = [1, 1, 2, 2, 3]
        for _ in range(counts[severity - 1]):
            _add_detector_line(
                mask,
                rng,
                challenging_pixels,
                challenging_bands,
                severity,
                thickness=thicknesses[severity - 1],
            )

    elif scenario == "partial_vertical_lines":
        counts = [6, 12, 24, 40, 64]
        fractions = [0.30, 0.45, 0.60, 0.75, 0.90]
        thicknesses = [1, 1, 2, 3, 4]
        for _ in range(counts[severity - 1]):
            _add_detector_line(
                mask,
                rng,
                challenging_pixels,
                challenging_bands,
                severity,
                thickness=thicknesses[severity - 1],
                partial_fraction=fractions[severity - 1],
            )

    elif scenario == "spatial_patch":
        patch_sizes = [5, 9, 15, 25, 41]
        band_widths = [1, 3, 7, 15, 31]
        counts = [4, 6, 8, 10, 12]

        size = patch_sizes[severity - 1]
        band_width = band_widths[severity - 1]
        margin = size // 2 + 1

        for _ in range(counts[severity - 1]):
            y, x = _choose_pixel(rng, shape, challenging_pixels, margin, severity)
            b = _choose_band_center(
                rng, n_bands, challenging_bands, band_width // 2 + 1, severity
            )
            y0, y1 = _bounds(y, size, y_size)
            x0, x1 = _bounds(x, size, x_size)
            b0, b1 = _band_bounds(b, band_width, n_bands)
            mask[y0:y1, x0:x1, b0:b1] = True

    elif scenario == "consecutive_spectral_bands":
        widths = [3, 7, 15, 31, 63]
        pixel_counts = [100, 250, 500, 1000, 2000]
        width = widths[severity - 1]

        for _ in range(pixel_counts[severity - 1]):
            y, x = _choose_pixel(rng, shape, challenging_pixels, 1, severity)
            b = _choose_band_center(
                rng, n_bands, challenging_bands, width // 2 + 1, severity
            )
            b0, b1 = _band_bounds(b, width, n_bands)
            mask[y, x, b0:b1] = True

    elif scenario == "spatio_spectral_block":
        spatial_sizes = [5, 9, 15, 25, 41]
        band_widths = [5, 11, 21, 41, 63]
        counts = [2, 3, 4, 5, 6]

        size = spatial_sizes[severity - 1]
        width = band_widths[severity - 1]
        margin = size // 2 + 1

        for _ in range(counts[severity - 1]):
            y, x = _choose_pixel(rng, shape, challenging_pixels, margin, severity)
            b = _choose_band_center(
                rng, n_bands, challenging_bands, width // 2 + 1, severity
            )
            y0, y1 = _bounds(y, size, y_size)
            x0, x1 = _bounds(x, size, x_size)
            b0, b1 = _band_bounds(b, width, n_bands)
            mask[y0:y1, x0:x1, b0:b1] = True

    elif scenario == "whole_spectrum_pixels":
        patch_sizes = [1, 3, 5, 9, 15]
        counts = [10, 8, 6, 4, 3]
        size = patch_sizes[severity - 1]
        margin = size // 2 + 1

        for _ in range(counts[severity - 1]):
            y, x = _choose_pixel(rng, shape, challenging_pixels, margin, severity)
            y0, y1 = _bounds(y, size, y_size)
            x0, x1 = _bounds(x, size, x_size)
            mask[y0:y1, x0:x1, :] = True

    elif scenario == "band_wide_saturation":
        widths = [1, 3, 7, 15, 31]
        width = widths[severity - 1]
        b = _choose_band_center(
            rng, n_bands, challenging_bands, width // 2 + 1, severity
        )
        b0, b1 = _band_bounds(b, width, n_bands)
        mask[:, :, b0:b1] = True

    elif scenario == "mixed_saturation":
        # Mixed corruption is staged deliberately so that each severity adds
        # new failure modes instead of being dominated by a full-band loss at
        # low severity.
        staged_components = {
            1: [
                ("isolated_voxels", 1),
                ("multiple_detector_lines", 1),
            ],
            2: [
                ("isolated_voxels", 2),
                ("multiple_detector_lines", 2),
                ("partial_vertical_lines", 1),
                ("consecutive_spectral_bands", 1),
            ],
            3: [
                ("isolated_voxels", 3),
                ("multiple_detector_lines", 3),
                ("partial_vertical_lines", 2),
                ("consecutive_spectral_bands", 2),
                ("spatial_patch", 2),
            ],
            4: [
                ("isolated_voxels", 4),
                ("multiple_detector_lines", 4),
                ("partial_vertical_lines", 3),
                ("consecutive_spectral_bands", 3),
                ("spatial_patch", 3),
                ("spatio_spectral_block", 2),
                ("whole_spectrum_pixels", 2),
            ],
            5: [
                ("isolated_voxels", 5),
                ("multiple_detector_lines", 5),
                ("partial_vertical_lines", 5),
                ("consecutive_spectral_bands", 5),
                ("spatial_patch", 5),
                ("spatio_spectral_block", 5),
                ("whole_spectrum_pixels", 5),
                # A complete-band component is introduced only at the most
                # severe mixed level. Seven neighboring full bands are used so
                # the case is genuinely difficult without masking most of the
                # spectral cube.
                ("band_wide_saturation", 3),
            ],
        }

        for component, comp_severity in staged_components[severity]:
            sub = _generate_component(
                shape,
                component,
                comp_severity,
                rng,
                challenging_pixels,
                challenging_bands,
            )
            mask |= sub

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return mask


def generate_saturation_mask(shape, scenario, severity, rng, challenging_pixels, challenging_bands):
    severity = int(severity)
    if severity < 1 or severity > 5:
        raise ValueError("Severity must be between 1 and 5.")

    return _generate_component(
        shape,
        scenario,
        severity,
        rng,
        challenging_pixels,
        challenging_bands,
    )
