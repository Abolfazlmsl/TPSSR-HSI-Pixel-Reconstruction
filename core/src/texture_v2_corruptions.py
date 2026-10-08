import numpy as np


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


def _bounds(center, size, limit):
    size = max(1, min(int(size), int(limit)))
    start = max(0, min(int(center) - size // 2, int(limit) - size))
    return start, start + size


def _band_bounds(center, width, n_bands):
    return _bounds(center, width, n_bands)


def _add_line(mask, rng, x, band, thickness=1, fraction=1.0, centered=True):
    py, px, pb = mask.shape
    thickness = max(1, int(thickness))
    x0, x1 = _bounds(x, thickness, px)
    if fraction >= 0.999:
        y0, y1 = 0, py
    else:
        length = max(1, int(round(py * float(fraction))))
        if centered:
            y0, y1 = _bounds(py // 2, length, py)
        else:
            y0 = int(rng.integers(0, max(1, py - length + 1)))
            y1 = y0 + length
    mask[y0:y1, x0:x1, int(np.clip(band, 0, pb - 1))] = True


def _single_component(shape, scenario, severity, rng, force_center=True):
    py, px, pb = map(int, shape)
    cy, cx, cb = py // 2, px // 2, pb // 2
    severity = int(np.clip(severity, 1, 5))
    mask = np.zeros(shape, dtype=bool)

    if scenario == "isolated_voxels":
        counts = [2, 6, 16, 40, 100]
        n = min(counts[severity - 1], py * px * pb)
        if force_center:
            mask[cy, cx, cb] = True
            n -= 1
        if n > 0:
            ys = rng.integers(0, py, size=n)
            xs = rng.integers(0, px, size=n)
            bs = rng.integers(0, pb, size=n)
            mask[ys, xs, bs] = True

    elif scenario == "single_detector_line":
        _add_line(mask, rng, cx, cb, thickness=1, fraction=1.0)

    elif scenario == "multiple_detector_lines":
        counts = [4, 8, 16, 32, 64]
        thicknesses = [1, 1, 2, 2, 3]
        _add_line(mask, rng, cx, cb, thickness=thicknesses[severity - 1])
        for _ in range(counts[severity - 1] - 1):
            _add_line(
                mask,
                rng,
                int(rng.integers(0, px)),
                int(rng.integers(0, pb)),
                thickness=thicknesses[severity - 1],
            )

    elif scenario == "partial_vertical_lines":
        counts = [6, 12, 24, 40, 64]
        fractions = [0.30, 0.45, 0.60, 0.75, 0.90]
        thicknesses = [1, 1, 2, 3, 4]
        _add_line(
            mask, rng, cx, cb,
            thickness=thicknesses[severity - 1],
            fraction=fractions[severity - 1],
            centered=True,
        )
        for _ in range(counts[severity - 1] - 1):
            _add_line(
                mask,
                rng,
                int(rng.integers(0, px)),
                int(rng.integers(0, pb)),
                thickness=thicknesses[severity - 1],
                fraction=fractions[severity - 1],
                centered=bool(rng.integers(0, 2)),
            )

    elif scenario == "spatial_patch":
        sizes = [5, 9, 15, 25, 41]
        widths = [1, 3, 7, 15, 31]
        sy = min(sizes[severity - 1], py)
        sx = min(sizes[severity - 1], px)
        bw = min(widths[severity - 1], pb)
        y0, y1 = _bounds(cy, sy, py)
        x0, x1 = _bounds(cx, sx, px)
        b0, b1 = _band_bounds(cb, bw, pb)
        mask[y0:y1, x0:x1, b0:b1] = True
        # One extra off-center patch provides context similar to multi-patch benchmark masks.
        if severity >= 3 and rng.random() < 0.5:
            ey = int(rng.integers(0, py))
            ex = int(rng.integers(0, px))
            eb = int(rng.integers(0, pb))
            ey0, ey1 = _bounds(ey, sy, py)
            ex0, ex1 = _bounds(ex, sx, px)
            eb0, eb1 = _band_bounds(eb, bw, pb)
            mask[ey0:ey1, ex0:ex1, eb0:eb1] = True

    elif scenario == "consecutive_spectral_bands":
        widths = [3, 7, 15, 31, 63]
        bw = min(widths[severity - 1], pb)
        b0, b1 = _band_bounds(cb, bw, pb)
        mask[cy, cx, b0:b1] = True

    elif scenario == "spatio_spectral_block":
        sizes = [5, 9, 15, 25, 41]
        widths = [5, 11, 21, 41, 63]
        sy = min(sizes[severity - 1], py)
        sx = min(sizes[severity - 1], px)
        bw = min(widths[severity - 1], pb)
        y0, y1 = _bounds(cy, sy, py)
        x0, x1 = _bounds(cx, sx, px)
        b0, b1 = _band_bounds(cb, bw, pb)
        mask[y0:y1, x0:x1, b0:b1] = True

    elif scenario == "whole_spectrum_pixels":
        sizes = [1, 3, 5, 9, 15]
        size = min(sizes[severity - 1], py, px)
        y0, y1 = _bounds(cy, size, py)
        x0, x1 = _bounds(cx, size, px)
        mask[y0:y1, x0:x1, :] = True

    elif scenario == "band_wide_saturation":
        widths = [1, 3, 7, 15, 31]
        bw = min(widths[severity - 1], pb)
        b0, b1 = _band_bounds(cb, bw, pb)
        mask[:, :, b0:b1] = True

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    if force_center:
        mask[cy, cx, cb] = True
    return mask


def generate_texture_v2_training_mask(shape, rng, severity=None):
    if severity is None:
        severity = int(rng.integers(1, 6))
    severity = int(np.clip(severity, 1, 5))
    scenario = SCENARIOS[int(rng.integers(0, len(SCENARIOS)))]

    if scenario != "mixed_saturation":
        return _single_component(shape, scenario, severity, rng, force_center=True), scenario, severity

    # Progressive mixed case. Components and severities mirror the benchmark families,
    # while every local training crop is guaranteed to contain a corrupted center voxel.
    staged = {
        1: [("isolated_voxels", 1), ("multiple_detector_lines", 1)],
        2: [("isolated_voxels", 2), ("multiple_detector_lines", 2),
            ("partial_vertical_lines", 1), ("consecutive_spectral_bands", 1)],
        3: [("isolated_voxels", 3), ("multiple_detector_lines", 3),
            ("partial_vertical_lines", 2), ("consecutive_spectral_bands", 2),
            ("spatial_patch", 2)],
        4: [("isolated_voxels", 4), ("multiple_detector_lines", 4),
            ("partial_vertical_lines", 3), ("consecutive_spectral_bands", 3),
            ("spatial_patch", 3), ("spatio_spectral_block", 2),
            ("whole_spectrum_pixels", 2)],
        5: [("isolated_voxels", 5), ("multiple_detector_lines", 5),
            ("partial_vertical_lines", 5), ("consecutive_spectral_bands", 5),
            ("spatial_patch", 5), ("spatio_spectral_block", 5),
            ("whole_spectrum_pixels", 5), ("band_wide_saturation", 3)],
    }
    mask = np.zeros(shape, dtype=bool)
    for component, component_severity in staged[severity]:
        mask |= _single_component(
            shape, component, component_severity, rng, force_center=False
        )
    cy, cx, cb = shape[0] // 2, shape[1] // 2, shape[2] // 2
    mask[cy, cx, cb] = True
    return mask, scenario, severity


def severity_for_epoch_v2(epoch, max_epochs, rng):
    progress = float(epoch) / max(float(max_epochs), 1.0)
    if progress < 0.12:
        probabilities = np.array([0.42, 0.32, 0.18, 0.08, 0.00])
    elif progress < 0.35:
        probabilities = np.array([0.20, 0.28, 0.27, 0.18, 0.07])
    elif progress < 0.60:
        probabilities = np.array([0.10, 0.18, 0.28, 0.27, 0.17])
    else:
        probabilities = np.array([0.06, 0.12, 0.22, 0.30, 0.30])
    return int(rng.choice(np.arange(1, 6), p=probabilities))
