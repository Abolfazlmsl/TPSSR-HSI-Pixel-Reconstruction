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
    start = max(0, min(center - size // 2, limit - size))
    return start, start + size


def _band_block(mask, cy, cx, width, center_band=None):
    pb = mask.shape[2]
    cb = pb // 2 if center_band is None else int(center_band)
    b0, b1 = _bounds(cb, width, pb)
    mask[cy, cx, b0:b1] = True


def generate_training_mask(shape, rng, severity=None, scenario=None, center=None, band_center=None):
    py, px, pb = map(int, shape)
    cy, cx = (py // 2, px // 2) if center is None else tuple(map(int, center))
    if not (0 <= cy < py and 0 <= cx < px):
        raise ValueError("Corruption center outside patch")
    cb = pb // 2 if band_center is None else int(band_center)
    if not 0 <= cb < pb:
        raise ValueError("Corruption band center outside spectrum")

    if severity is None:
        severity = int(rng.integers(1, 6))
    severity = int(np.clip(severity, 1, 5))

    scenario = (SCENARIOS[int(rng.integers(0, len(SCENARIOS)))]
                if scenario is None else str(scenario))
    if scenario not in SCENARIOS:
        raise ValueError(f"Unsupported training scenario {scenario}")
    mask = np.zeros(shape, dtype=bool)

    if scenario == "isolated_voxels":
        counts = [1, 2, 4, 8, 16]
        bands = rng.choice(pb, size=min(counts[severity - 1], pb), replace=False)
        mask[cy, cx, bands] = True

    elif scenario == "single_detector_line":
        # Preserve the physical meaning of one failed detector element.
        mask[:, cx, cb] = True

    elif scenario == "multiple_detector_lines":
        counts = [2, 3, 5, 7, 10]
        mask[:, cx, cb] = True
        for _ in range(counts[severity - 1] - 1):
            x = int(rng.integers(0, px))
            b = int(rng.integers(0, pb))
            mask[:, x, b] = True

    elif scenario == "partial_vertical_lines":
        fractions = [0.30, 0.45, 0.60, 0.80, 1.00]
        n = [2, 3, 5, 7, 10][severity - 1]
        length = max(1, int(round(py * fractions[severity - 1])))
        y0 = max(0, min(cy - length // 2, py - length))
        for i in range(n):
            x = cx if i == 0 else int(rng.integers(0, px))
            b = cb if i == 0 else int(rng.integers(0, pb))
            mask[y0:y0 + length, x, b] = True

    elif scenario == "spatial_patch":
        sizes = [3, 7, 11, 19, 27]
        widths = [1, 3, 7, 15, 31]
        sy = min(sizes[severity - 1], py)
        sx = min(sizes[severity - 1], px)
        bw = min(widths[severity - 1], pb)
        y0, y1 = _bounds(cy, sy, py)
        x0, x1 = _bounds(cx, sx, px)
        b0, b1 = _bounds(cb, bw, pb)
        mask[y0:y1, x0:x1, b0:b1] = True

    elif scenario == "consecutive_spectral_bands":
        widths = [3, 7, 15, 31, 63]
        _band_block(mask, cy, cx, min(widths[severity - 1], pb), cb)

    elif scenario == "spatio_spectral_block":
        sizes = [3, 7, 11, 19, 27]
        widths = [5, 11, 21, 41, 63]
        sy = min(sizes[severity - 1], py)
        sx = min(sizes[severity - 1], px)
        bw = min(widths[severity - 1], pb)
        y0, y1 = _bounds(cy, sy, py)
        x0, x1 = _bounds(cx, sx, px)
        b0, b1 = _bounds(cb, bw, pb)
        mask[y0:y1, x0:x1, b0:b1] = True

    elif scenario == "whole_spectrum_pixels":
        sizes = [1, 3, 7, 11, 19]
        size = min(sizes[severity - 1], py, px)
        y0, y1 = _bounds(cy, size, py)
        x0, x1 = _bounds(cx, size, px)
        mask[y0:y1, x0:x1, :] = True

    elif scenario == "band_wide_saturation":
        widths = [1, 3, 7, 15, 31]
        bw = min(widths[severity - 1], pb)
        b0, b1 = _bounds(cb, bw, pb)
        mask[:, :, b0:b1] = True

    elif scenario == "mixed_saturation":
        # Staged local mixture that mirrors the evaluation benchmark.
        # The center voxel is always affected, while new corruption families
        # are added progressively from severity 1 to 5.
        mask[:, cx, cb] = True

        if severity >= 1:
            # Sparse spectral defects at the target pixel.
            n_sparse = [2, 4, 8, 12, 16][severity - 1]
            bands = rng.choice(pb, size=min(n_sparse, pb), replace=False)
            mask[cy, cx, bands] = True

        if severity >= 2:
            # Additional detector elements plus a short spectral gap.
            n_lines = [0, 2, 4, 6, 10][severity - 1]
            for _ in range(n_lines):
                x = int(rng.integers(0, px))
                b = int(rng.integers(0, pb))
                mask[:, x, b] = True

            gap_width = min([1, 3, 7, 15, 31][severity - 1], pb)
            b0, b1 = _bounds(cb, gap_width, pb)
            mask[cy, cx, b0:b1] = True

        if severity >= 3:
            # Compact spatial-spectral patch.
            size = min([1, 1, 7, 13, 21][severity - 1], py, px)
            width = min([1, 1, 7, 21, 41][severity - 1], pb)
            y0, y1 = _bounds(cy, size, py)
            x0, x1 = _bounds(cx, size, px)
            b0, b1 = _bounds(cb, width, pb)
            mask[y0:y1, x0:x1, b0:b1] = True

        if severity >= 4:
            # Whole-spectrum loss over a local spatial region.
            whole_size = min(3 if severity == 4 else 7, py, px)
            y0, y1 = _bounds(cy, whole_size, py)
            x0, x1 = _bounds(cx, whole_size, px)
            mask[y0:y1, x0:x1, :] = True

        if severity >= 5:
            # Full-band loss is reserved for the most severe mixed case.
            full_band_width = min(7, pb)
            b0, b1 = _bounds(cb, full_band_width, pb)
            mask[:, :, b0:b1] = True

    mask[cy, cx, cb] = True
    return mask, scenario, severity


def severity_for_epoch(epoch, max_epochs, rng):
    progress = epoch / max(1, max_epochs)

    if progress < 0.15:
        probabilities = np.array([0.48, 0.32, 0.15, 0.05, 0.00])
    elif progress < 0.35:
        probabilities = np.array([0.25, 0.30, 0.25, 0.15, 0.05])
    elif progress < 0.60:
        probabilities = np.array([0.12, 0.20, 0.28, 0.25, 0.15])
    else:
        probabilities = np.array([0.08, 0.14, 0.22, 0.28, 0.28])

    return int(rng.choice(np.arange(1, 6), p=probabilities))
