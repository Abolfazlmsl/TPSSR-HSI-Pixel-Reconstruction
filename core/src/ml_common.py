import numpy as np

import config


def band_stats(cube, chunk_y=64):
    if isinstance(cube, (list, tuple)):
        cubes = list(cube)
    else:
        cubes = [cube]

    if not cubes:
        raise ValueError(
            "At least one cube is required."
        )

    n_bands = cubes[0].shape[2]

    for index, current in enumerate(cubes):
        if current.shape[2] != n_bands:
            raise ValueError(
                "All cubes must have the same number of bands. "
                f"Cube 0 has {n_bands}, cube {index} has "
                f"{current.shape[2]}."
            )

    sums = np.zeros(
        n_bands,
        dtype=np.float64,
    )
    sums2 = np.zeros(
        n_bands,
        dtype=np.float64,
    )
    counts = np.zeros(
        n_bands,
        dtype=np.int64,
    )

    for current in cubes:
        for y0 in range(
            0,
            current.shape[0],
            chunk_y,
        ):
            block = np.asarray(
                current[
                    y0:y0 + chunk_y
                ],
                dtype=np.float32,
            )

            finite = np.isfinite(
                block
            )

            safe = np.where(
                finite,
                block,
                0.0,
            ).astype(
                np.float64
            )

            sums += np.sum(
                safe,
                axis=(0, 1),
            )
            sums2 += np.sum(
                safe * safe,
                axis=(0, 1),
            )
            counts += np.sum(
                finite,
                axis=(0, 1),
            )

    counts_safe = np.maximum(
        counts,
        1,
    )

    mean = (
        sums
        / counts_safe
    )
    variance = (
        sums2
        / counts_safe
        - mean * mean
    )
    variance = np.maximum(
        variance,
        1e-12,
    )

    std = np.sqrt(
        variance
    )

    mean = mean.astype(
        np.float32
    )
    std = std.astype(
        np.float32
    )

    mean[
        ~np.isfinite(mean)
    ] = 0.0
    std[
        ~np.isfinite(std)
        | (std < 1e-6)
    ] = 1.0

    return mean, std


def extract_patch(cube, y, x, b, patch_y, patch_x, patch_b, mask=None):
    """
    Extract a fixed-size 3D patch.

    Boundary coordinates are clipped to the nearest valid image/spectral
    coordinate. Interior patches are identical to ordinary slicing.
    """
    ry = patch_y // 2
    rx = patch_x // 2
    rb = patch_b // 2

    y_idx = np.clip(
        np.arange(y - ry, y + ry + 1),
        0,
        cube.shape[0] - 1,
    )
    x_idx = np.clip(
        np.arange(x - rx, x + rx + 1),
        0,
        cube.shape[1] - 1,
    )
    b_idx = np.clip(
        np.arange(b - rb, b + rb + 1),
        0,
        cube.shape[2] - 1,
    )

    values = np.asarray(
        cube[np.ix_(y_idx, x_idx, b_idx)],
        dtype=np.float32,
    ).copy()

    if mask is None:
        local_mask = np.zeros(values.shape, dtype=bool)
    else:
        local_mask = np.asarray(
            mask[np.ix_(y_idx, x_idx, b_idx)],
            dtype=bool,
        ).copy()

    return values, local_mask


def rf_feature(values, bad, band_mean, center_band):
    rb = config.RF_PATCH_B // 2
    band_indices = np.clip(
        np.arange(
            center_band - rb,
            center_band + rb + 1,
        ),
        0,
        len(band_mean) - 1,
    )

    fill = np.broadcast_to(
        band_mean[band_indices][None, None, :],
        values.shape,
    )

    values = values.copy()
    bad = bad | ~np.isfinite(values)
    values[bad] = fill[bad]

    cy = config.RF_PATCH_Y // 2
    cx = config.RF_PATCH_X // 2
    cb = config.RF_PATCH_B // 2

    values[cy, cx, cb] = fill[cy, cx, cb]
    bad[cy, cx, cb] = True

    band_position = np.array(
        [center_band / max(1, len(band_mean) - 1)],
        dtype=np.float32,
    )

    return np.concatenate([
        values.reshape(-1).astype(np.float32),
        bad.reshape(-1).astype(np.float32),
        band_position,
    ])
