import numpy as np


def _neighbors(values, bad, index):
    left = index - 1
    while left >= 0:
        if not bad[left] and np.isfinite(values[left]):
            break
        left -= 1

    right = index + 1
    while right < len(values):
        if not bad[right] and np.isfinite(values[right]):
            break
        right += 1

    left_valid = left >= 0
    right_valid = right < len(values)

    if not left_valid and not right_valid:
        return np.nan, np.inf

    if left_valid and right_valid:
        dl = index - left
        dr = right - index
        total = dl + dr
        value = (values[left] * dr + values[right] * dl) / total
        return float(value), float(total)

    if left_valid:
        return float(values[left]), float(index - left)

    return float(values[right]), float(right - index)


def reconstruct(cube, mask, coords, mode):
    predictions = np.full(len(coords), np.nan, dtype=np.float32)

    for i, (y, x, b) in enumerate(coords):
        y, x, b = map(int, (y, x, b))

        spatial, spatial_span = _neighbors(
            cube[y, :, b],
            mask[y, :, b],
            x,
        )

        spectral, spectral_span = _neighbors(
            cube[y, x, :],
            mask[y, x, :],
            b,
        )

        if mode == "spatial":
            predictions[i] = spatial

        elif mode == "spectral":
            predictions[i] = spectral

        elif mode == "spatio_spectral":
            candidates = []
            weights = []

            if np.isfinite(spatial):
                candidates.append(spatial)
                weights.append(1.0 / max(spatial_span, 1.0))

            if np.isfinite(spectral):
                candidates.append(spectral)
                weights.append(1.0 / max(spectral_span, 1.0))

            if candidates:
                predictions[i] = float(
                    np.average(candidates, weights=weights)
                )

        else:
            raise ValueError(mode)

    return predictions
