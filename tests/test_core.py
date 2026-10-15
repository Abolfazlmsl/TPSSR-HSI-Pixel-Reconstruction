"""Fast, CPU-only checks for mask semantics and project wiring."""
from __future__ import annotations

import importlib
import numpy as np


def test_required_components_import():
    for name in (
        "src.pipeline", "src.spatial_inference", "src.topology_completion",
        "src.routing_policy", "training.spectral_backbone", "training.spectral_gate",
        "src.local_methods", "src.ml_common", "src.training_corruptions",
    ):
        assert importlib.import_module(name) is not None


def test_classical_interpolation_does_not_read_masked_target():
    from src.local_methods import reconstruct
    rng = np.random.default_rng(10)
    cube = rng.random((7, 9, 21)).astype(np.float32)
    mask = np.zeros(cube.shape, dtype=bool)
    mask[3, 4, 8:13] = True
    coords = np.argwhere(mask)
    modified = cube.copy()
    modified[mask] = 12345.0
    for mode in ("spectral", "spatial", "spatio_spectral"):
        actual = reconstruct(cube, mask, coords, mode)
        same_except_masked = reconstruct(modified, mask, coords, mode)
        np.testing.assert_allclose(actual, same_except_masked, equal_nan=True)


def test_random_forest_features_hide_masked_values():
    from src.ml_common import rf_feature
    import config
    rng = np.random.default_rng(21)
    shape = (config.RF_PATCH_Y, config.RF_PATCH_X, config.RF_PATCH_B)
    source = rng.random(shape).astype(np.float32)
    masked = np.zeros(shape, dtype=bool)
    center = tuple(dim // 2 for dim in shape)
    masked[center] = True
    altered = source.copy()
    altered[center] = 999.0
    band_mean = np.linspace(0.2, 0.6, 30, dtype=np.float32)
    np.testing.assert_array_equal(
        rf_feature(source, masked.copy(), band_mean, 15),
        rf_feature(altered, masked.copy(), band_mean, 15),
    )


def test_training_corruption_masks_include_center():
    from src.training_corruptions import generate_training_mask
    rng = np.random.default_rng(4)
    for _ in range(100):
        mask, scenario, severity = generate_training_mask((31, 31, 63), rng)
        assert mask[15, 15, 31]
        assert scenario and 1 <= severity <= 5


def test_shared_mask_budget():
    from mask_budget import planned_counts
    assert planned_counts(200, 2) == [100, 100]
    assert sum(planned_counts(200, 2)) == 200
    assert planned_counts(200, 3) == [67, 67, 66]


def test_model_is_spectral_only():
    import torch
    from training.spectral_gate import FrozenHybridGated
    model = FrozenHybridGated(16, 32, 4, 1, variant="band").eval()
    features = torch.zeros(2, 16, 4)
    features[:, :, 2] = 1.0
    with torch.inference_mode():
        output = model(features)
    assert output.shape == (2, 16)
    assert torch.isfinite(output).all()
