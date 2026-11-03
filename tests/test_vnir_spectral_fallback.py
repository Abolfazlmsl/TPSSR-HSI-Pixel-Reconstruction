"""Regression tests for an explicitly rejected VNIR spectral gate."""
from pathlib import Path

import numpy as np
import pytest
import torch

from training.spectral_reference import (
    ACCEPTED_GATE, INTERPOLATION, FrozenInterpolation,
    select_spectral_source_checkpoint,
)


def _make_rejected_gate(tmp_path):
    base = tmp_path / 'tpssr'
    (base / 'spectral_gate' / 'band').mkdir(parents=True)
    (base / 'spectral_backbone' / 'hybrid').mkdir(parents=True)
    (base / 'spectral_gate' / 'GATE_SUMMARY.txt').write_text(
        'global: accepted=False; epoch=8\nband: accepted=False; epoch=7\n', encoding='utf-8')
    (base / 'spectral_gate' / 'band' / 'diagnostic_best_val.pt').write_bytes(b'diagnostic-only')
    (base / 'spectral_backbone' / 'hybrid' / 'diagnostic_best_val.pt').write_bytes(b'stats-only')
    return base


def test_identity_spectral_fallback_is_exact_interpolation():
    sample = torch.randn(3, 90, 4)
    source = FrozenInterpolation()
    assert torch.equal(source(sample), sample[..., 1])
    with pytest.raises(ValueError, match='bands'):
        source(torch.randn(3, 90, 3))


def test_rejected_vnir_selects_backbone_only_as_provenance(tmp_path):
    base = _make_rejected_gate(tmp_path)
    path, mode = select_spectral_source_checkpoint(base, 'VNIR')
    assert mode == INTERPOLATION
    assert path == base / 'spectral_backbone/hybrid/diagnostic_best_val.pt'
    with pytest.raises(FileNotFoundError):
        select_spectral_source_checkpoint(base, 'SWIR')
    (base / 'spectral_gate' / 'GATE_SUMMARY.txt').write_text('band: accepted=True; epoch=7\n')
    with pytest.raises(RuntimeError, match='explicit rejected'):
        select_spectral_source_checkpoint(base, 'VNIR')


def test_accepted_gate_remains_preferred(tmp_path):
    base = _make_rejected_gate(tmp_path)
    gate = base / 'spectral_gate/band/best_model.pt'
    gate.write_bytes(b'accepted-gate')
    for sensor in ('SWIR', 'VNIR'):
        path, mode = select_spectral_source_checkpoint(base, sensor)
        assert path == gate and mode == ACCEPTED_GATE


def test_loader_with_rejected_gate_uses_identity_not_diagnostic_network(tmp_path, monkeypatch):
    base = _make_rejected_gate(tmp_path)
    backbone_path = base / 'spectral_backbone/hybrid/diagnostic_best_val.pt'
    wave = np.array([445., 555., 665.], dtype=np.float32)
    mean = np.array([.2, .3, .4], dtype=np.float32)
    std = np.array([.1, .15, .2], dtype=np.float32)
    source_train = ('VNIR1', 'VNIR2', 'VNIR3', 'VNIR6', 'VNIR7', 'VNIR8')
    source_val = ('VNIR4', 'VNIR9')
    torch.save({
        'architecture': 'spectral_only_hybrid_residual_v1',
        'model_params': {'bands': 3},
        'run_config': {'train_cores': source_train, 'val_cores': source_val},
        'wavelengths_nm': wave, 'band_mean': mean, 'band_std': std,
        # If this field were used to instantiate a neural model, the test
        # would fail; for rejected VNIR the weights must be entirely ignored.
        'model_state_dict': None,
    }, backbone_path)

    import config
    import training.spectral_gate as spectral_gate
    from src.pipeline import _load_accepted_spectral_hybrid_v2

    data = tmp_path / 'processed_data' / 'VNIR5'
    data.mkdir(parents=True)
    np.save(data / 'wavelengths_nm.npy', wave)
    monkeypatch.setattr(config, 'MODELS_DIR', tmp_path)
    monkeypatch.setattr(config, 'PROCESSED_DATA_ROOT', tmp_path / 'processed_data')
    monkeypatch.setattr(config, 'TEST_DATASET_FOLDER', 'VNIR5')
    monkeypatch.setattr(spectral_gate, 'TRAIN_CORES', source_train)
    monkeypatch.setattr(spectral_gate, 'VALID_CORES', source_val)
    model, got_mean, got_std, wn, source_path, pin_path = (
        _load_accepted_spectral_hybrid_v2(torch.device('cpu')))
    x = torch.randn(2, 3, 4)
    assert isinstance(model, FrozenInterpolation)
    assert torch.equal(model(x), x[..., 1])
    np.testing.assert_array_equal(got_mean, mean)
    np.testing.assert_array_equal(got_std, std)
    assert source_path == pin_path == backbone_path
