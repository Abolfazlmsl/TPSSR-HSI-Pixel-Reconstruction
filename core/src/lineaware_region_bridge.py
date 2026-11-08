"""Load trained 8x2 line-aware Geometry and Topology in the accepted TPSSR runtime.

No checkpoint substitution: verified exact paths and same spectral grid.
The base three-expert router stays unchanged and therefore remains uncalibrated
for this experimental Region expert.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import config
from src import spatial_inference

TRIAL_NAME = "region_lineaware_trial_v1"


def checkpoint_paths():
    if str(config.TEST_DATASET_FOLDER).upper() != 'SWIR5':
        raise RuntimeError('Line-aware Region only supports the SWIR 8x2 checkpoint tag SWIR5')
    root = Path(config.MODELS_DIR) / 'tpssr'
    trial = root / TRIAL_NAME
    return (trial / 'geometry_swir5' / 'best_model.pt',
            trial / 'topology_swir5' / 'best_model.pt')


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def validate_paths(geometry_path, topology_path):
    reference = Path(config.MODELS_DIR) / 'tpssr'
    original_geom = reference / 'geometry_swir5' / 'best_model.pt'
    original_topo = reference / 'topology_swir5' / 'best_model.pt'
    for label, p in [('Line-aware Geometry', geometry_path),
                     ('Line-aware Topology', topology_path),
                     ('Accepted Geometry', original_geom),
                     ('Accepted Topology', original_topo)]:
        if not p.is_file():
            raise FileNotFoundError(f'{label} checkpoint missing: {p}')
    if geometry_path.resolve() == original_geom.resolve() or topology_path.resolve() == original_topo.resolve():
        raise RuntimeError('Trial and accepted checkpoints resolve to the same file')
    return original_geom, original_topo


def _validate_ck(label, ck, waves):
    n = len(waves)
    if int(ck['n_bands']) != n:
        raise RuntimeError(f'{label} bands {ck["n_bands"]} != {n}')
    for key in ('band_mean','band_std'):
        arr = np.asarray(ck[key], dtype=np.float32)
        if arr.shape != (n,) or not np.isfinite(arr).all() or (key == 'band_std' and np.any(arr <= 0)):
            raise RuntimeError(f'{label} invalid {key}')
    if 'wavelengths_nm' in ck:
        old = np.asarray(ck['wavelengths_nm'], np.float32)
        if old.shape != (n,) or not np.allclose(old, waves, rtol=0, atol=0.5):
            raise RuntimeError(f'{label} spectral grid differs from the current dataset')


def load_lineaware_region(device, wavelengths):
    """Return (topology, topo_ck, geometry, geom_ck, topology_path, geometry_path)."""
    geometry_path, topology_path = checkpoint_paths()
    validate_paths(geometry_path, topology_path)
    if not np.isfinite(wavelengths).all():
        raise ValueError('Nonfinite wavelength grid')
    # Reuse exactly the project's model constructor and checkpoint normalization.
    # The topology metadata must pin the trial Geometry, not a production weight.
    original = spatial_inference.topology_checkpoint_path
    try:
        spatial_inference.topology_checkpoint_path = lambda: topology_path
        result = spatial_inference.load_models(device)
    finally:
        spatial_inference.topology_checkpoint_path = original
    topology, topo_ck, geometry, geom_ck, top_path, geo_path = result
    if Path(top_path).resolve() != topology_path.resolve():
        raise RuntimeError(f'Loaded wrong Topology checkpoint: {top_path}')
    if Path(geo_path).resolve() != geometry_path.resolve():
        raise RuntimeError(f'Loaded wrong Geometry checkpoint: {geo_path}; check source_geometry_checkpoint in trial Topology')
    _validate_ck('Topology', topo_ck, np.asarray(wavelengths, np.float32))
    _validate_ck('Geometry', geom_ck, np.asarray(wavelengths, np.float32))
    print('[TPSSR] VERIFIED line-aware Region Geometry:', geo_path, fingerprint(geo_path), flush=True)
    print('[TPSSR] VERIFIED line-aware Region Topology:', top_path, fingerprint(top_path), flush=True)
    print('[TPSSR] WARNING: existing learned router is NOT recalibrated for this Region.', flush=True)
    return result
