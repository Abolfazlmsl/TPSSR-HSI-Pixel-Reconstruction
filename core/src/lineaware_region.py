"""Load the selected SWIR 8x2 line-aware Region checkpoints.

The original geometry and topology checkpoints remain in place because the
frozen quality router was trained against their identities. Region weights
are switched only after the router's provenance checks have passed.
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
    if (str(config.TEST_DATASET_FOLDER).upper() != 'SWIR5' or
        tuple(config.TEST_DATASET_FOLDERS) != ('SWIR5', 'SWIR10') or
        tuple(config.TRAIN_DATASET_FOLDERS) !=
        ('SWIR1','SWIR2','SWIR3','SWIR4','SWIR6','SWIR7','SWIR8','SWIR9')):
        raise RuntimeError('Line-aware final Region is selected only for the accepted SWIR 8x2 model')
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
    original_topology = spatial_inference.topology_checkpoint_path
    original_candidates = spatial_inference._candidate_geometry_paths
    try:
        spatial_inference.topology_checkpoint_path = lambda: topology_path
        spatial_inference._candidate_geometry_paths = lambda _ck=None: [geometry_path]
        result = spatial_inference.load_models(device)
    finally:
        spatial_inference.topology_checkpoint_path = original_topology
        spatial_inference._candidate_geometry_paths = original_candidates
    topology, topo_ck, geometry, geom_ck, top_path, geo_path = result
    if Path(top_path).resolve() != topology_path.resolve():
        raise RuntimeError(f'Loaded wrong Topology checkpoint: {top_path}')
    if Path(geo_path).resolve() != geometry_path.resolve():
        raise RuntimeError(f'Loaded wrong Geometry checkpoint: {geo_path}; check source_geometry_checkpoint in trial Topology')
    source = Path(str(topo_ck.get('source_geometry_checkpoint', '')))
    if source.parent.name != 'geometry_swir5' or source.name != 'best_model.pt':
        raise RuntimeError('Topology source does not identify the expected line-aware Geometry')
    for key in ('band_mean', 'band_std', 'wavelengths_nm'):
        if not np.allclose(np.asarray(topo_ck[key]), np.asarray(geom_ck[key]),
                           atol=1e-6, rtol=1e-6):
            raise RuntimeError('Topology normalization does not match the chosen Geometry')
    _validate_ck('Topology', topo_ck, np.asarray(wavelengths, np.float32))
    _validate_ck('Geometry', geom_ck, np.asarray(wavelengths, np.float32))
    print('[TPSSR] VERIFIED line-aware Region Geometry:', geo_path, fingerprint(geo_path), flush=True)
    print('[TPSSR] VERIFIED line-aware Region Topology:', top_path, fingerprint(top_path), flush=True)
    print('[TPSSR] Router retained from original model; Region calibration unchanged.', flush=True)
    return result
