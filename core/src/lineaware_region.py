"""Load split-specific line-aware Region checkpoints with provenance checks."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import config
from src import spatial_inference

TRIAL_NAME = "region_lineaware_trial_v1"


def checkpoint_paths(*, full_split=False):
    swir8 = ((full_split or str(config.TEST_DATASET_FOLDER).upper() == 'SWIR5') and
        tuple(config.TEST_DATASET_FOLDERS) == ('SWIR5', 'SWIR10') and
        tuple(config.TRAIN_DATASET_FOLDERS) ==
        ('SWIR1','SWIR2','SWIR3','SWIR4','SWIR6','SWIR7','SWIR8','SWIR9'))
    a2b = ((full_split or str(config.TEST_DATASET_FOLDER).upper() == 'SWIR6') and
        tuple(config.TEST_DATASET_FOLDERS) == ('SWIR6','SWIR7','SWIR8','SWIR9','SWIR10') and
        tuple(config.TRAIN_DATASET_FOLDERS) == ('SWIR1','SWIR2','SWIR3','SWIR4','SWIR5') and
        (full_split or __import__('os').environ.get('TPSSR_A2B_LINEAWARE_TEST') == '1'))
    b2a = ((full_split or str(config.TEST_DATASET_FOLDER).upper() == 'SWIR1') and
        tuple(config.TEST_DATASET_FOLDERS) == ('SWIR1','SWIR2','SWIR3','SWIR4','SWIR5') and
        tuple(config.TRAIN_DATASET_FOLDERS) == ('SWIR6','SWIR7','SWIR8','SWIR9','SWIR10') and
        (full_split or __import__('os').environ.get('TPSSR_B2A_LINEAWARE_TEST') == '1'))
    if not (swir8 or a2b or b2a):
        raise RuntimeError('No verified line-aware Region is selected for this split')
    tag = 'swir5' if swir8 else ('swir6' if a2b else 'swir1')
    root = Path(config.MODELS_DIR) / 'tpssr' / TRIAL_NAME
    return (root / f'geometry_{tag}' / 'best_model.pt',
            root / f'topology_{tag}' / 'best_model.pt')


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def validate_paths(geometry_path, topology_path):
    reference = Path(config.MODELS_DIR) / 'tpssr'
    original_geom = reference / geometry_path.parent.name / 'best_model.pt'
    original_topo = reference / topology_path.parent.name / 'best_model.pt'
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


def load_region_from_paths(device, wavelengths, geometry_path, topology_path):
    """Load a pinned Geometry/Topology pair without falling back to old weights."""
    geometry_path, topology_path = Path(geometry_path), Path(topology_path)
    if not geometry_path.is_file() or not topology_path.is_file():
        raise FileNotFoundError(f'Missing line-aware Region checkpoints: {geometry_path}, {topology_path}')
    if not np.isfinite(wavelengths).all():
        raise ValueError('Nonfinite wavelength grid')
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
        raise RuntimeError(f'Loaded wrong Geometry checkpoint: {geo_path}')
    source = Path(str(topo_ck.get('source_geometry_checkpoint', '')))
    if source.parent.name != geometry_path.parent.name or source.name != 'best_model.pt':
        raise RuntimeError('Topology source does not identify the expected line-aware Geometry')
    if source.is_file() and source.resolve() != geometry_path.resolve():
        raise RuntimeError('Topology source points to a different available Geometry checkpoint')
    # If the project was moved, the basename check above is insufficient:
    # require the stored checkpoint fingerprint whenever it is available.
    from training.spectral_gate import sha
    source_hash = topo_ck.get('source_geometry_sha256')
    if source_hash is not None and source_hash != sha(geometry_path):
        raise RuntimeError('Line-aware Topology was trained with different Geometry weights')
    for key in ('band_mean', 'band_std', 'wavelengths_nm'):
        if not np.allclose(np.asarray(topo_ck[key]), np.asarray(geom_ck[key]),
                           atol=1e-6, rtol=1e-6):
            raise RuntimeError('Topology normalization does not match the chosen Geometry')
    _validate_ck('Topology', topo_ck, np.asarray(wavelengths, np.float32))
    _validate_ck('Geometry', geom_ck, np.asarray(wavelengths, np.float32))
    print('[TPSSR] VERIFIED line-aware Region Geometry:', geo_path, fingerprint(geo_path), flush=True)
    print('[TPSSR] VERIFIED line-aware Region Topology:', top_path, fingerprint(top_path), flush=True)
    return result


def load_lineaware_region(device, wavelengths, *, full_split=False):
    """Return the selected (topology, topo_ck, geometry, geom_ck, paths)."""
    geometry_path, topology_path = checkpoint_paths(full_split=full_split)
    validate_paths(geometry_path, topology_path)
    return load_region_from_paths(device, wavelengths, geometry_path, topology_path)
