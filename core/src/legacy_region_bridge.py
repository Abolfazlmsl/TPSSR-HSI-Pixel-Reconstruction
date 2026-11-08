"""Load the frozen SWIR1 V27/V13 Region checkpoints in the current inference runtime.

Both networks have the same state-dict layout as the renamed Geometry/Topology
classes. All normalization is taken from the legacy checkpoint itself.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from src.spatial_inference import GEOMETRY_ARCH, torch_load
from src.topology_completion import TopologyCompletionNetwork


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _legacy_folder(root):
    root = Path(root).expanduser().resolve()
    if not (root / 'core' / 'src' / 'final_tpssr_runtime.py').is_file():
        raise FileNotFoundError(f'Legacy TPSSR project not found: {root}')
    parent = root / 'trained_models' / 'global_SWIR2_SWIR3_SWIR4_SWIR5__test_SWIR1'
    path = parent / 'texture_restoration_router_rf_v2'
    if path.is_dir():
        return path
    candidates = sorted((root / 'trained_models').glob(
        '*/texture_restoration_router_rf_v2/new_v13_topology_first_swir1/best_model.pt'
    ))
    if len(candidates) != 1:
        raise FileNotFoundError(
            'Cannot uniquely locate the legacy SWIR1 V13 checkpoint under '
            f'{root / "trained_models"}. Found {len(candidates)} candidates.'
        )
    return candidates[0].parent.parent


def load_legacy_region(device, project_root, bands, wavelengths=None):
    """Return (topology, topo_ck, geometry, geom_ck, topo_path, geom_path)."""
    folder = _legacy_folder(project_root)
    topo_path = folder / 'new_v13_topology_first_swir1' / 'best_model.pt'
    if not topo_path.is_file():
        raise FileNotFoundError(f'Legacy V13 checkpoint missing: {topo_path}')
    topo_ck = torch_load(topo_path, 'cpu')
    if topo_ck.get('uses_v27_final_reconstruction', False):
        raise RuntimeError('Legacy V13 checkpoint must use V27 Stage-A guidance only')
    source = topo_ck.get('source_v27_checkpoint')
    candidates = []
    if source:
        recorded = Path(str(source))
        if recorded.is_file() and folder in recorded.resolve().parents:
            candidates.append(recorded)
        candidates.append(folder / recorded.parent.name / recorded.name)
    candidates.extend([
        folder / 'region_v27_original_swir1_dev_2gpu' / 'best_model.pt',
        folder / 'region_v27_learned_correspondence_swir1_dev_2gpu' / 'best_model.pt',
    ])
    geom_path = next((p.resolve() for p in candidates if p.is_file()), None)
    if geom_path is None:
        raise FileNotFoundError('Legacy V27 Stage-A checkpoint missing. Checked:\n' +
                                '\n'.join(str(p) for p in candidates))
    geom_ck = torch_load(geom_path, 'cpu')
    n = int(bands)
    if int(topo_ck['n_bands']) != n or int(geom_ck['n_bands']) != n:
        raise ValueError(f'Legacy checkpoints have {topo_ck["n_bands"]}/{geom_ck["n_bands"]} '
                         f'bands; current dataset has {n} bands.')
    for label, ck in [('V13', topo_ck), ('V27', geom_ck)]:
        for key in ('band_mean', 'band_std'):
            values = np.asarray(ck[key], np.float32)
            if values.shape != (n,) or not np.isfinite(values).all():
                raise ValueError(f'Invalid {label} {key} in legacy checkpoint')
            if key == 'band_std' and np.any(values <= 0):
                raise ValueError(f'Non-positive {label} standard deviation')
        if wavelengths is not None and 'wavelengths_nm' in ck:
            legacy_waves = np.asarray(ck['wavelengths_nm'], np.float32)
            if legacy_waves.shape != (n,) or not np.allclose(
                legacy_waves, np.asarray(wavelengths), rtol=0, atol=0.5
            ):
                raise ValueError(f'Legacy {label} wavelength grid mismatches current dataset')

    s = topo_ck['settings']
    topology = TopologyCompletionNetwork(
        n_bands=n,
        structure_feature_channels=int(s['structure_feature_channels']),
        spectral_width=int(s['spectral_width']),
        base_width=int(s['base_width']),
        transport_embed_ch=int(s['transport_embed_ch']),
        transport_heads=int(s['transport_heads']),
        transport_topk=int(s['transport_topk']),
        transport_radius=int(s['transport_radius']),
        transport_temperature=float(s['transport_temperature']),
        radiometry_delta_scale=float(s['radiometry_delta_scale']),
        detail_scale=float(s['detail_scale']),
    )
    try:
        topology.load_state_dict(topo_ck['model_state_dict'], strict=True)
    except RuntimeError as exc:
        raise RuntimeError('Legacy V13 state is incompatible with the current Topology class; '
                           'no weights were substituted. ' + str(exc)) from exc
    settings = geom_ck['settings']
    geometry = GEOMETRY_ARCH.GeometryGuidanceModel(
        n_bands=n,
        spectral_width=int(settings['spectral_width']),
        base_width=int(settings['base_width']),
        attention_heads=int(settings['attention_heads']),
    )
    try:
        geometry.load_state_dict(geom_ck['model_state_dict'], strict=True)
    except RuntimeError as exc:
        raise RuntimeError('Legacy V27 state is incompatible with the current Geometry class; '
                           'no weights were substituted. ' + str(exc)) from exc
    geometry = geometry.to(device).eval()
    topology = topology.to(device).eval()
    print('[TPSSR] Region override: legacy V27 Stage-A + V13 TopologyFirst', flush=True)
    print(f'[TPSSR] Legacy Geometry: {geom_path} (sha256 {_sha256(geom_path)[:16]})', flush=True)
    print(f'[TPSSR] Legacy Topology: {topo_path} (sha256 {_sha256(topo_path)[:16]})', flush=True)
    print('[TPSSR] WARNING: Legacy Region SWIR1 was trained on SWIR5. '
          'SWIR5 evaluation is NOT held-out; router risk model is not recalibrated.', flush=True)
    return topology, topo_ck, geometry, geom_ck, topo_path.resolve(), geom_path
