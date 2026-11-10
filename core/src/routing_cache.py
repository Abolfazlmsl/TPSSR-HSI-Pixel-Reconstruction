"""Verified, lazy, per-mask caches for learned three-expert routing.

Spectral/Fusion values are already computed while extracting RankCV features.
Region predictions are cached under the *exact* selected-coordinate set,
because the Region tiling protocol depends on that set. The cache never uses
masked ground-truth values to select an expert.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np

SCHEMA = 'tpssr_rankcv_lazy_cache_v1'


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for piece in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(piece)
    return h.hexdigest()


def source_signature(pipeline, dataset):
    """Build once per worker, not once per mask."""
    import config
    from training.three_expert_router import path_for
    model_root = Path(config.MODELS_DIR)
    router_path = path_for(model_root, objective='rank_cv', region_expert='lineaware')
    file_paths = {
        'router': router_path,
        'fusion': pipeline.fusion_checkpoint_path,
        'spectral_gate': pipeline.hybrid_gate_checkpoint_path,
        'spectral_backbone': pipeline.hybrid_backbone_checkpoint_path,
        'geometry': pipeline.geometry_path,
        'topology': pipeline.topology_path,
    }
    sources = {
        'pipeline_source': Path(__file__).with_name('pipeline.py'),
        'routing_features_source': Path(__file__).resolve().parents[2] / 'training' / 'three_expert_router.py',
        'fusion_features_source': Path(__file__).resolve().parents[2] / 'training' / 'neighbor_fusion.py',
        'region_inference_source': Path(__file__).with_name('spatial_inference.py'),
        'region_architecture_source': Path(__file__).with_name('topology_completion.py'),
        'lineaware_region_source': Path(__file__).with_name('lineaware_region.py'),
        'region_geometry_source': Path(__file__).parents[1] / 'reconstruction_methods' / 'tpssr' / 'train_geometry.py',
        'component_routing_source': Path(__file__).with_name('component_routing_utils.py'),
    }
    checkpoint_hashes = {}
    for name, path in {**file_paths, **sources}.items():
        if path is None or not Path(path).is_file():
            raise FileNotFoundError(f'Lazy cache identity missing {name}: {path}')
        checkpoint_hashes[name] = sha_file(path)
    processed = Path(config.PROCESSED_DATA_ROOT) / str(dataset)
    input_files = [processed / name for name in (
        'calibrated_reflectance.npy', 'invalid_mask.npy', 'wavelengths_nm.npy')]
    input_stats = {}
    for path in input_files:
        if not path.is_file():
            raise FileNotFoundError(f'Lazy cache input missing: {path}')
        s = path.stat()
        input_stats[path.name] = [str(path.resolve()), int(s.st_size), int(s.st_mtime_ns)]
    return {
        'schema': SCHEMA,
        'experiment': str(config.EXPERIMENT_NAME),
        'active_test': str(dataset),
        'seed': int(config.SEED),
        'fusion_donor_match': os.environ.get('TPSSR_FUSION_DONOR_MATCH', 'legacy').strip().lower(),
        'model_and_source_sha256': checkpoint_hashes,
        'processed_data_stats': input_stats,
    }


def hash_array(value, h):
    arr = np.ascontiguousarray(value)
    h.update(str(arr.shape).encode())
    h.update(arr.dtype.str.encode())
    h.update(memoryview(arr).cast('B'))


class MaskCache:
    def __init__(self, root, signature, mask_path, mask, coords, mode='lazy'):
        if mode not in ('lazy', 'replay_only'):
            raise ValueError('Invalid lazy cache mode')
        self.mode = mode
        self.signature = signature
        h = hashlib.sha256(json.dumps(signature, sort_keys=True).encode('utf-8'))
        h.update(str(Path(mask_path).resolve()).encode('utf-8'))
        h.update(str(np.asarray(mask).shape).encode('ascii'))
        hash_array(np.packbits(np.asarray(mask, bool).ravel()), h)
        hash_array(np.asarray(coords, dtype=np.int64), h)
        self.identity = h.hexdigest()
        stem = Path(mask_path).stem
        self.root = Path(root) / f'{stem}_{self.identity[:20]}'
        self.base_path = self.root / 'rankcv_features.npz'
        self.feature_hit = False
        self.region_hit = False
        self.region_used = False

    @staticmethod
    def _read(path):
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                return {k: archive[k] for k in archive.files}
        except (OSError, ValueError, EOFError) as exc:
            raise RuntimeError(f'Corrupted lazy cache, delete only this file: {path}') from exc

    @staticmethod
    def _write(path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with tmp.open('wb') as stream:
                np.savez_compressed(stream, **values)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

    def load_features(self, pixels, bands):
        data = self._read(self.base_path)
        if data is None:
            if self.mode == 'replay_only':
                raise RuntimeError('Replay-only: missing feature cache: ' + str(self.base_path))
            return None
        n = len(pixels)
        expected = {'schema', 'identity', 'pixels', 'eligible', 'risks', 'raw_choices', 'interp', 'fusion'}
        if set(data) != expected:
            raise RuntimeError('Unrecognized RankCV cache fields: ' + str(self.base_path))
        if (str(data['schema'].item()) != SCHEMA or str(data['identity'].item()) != self.identity
                or not np.array_equal(data['pixels'], pixels)
                or data['eligible'].shape != (n, 3)
                or data['risks'].shape != (n, 3)
                or data['raw_choices'].shape != (n,)
                or data['interp'].shape != (n, bands)
                or data['fusion'].shape != (n, bands)):
            raise RuntimeError('RankCV cache identity/shape mismatch: ' + str(self.base_path))
        chosen = data['raw_choices'].astype(np.int64)
        eligible = data['eligible'].astype(bool)
        if (np.any(chosen < 0) or np.any(chosen > 2)
                or not np.all(eligible[np.arange(n), chosen])):
            raise RuntimeError('Invalid cached RankCV decisions')
        self.feature_hit = True
        return data

    def save_features(self, pixels, eligible, risks, raw_choices, interp_all, fusion_all, bands):
        if self.mode == 'replay_only':
            raise RuntimeError('Replay-only forbids feature inference')
        n = len(pixels)
        def stack(items):
            array = np.full((n, bands), np.nan, dtype=np.float32)
            for i, value in enumerate(items):
                if value is not None:
                    a = np.asarray(value, dtype=np.float32)
                    if a.shape != (bands,):
                        raise RuntimeError('Unexpected spectral candidate shape in lazy cache')
                    array[i] = a
            return array
        self._write(self.base_path, {
            'schema': np.asarray(SCHEMA), 'identity': np.asarray(self.identity),
            'pixels': np.asarray(pixels, np.int64), 'eligible': np.asarray(eligible, bool),
            'risks': np.asarray(risks, np.float32),
            'raw_choices': np.asarray(raw_choices, np.int8),
            'interp': stack(interp_all), 'fusion': stack(fusion_all),
        })

    def region_file(self, coords, selected_coords, region_full_eval, region_batch, verify_batch):
        h = hashlib.sha256()
        h.update(self.identity.encode())
        h.update(b'full' if region_full_eval else b'selected')
        h.update(f'{int(region_batch)}:{int(verify_batch)}'.encode())
        hash_array(np.asarray(coords, dtype=np.int64), h)
        hash_array(np.asarray(selected_coords, dtype=np.int64), h)
        return self.root / ('region_' + h.hexdigest()[:32] + '.npz')

    def load_region(self, selected_coords, input_coords, region_full_eval, region_batch, verify_batch):
        self.region_used = True
        path = self.region_file(input_coords, selected_coords, region_full_eval, region_batch, verify_batch)
        data = self._read(path)
        if data is None:
            if self.mode == 'replay_only':
                raise RuntimeError('Replay-only: missing exact Region cache for chosen pixels: ' + str(path))
            return None
        if (set(data) != {'selected_coords', 'region_values', 'n_refined'}
                or not np.array_equal(data['selected_coords'], selected_coords)
                or data['region_values'].shape != (len(selected_coords),)
                or not np.isfinite(data['region_values']).all()):
            raise RuntimeError('Invalid or mismatched Region cache: ' + str(path))
        self.region_hit = True
        return data['region_values'].astype(np.float32), int(data['n_refined'].item())

    def save_region(self, selected_coords, input_coords, region_full_eval, region_batch, verify_batch, values, n_refined):
        if self.mode == 'replay_only':
            raise RuntimeError('Replay-only forbids Region inference')
        self._write(self.region_file(input_coords, selected_coords, region_full_eval, region_batch, verify_batch), {
            'selected_coords': np.asarray(selected_coords, np.int64),
            'region_values': np.asarray(values, np.float32),
            'n_refined': np.asarray(int(n_refined), np.int64),
        })
