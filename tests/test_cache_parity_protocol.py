"""The test pathway must use the same Region support as the expert cache."""
from types import SimpleNamespace

import numpy as np
import pytest


def test_fixed_union_region_support_changes_only_requested_region_context(tmp_path, monkeypatch):
    from src import pipeline as pl
    import training.neighbor_fusion as neighbor
    import training.three_expert_router as router

    (tmp_path / 'fake_core').mkdir()
    np.save(tmp_path / 'fake_core' / 'wavelengths_nm.npy', np.arange(4, dtype=np.float32))
    monkeypatch.setattr(pl.config, 'PROCESSED_DATA_ROOT', tmp_path)
    monkeypatch.setattr(pl.config, 'TEST_DATASET_FOLDER', 'fake_core')
    monkeypatch.setattr(neighbor, 'crop_at', lambda cube, mask, y, x, radius: (cube, mask, y, x))

    def fake_features(patch, damaged, cy, cx, *args):
        f = np.zeros(len(router.FEATURE_NAMES), dtype=np.float32)
        f[0] = float(cx)
        return f, np.arange(4, dtype=np.float32), np.arange(4, dtype=np.float32) + 0.5

    monkeypatch.setattr(router, 'features_for_pixel', fake_features)
    monkeypatch.setattr(router, 'predict_risks', lambda payload, x, eligibility: np.asarray(
        [[3, 4, 1], [3, 1, 4]], dtype=np.float32))
    monkeypatch.setattr(router, 'refine_region', lambda cube, mask, coords, raw: (np.asarray(raw), 0))
    seen = []

    def fake_region(*args, eval_coords, **kwargs):
        seen.append(np.asarray(eval_coords).copy())
        # Return values dependent on the *request* to emulate coordinate-dependent tiling.
        result = {}
        for band in np.unique(eval_coords[:, 2]):
            selected = eval_coords[eval_coords[:, 2] == band]
            result[int(band)] = (selected[:, :2], np.full(len(selected),
                                                          len(eval_coords) / 10., dtype=np.float32))
        return result, []

    monkeypatch.setattr(pl, 'reconstruct_final_region', fake_region)
    cube = np.zeros((4, 4, 4), np.float32)
    mask = np.zeros_like(cube, bool)
    coords = np.asarray([[1, 1, 1], [1, 2, 2]], dtype=np.int64)
    mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    bundle = SimpleNamespace(fusion_model=object(), router3_checkpoint={'region_expert': 'lineaware'},
                             topology_model=None, topology_checkpoint={}, geometry_model=None,
                             device='cpu', region_batch=8, verify_region_batch=False,
                             hybrid_mean=None, hybrid_std=None, hybrid_model=None)
    routed = pl.predict_mask_learned_router3(bundle, cube, mask, coords,
                                             region_full_eval=False)
    fixed = pl.predict_mask_learned_router3(bundle, cube, mask, coords,
                                            region_full_eval=True)
    assert np.array_equal(seen[0], coords[:1])
    assert np.array_equal(seen[1], coords)
    assert routed['prediction'][0] == np.float32(0.1)
    assert fixed['prediction'][0] == np.float32(0.2)
    assert routed['prediction'][1] == fixed['prediction'][1]
    assert routed['candidate_method'].tolist() == fixed['candidate_method'].tolist()


def test_fixed_protocol_result_folder_is_not_legacy_folder():
    from evaluate import results_folder
    spec = SimpleNamespace(experiment_name='fake')
    from unittest.mock import patch
    with patch('evaluate.selected_results_folder', return_value='router3_rankcv'):
        assert results_folder(spec, 'some_router') == 'router3_rankcv'
        assert results_folder(spec, 'some_router', True) == 'router3_rankcv__fixed_union_region'


def test_score_summary_measures_coverage_from_files(tmp_path, monkeypatch):
    import csv
    from tools import router_expert_cache as cache
    spec = SimpleNamespace(tests=('SWIR1',), experiment_name='exp')
    monkeypatch.setattr(cache, 'cache_root', lambda spec: tmp_path)
    values = [dict(router='rankcv', dataset='SWIR1', count=2, filled_voxels=2,
                   rmse=.2, mae=.1, squared_error=.08, region_voxels=1,
                   r2=.9, sam_deg=.7, ssim=.8),
              dict(router='rankcv', dataset='SWIR1', count=2, filled_voxels=2,
                   rmse=.4, mae=.3, squared_error=.32, region_voxels=2,
                   r2=.8, sam_deg=.5, ssim=.9)]
    cache.atomic_csv(tmp_path/'score_SWIR1.csv', values)
    cache.score_summary(spec, ['rankcv'])
    with (tmp_path/'SCORE_SUMMARY.csv').open(newline='') as stream:
        item = next(csv.DictReader(stream))
    assert float(item['mean_case_rmse']) == pytest.approx(.3)
    assert float(item['pooled_voxel_rmse']) == pytest.approx(np.sqrt(.1))
    assert float(item['coverage']) == 1.
    assert float(item['mean_ssim']) == pytest.approx(.85)
