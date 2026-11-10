"""The replay path must exactly select the cached counterfactual values."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import router_expert_cache as cache


class FakeRegressor:
    def __init__(self, score):
        self.score = score

    def predict(self, x):
        return np.full(len(x), self.score, dtype=np.float64)


def fake_payload():
    from training.three_expert_router import FEATURE_NAMES, EXPERTS, ARCH_RANK_CV
    return {'architecture': ARCH_RANK_CV, 'feature_names': FEATURE_NAMES,
            'expert_names': EXPERTS, 'models': [FakeRegressor(-5), FakeRegressor(-6), FakeRegressor(-4)],
            'risk_target': 'log_rmse', 'model_family': 'log_risk_extratrees'}


def test_replay_selects_three_expert_predictions_per_pixel():
    from training.three_expert_router import FEATURE_NAMES
    record = {
        'features': np.zeros((3, len(FEATURE_NAMES)), np.float32),
        'eligible': np.asarray([[1, 1, 1], [0, 0, 1], [1, 1, 1]], bool),
        'reverse': np.asarray([0, 0, 1, 2, 2]),
        'predictions': np.asarray([[10, 11, 12], [20, 21, 22],
                                   [np.nan, np.nan, 33], [40, 41, 42], [50, 51, 52]], np.float32),
        'truth': np.asarray([11, 21, 30, 41, 51], np.float32),
    }
    t, p, selected = cache.reconstruct_from_cache(record, fake_payload())
    assert selected.tolist() == [1, 1, 2, 1, 1]
    assert p.tolist() == [11, 21, 33, 41, 51]
    assert cache.case_score(t, p)[0] == pytest.approx((9 / 5) ** .5)


def test_atomic_npz_resume_and_strict_input_version(tmp_path):
    from training.three_expert_router import FEATURE_NAMES
    original = tmp_path / 'mask_a.npz'
    original.write_bytes(b'mask')
    cube = tmp_path / 'calibrated_reflectance.npy'
    cube.write_bytes(b'cube')
    spec = SimpleNamespace(experiment_name='exp', train=('SWIR6',), tests=('SWIR1',))
    payload = dict(metadata=np.asarray(json.dumps(dict(cache_version=cache.CACHE_VERSION,
        experiment='exp', dataset='SWIR1', mask_file='mask_a.npz',
        mask_stat=cache.input_stat(original), cube_stat=cache.input_stat(cube),
        evaluation_only=True, region_expert='lineaware',
        region_tiling_protocol='fixed_union_of_sampled_coordinates_v1', n_voxels=2))),
        coords=np.zeros((2, 3), np.int32), reverse=np.zeros(2, np.int32),
        features=np.zeros((1, len(FEATURE_NAMES)), np.float32),
        eligible=np.ones((1,3),bool), predictions=np.ones((2,3),np.float32),
        truth=np.zeros(2,np.float32))
    path = tmp_path / 'cached.npz'
    cache.atomic_npz(path, payload)
    assert cache.validate_case(path, spec, 'SWIR1', original, cube)['n_voxels'] == 2
    original.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='Stale'):
        cache.validate_case(path, spec, 'SWIR1', original, cube)


def test_no_ineligible_expert_wins_replay():
    from training.three_expert_router import FEATURE_NAMES
    x = np.ones((2, len(FEATURE_NAMES)), np.float32)
    options = np.asarray([[1, 1, 1], [0, 0, 1]], bool)
    choices = cache.predict_from_cache(fake_payload(), x, options)
    assert choices.tolist() == [1, 2]


def test_case_average_and_pooled_are_distinct():
    # Individual case aggregation and per-voxel aggregation have different weights.
    r1, _, e1, n1 = cache.case_score([0, 0], [1, 1])
    r2, _, e2, n2 = cache.case_score([0], [3])
    assert np.mean([r1, r2]) == pytest.approx(2)
    assert np.sqrt((e1 + e2) / (n1 + n2)) == pytest.approx(np.sqrt(11 / 3))


def test_cost_router_masks_missing_experts():
    from training.three_expert_router import (ARCH_REGRET_CV, FEATURE_NAMES,
                                              predict_risks, _regret_weights)
    from sklearn.dummy import DummyClassifier
    x = np.zeros((4, len(FEATURE_NAMES)), np.float32)
    x[:,1] = .5
    oracle = DummyClassifier(strategy='most_frequent').fit(x, [1,1,1,1])
    payload = dict(architecture=ARCH_REGRET_CV, model_family='cost_oracle',
                   risk_target='cost_sensitive_ranking', oracle_model=oracle)
    availability = np.array([[True,True,True],[False,False,True],
                             [True,False,True],[True,True,False]])
    risks = predict_risks(payload, x, availability)
    assert risks.shape == (4,3)
    assert np.argmin(risks,axis=1).tolist() == [1,2,0,1]
    weights = _regret_weights([0,1e-10,1,100], [1,1,1,1])
    assert np.isfinite(weights).all() and np.all(weights > 0)


def test_crossfit_cost_sensitive_uses_held_out_core():
    from training.three_expert_router import (FEATURE_NAMES, _crossfit_regret)
    n = 120
    rng=np.random.default_rng(123)
    x=rng.random((n,len(FEATURE_NAMES))).astype(np.float32)
    x[:,1]=.35
    err=np.column_stack((.03+.01*x[:,0], .005+.005*x[:,1], .01+.01*x[:,2]))
    present=np.ones((n,3), bool)
    # RegretCV must not select unavailable experts even when availability differs.
    present[::10,:2]=False
    err[~present]=np.nan
    core=np.repeat([0,1],n//2)
    winner, board=_crossfit_regret(x,err,present,core)
    assert winner in ('log_risk_extratrees','cost_oracle','cost_pairwise')
    assert len([row for row in board if 'mean_case_rmse' in row]) >= 1

