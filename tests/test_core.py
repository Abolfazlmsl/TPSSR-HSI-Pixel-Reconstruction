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


def test_spectral_validation_does_not_include_test_acquisitions():
    from argparse import Namespace
    from experiment_config import resolve_experiment, spectral_validation_cores
    expected = {
        "swir_8x2": ("SWIR4", "SWIR9"),
        "swir_region_a_to_b": ("SWIR4", "SWIR5"),
        "swir_region_b_to_a": ("SWIR9", "SWIR10"),
        "vnir_8x2": ("VNIR4", "VNIR9"),
    }
    for preset, validation in expected.items():
        spec = resolve_experiment(Namespace(preset=preset, train_datasets=None,
            test_datasets=None, dataset=None, active_test=None, experiment_name=None))
        assert spectral_validation_cores(spec) == validation
        assert not set(validation) & set(spec.tests)
        assert set(validation).issubset(spec.train)
        assert len([x for x in spec.train if x not in validation]) >= 3


def test_fusion_inputs_never_read_synthetic_hidden_truth():
    from training.neighbor_fusion import crop_at, make_features
    bands = 79
    rng = np.random.default_rng(177)
    cube = (.3 + .07*np.sin(np.arange(bands)[None,None,:]/7.)
            + rng.normal(0,.005,(33,35,1))).astype(np.float32)
    damaged = np.zeros_like(cube,dtype=bool)
    damaged[13:23,12:24,25:40] = True
    donor_damage = damaged.copy()
    donor_damage[4,8,65] = True
    modified = cube.copy()
    modified[donor_damage] = 9000.
    mean = np.zeros(bands,np.float32)
    std = np.ones(bands,np.float32)
    wave = np.arange(bands,dtype=np.float32)
    a,ma,cy,cx = crop_at(cube,donor_damage,18,18)
    b,mb,dy,dx = crop_at(modified,donor_damage,18,18)
    first=make_features(a,ma,cy,cx,mean,std,wave)
    second=make_features(b,mb,dy,dx,mean,std,wave)
    for lhs,rhs in zip(first,second):
        np.testing.assert_array_equal(lhs,rhs)


def test_fusion_module_gradients_and_no_donor_fallback():
    import torch
    from training.neighbor_fusion import NeighborFusion, MAX_DONORS
    torch.manual_seed(92)
    length=65
    feat=torch.zeros((2,length,11),dtype=torch.float32)
    feat[:,:,1]=.25
    feat[:,:,2]=1.
    feat[:,:,3]=torch.linspace(0,1,length)
    hybrid=torch.full((2,length),.5)
    donor=torch.zeros((2,MAX_DONORS,length))
    quality=torch.zeros((2,MAX_DONORS,2))
    present=torch.zeros((2,MAX_DONORS))
    model=NeighborFusion(width=16)
    result=model(feat,hybrid,donor,quality,present)
    assert torch.isfinite(result).all()
    assert (result >= .25).all() and (result <= .5).all()
    loss=(result**2).mean()
    loss.backward()
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()


def test_synthetic_corruption_can_shift_band_center():
    from src.training_corruptions import generate_training_mask
    rng=np.random.default_rng(941)
    center=8
    mask, family, severity = generate_training_mask((29,29,100),rng,
        severity=4, scenario='consecutive_spectral_bands',band_center=center)
    assert family=='consecutive_spectral_bands'
    assert severity==4
    assert mask[14,14,center]
    assert mask[14,14,50]==False


def test_fusion_validation_does_not_sample_whole_spectrum_center():
    from training.neighbor_fusion import TRAIN_CASES
    assert ('mixed_saturation',4) not in TRAIN_CASES
    assert ('mixed_saturation',5) not in TRAIN_CASES
    assert all(scenario!='whole_spectrum_pixels' for scenario,_ in TRAIN_CASES)


def test_learned_fusion_evaluation_keeps_v2_region_and_hides_truth(monkeypatch,tmp_path):
    import torch
    import config
    import src.pipeline as pipeline_module
    from types import SimpleNamespace
    bands=55
    cube=np.broadcast_to(np.linspace(.20,.45,bands,dtype=np.float32),(9,9,bands)).copy()
    missing=np.zeros_like(cube,bool)
    missing[4,4,17:28]=True
    missing[2,3,:]=True
    target=np.asarray([[4,4,19],[4,4,25],[2,3,25]],dtype=np.int64)
    folder=tmp_path/config.TEST_DATASET_FOLDER
    folder.mkdir(parents=True)
    np.save(folder/'wavelengths_nm.npy',np.arange(bands,dtype=np.float32))
    monkeypatch.setattr(config,'PROCESSED_DATA_ROOT',tmp_path)

    def v2_stub(_model,_cube,_mask,coords,**kw):
        route=np.array([0,0,2])
        return {'prediction':np.array([0.,0.,.47],np.float32),
                'final_route':route,
                'spectral_prediction':np.array([0.,0.,np.nan],np.float32),
                'candidate_method':np.array(['interp','interp','region'],dtype='<U20'),
                'candidate_method_voxels':{},'candidate_policy_audit':{}}
    monkeypatch.setattr(pipeline_module,'predict_mask_candidate_v2',v2_stub)

    class IdentityHybrid(torch.nn.Module):
        def forward(self,feat):return feat[...,1]

    class IdentityFusion(torch.nn.Module):
        def forward(self,feat,hybrid,donors,meta,valid):return feat[...,1]

    dummy=SimpleNamespace(device=torch.device('cpu'),hybrid_batch=2,
        hybrid_mean=np.zeros(bands,np.float32),hybrid_std=np.ones(bands,np.float32),
        hybrid_model=IdentityHybrid(),fusion_model=IdentityFusion())
    output=pipeline_module.predict_mask_learned_fusion(dummy,cube,missing,target)
    changed=cube.copy()
    changed[missing]=10000.
    output2=pipeline_module.predict_mask_learned_fusion(dummy,changed,missing,target)
    np.testing.assert_array_equal(output['prediction'],output2['prediction'])
    np.testing.assert_allclose(output['prediction'][:2],cube[target[:2,0],target[:2,1],target[:2,2]],rtol=0,atol=1e-6)
    assert output['prediction'][2]==.47
    assert output['candidate_method_voxels']['learned_fusion']==2


def test_guarded_fusion_preserves_v2_protected_voxels(monkeypatch, tmp_path):
    import torch
    import config
    import src.pipeline as pipeline_module
    from types import SimpleNamespace

    bands = 61
    cube = np.broadcast_to(np.linspace(.20, .50, bands, dtype=np.float32),
                           (9, 9, bands)).copy()
    damaged = np.zeros_like(cube, dtype=bool)
    damaged[4, 4, 18:38] = True
    damaged[2, 3, :] = True
    coords = np.array([[4, 4, 22], [4, 4, 32], [2, 3, 25]], np.int64)

    folder = tmp_path / config.TEST_DATASET_FOLDER
    folder.mkdir(parents=True)
    np.save(folder / 'wavelengths_nm.npy', np.arange(bands, dtype=np.float32))
    monkeypatch.setattr(config, 'PROCESSED_DATA_ROOT', tmp_path)

    def v2_stub(_pipeline, _cube, _mask, _coords, **kwargs):
        return {
            'prediction': np.array([.91, .92, .47], np.float32),
            'final_route': np.array([0, 0, 2]),
            'spectral_prediction': np.array([.91, .92, np.nan], np.float32),
            'candidate_method': np.array(['band_guard', 'hybrid', 'region'], dtype='<U20'),
            'candidate_method_voxels': {'band_guard': 1},
            'candidate_policy_audit': {'v2_band_guard_voxels': 1},
        }

    monkeypatch.setattr(pipeline_module, 'predict_mask_candidate_v2', v2_stub)

    class IdentityHybrid(torch.nn.Module):
        def forward(self, features):
            return features[..., 1]

    class IdentityFusion(torch.nn.Module):
        def forward(self, features, hybrid, donors, quality, present):
            return features[..., 1]

    bundle = SimpleNamespace(
        device=torch.device('cpu'), hybrid_batch=2,
        hybrid_mean=np.zeros(bands, np.float32),
        hybrid_std=np.ones(bands, np.float32),
        hybrid_model=IdentityHybrid(), fusion_model=IdentityFusion(),
    )
    guarded = pipeline_module.predict_mask_learned_fusion_guarded(
        bundle, cube, damaged, coords)
    original = pipeline_module.predict_mask_learned_fusion(
        bundle, cube, damaged, coords)

    assert guarded['prediction'][0] == np.float32(.91)
    assert original['prediction'][0] != np.float32(.91)
    assert guarded['candidate_method'][0] == 'band_guard'
    assert guarded['candidate_method'][1] == 'learned_fusion'
    assert guarded['prediction'][2] == np.float32(.47)
    assert guarded['candidate_policy_audit']['fusion_v2_band_guard_preserved_voxels'] == 1
    assert guarded['candidate_method_voxels']['band_guard'] == 1
    assert guarded['candidate_method_voxels']['learned_fusion'] == 1

    altered = cube.copy()
    altered[damaged] = 9999.
    guarded_altered = pipeline_module.predict_mask_learned_fusion_guarded(
        bundle, altered, damaged, coords)
    np.testing.assert_array_equal(guarded['prediction'], guarded_altered['prediction'])


def test_guarded_fusion_all_spectral_voxels_protected(monkeypatch):
    import torch
    import src.pipeline as pipeline_module
    from types import SimpleNamespace

    def v2_stub(_pipeline, _cube, _mask, _coords, **kwargs):
        return {
            'prediction': np.array([.42, .71], np.float32),
            'final_route': np.array([0, 2]),
            'candidate_method': np.array(['band_guard', 'region'], dtype='<U20'),
            'candidate_policy_audit': {},
        }

    monkeypatch.setattr(pipeline_module, 'predict_mask_candidate_v2', v2_stub)
    bundle = SimpleNamespace(fusion_model=torch.nn.Identity())
    out = pipeline_module.predict_mask_learned_fusion_guarded(
        bundle, np.zeros((2, 2, 5), np.float32),
        np.zeros((2, 2, 5), bool), np.array([[0, 0, 1], [0, 1, 2]], np.int64))
    np.testing.assert_array_equal(out['prediction'], [np.float32(.42), np.float32(.71)])
    assert out['candidate_policy_audit']['fusion_v2_band_guard_preserved_voxels'] == 1
    assert out['candidate_policy_audit']['learned_fusion_voxels'] == 0


def test_router_features_use_only_masked_observations():
    from training.neighbor_fusion import crop_at, make_features
    from training.learned_router import feature_row, FEATURE_NAMES
    rng = np.random.default_rng(2026)
    bands = 55
    cube = rng.uniform(.2, .6, size=(33, 33, bands)).astype(np.float32)
    missing = np.zeros_like(cube, bool)
    missing[14:20, 14:20, 15:33] = True
    modified = cube.copy()
    modified[missing] = 10000
    params = (np.zeros(bands, np.float32), np.ones(bands, np.float32),
              np.arange(bands, dtype=np.float32))
    features = []
    for raw in (cube, modified):
        patch, bad, cy, cx = crop_at(raw, missing, 16, 16)
        f = make_features(patch, bad, cy, cx, *params)
        assert f is not None
        fusion_prediction = f[0][:, 1] + 0.05 * f[0][:, 2]
        features.append(feature_row(f[0], f[2], f[3], f[4], fusion_prediction))
    assert len(FEATURE_NAMES) == len(features[0])
    np.testing.assert_array_equal(*features)


def test_learned_router_preserves_band_guard_region_and_hidden_truth(monkeypatch, tmp_path):
    import torch
    from types import SimpleNamespace
    import src.pipeline as pipeline_module
    import config

    bands = 61
    cube = np.broadcast_to(np.linspace(.2, .5, bands, dtype=np.float32),
                           (9, 9, bands)).copy()
    mask = np.zeros_like(cube, bool)
    mask[4, 4, 18:38] = True
    mask[2, 3, :] = True
    coords = np.array([[4, 4, 22], [4, 4, 32], [2, 3, 25]], np.int64)
    folder = tmp_path / config.TEST_DATASET_FOLDER
    folder.mkdir(parents=True)
    np.save(folder / 'wavelengths_nm.npy', np.arange(bands, dtype=np.float32))
    monkeypatch.setattr(config, 'PROCESSED_DATA_ROOT', tmp_path)

    def v2_stub(_pipeline, _cube, _mask, _coords, **kw):
        return {'prediction': np.array([.91, .92, .47], np.float32),
                'final_route': np.array([0, 0, 2]),
                'spectral_prediction': np.array([.91, .92, np.nan], np.float32),
                'candidate_method': np.array(['band_guard', 'hybrid', 'region'], dtype='<U20'),
                'candidate_method_voxels': {}, 'candidate_policy_audit': {}}

    monkeypatch.setattr(pipeline_module, 'predict_mask_candidate_v2', v2_stub)

    class Hybrid(torch.nn.Module):
        def forward(self, x):
            return x[..., 1]

    class Fusion(torch.nn.Module):
        def forward(self, feat, hy, donors, meta, valid):
            return feat[..., 1] + .05 * feat[..., 2]

    class SelectInterpolation:
        def predict(self, features):
            return np.ones(len(features), np.float32)

    model = SimpleNamespace(device=torch.device('cpu'), hybrid_batch=2,
                            hybrid_mean=np.zeros(bands, np.float32),
                            hybrid_std=np.ones(bands, np.float32),
                            hybrid_model=Hybrid(), fusion_model=Fusion(),
                            router_checkpoint={'enabled': True, 'model': SelectInterpolation()})
    output = pipeline_module.predict_mask_learned_router(model, cube, mask, coords)
    altered = cube.copy()
    altered[mask] = -9999
    output2 = pipeline_module.predict_mask_learned_router(model, altered, mask, coords)
    np.testing.assert_array_equal(output['prediction'], output2['prediction'])
    assert output['prediction'][0] == np.float32(.91)
    assert output['prediction'][2] == np.float32(.47)
    assert output['candidate_method'][0] == 'band_guard'
    assert output['candidate_method'][1] == 'router_interp'
    assert output['candidate_method'][2] == 'region'
    assert output['candidate_policy_audit']['learned_router_interp_voxels'] == 1
    assert '_router_features' not in output


def test_router_internal_fit_has_distinct_calibration_cores(tmp_path):
    from training.learned_router import fit_selector, load_router_checkpoint, FEATURE_NAMES
    rng = np.random.default_rng(17)
    n = 500
    x = rng.uniform(-1, 1, (n, len(FEATURE_NAMES))).astype(np.float32)
    cores = np.repeat([0, 1], n//2)
    # Both outcomes occur; the learner must never fit on the heldout half.
    sq_s = np.where(x[:, 0] > 0, .001, .1).astype(np.float64)
    sq_f = np.where(x[:, 0] > 0, .1, .001).astype(np.float64)
    vox = np.ones(n, np.float64) * 10
    gate = tmp_path / 'gate.pt'
    fusion = tmp_path / 'fusion.pt'
    gate.write_bytes(b'gate')
    fusion.write_bytes(b'fusion')
    payload = fit_selector(x, sq_s, sq_f, vox, cores, ('SWIR4', 'SWIR5'), gate, fusion)
    assert payload['fit_cores'] == ['SWIR4']
    assert payload['holdout_cores'] == ['SWIR5']
    assert payload['fit_cases'] == payload['holdout_cases'] == n//2
    import joblib
    ck = tmp_path / 'router.joblib'
    joblib.dump(payload, ck)
    loaded = load_router_checkpoint(ck, gate, fusion, ('SWIR4', 'SWIR5'))
    assert loaded['architecture'] == payload['architecture']
    import pytest
    with pytest.raises(RuntimeError):
        load_router_checkpoint(ck, gate, fusion, ('SWIR9', 'SWIR10'))
"""Tests for actual learned three-way decision routing."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from training.three_expert_router import features_for_pixel, predict_risks, FEATURE_NAMES
from src.pipeline import predict_mask_learned_router3


class Hybrid(torch.nn.Module):
    def forward(self, h):
        return h[..., 1]


class Fusion(torch.nn.Module):
    def forward(self, f, h, donors, meta, valid):
        return f[..., 1] + f[..., 2] * .1


class ConstantRisk:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, features):
        return np.full(len(features), self.value, np.float64)


def sample():
    rng = np.random.default_rng(212)
    b = 40
    cube = (rng.random((9,9,b)) * .3 + .05).astype(np.float32)
    mask = np.zeros_like(cube,bool)
    mask[4,4,15:18] = True
    coords = np.asarray([[4,4,k] for k in range(15,18)],np.int64)
    return cube,mask,coords


def bundle(chosen):
    models = [ConstantRisk(3.0) for _ in range(3)]
    models[chosen] = ConstantRisk(-5.0)
    return SimpleNamespace(
        fusion_model=Fusion(), hybrid_model=Hybrid(),
        router3_checkpoint={'models':models},
        hybrid_mean=np.zeros(40,np.float32),
        hybrid_std=np.ones(40,np.float32),
        device=torch.device('cpu'),
        topology_model=object(), topology_checkpoint={}, geometry_model=object(),
        region_batch=4,verify_region_batch=False,
    )


def mock_region(*args, **kwargs):
    coords = kwargs['eval_coords']
    return ({int(b): (np.array([[4,4]]),np.array([.7],np.float32)) for b in coords[:,2]},
            np.unique(coords[:,2]))


def test_observed_only_features_unchanged_when_hidden_truth_changes():
    cube,mask,_ = sample()
    args=(4,4,np.zeros(40,np.float32),np.ones(40,np.float32),
          np.arange(40,dtype=np.float32),Fusion(),Hybrid(),torch.device('cpu'))
    a,ia,fa=features_for_pixel(cube,mask,*args)
    replaced=cube.copy();replaced[mask] = 1000.
    b,ib,fb=features_for_pixel(replaced,mask,*args)
    np.testing.assert_allclose(a,b,atol=0,rtol=0)
    np.testing.assert_allclose(ia,ib,atol=0,rtol=0)
    np.testing.assert_allclose(fa,fb,atol=0,rtol=0)


@pytest.mark.parametrize('chosen,name',[(0,'router3_interp'),(1,'router3_fusion'),(2,'router3_region')])
def test_three_way_prediction_is_learned_not_v2(tmp_path,monkeypatch,chosen,name):
    cube,mask,coords=sample()
    import src.pipeline as pipeline
    monkeypatch.setattr(pipeline.config, 'PROCESSED_DATA_ROOT',tmp_path)
    monkeypatch.setattr(pipeline.config, 'TEST_DATASET_FOLDER','SWIR6')
    folder=tmp_path/'SWIR6';folder.mkdir()
    np.save(folder/'wavelengths_nm.npy',np.arange(40,dtype=np.float32))
    monkeypatch.setattr(pipeline, 'reconstruct_final_region',mock_region)
    monkeypatch.setattr(pipeline, 'predict_mask_candidate_v2',
        lambda *a,**k: (_ for _ in ()).throw(RuntimeError('V2 should never be called')))
    out=predict_mask_learned_router3(bundle(chosen),cube,mask,coords)
    assert np.isfinite(out['prediction']).all()
    assert set(out['candidate_method']) == {name}
    assert np.array_equal(out['final_route'],np.full(3,2 if chosen==2 else 0))
    assert out['candidate_policy_audit'][name+'_voxels'] == 3


def test_region_available_without_spectral_anchors(tmp_path,monkeypatch):
    cube,mask,coords=sample()
    mask[4,4,:] = True
    coords=np.asarray([[4,4,k] for k in range(40)],np.int64)
    import src.pipeline as pipeline
    monkeypatch.setattr(pipeline.config,'PROCESSED_DATA_ROOT',tmp_path)
    monkeypatch.setattr(pipeline.config,'TEST_DATASET_FOLDER','SWIR6')
    folder=tmp_path/'SWIR6';folder.mkdir()
    np.save(folder/'wavelengths_nm.npy',np.arange(40,dtype=np.float32))
    monkeypatch.setattr(pipeline,'reconstruct_final_region',mock_region)
    result=predict_mask_learned_router3(bundle(0),cube,mask,coords)
    assert set(result['candidate_method']) == {'router3_region'}


def test_risk_regressor_masks_only_infeasible_experts():
    payload={'models':[ConstantRisk(-4),ConstantRisk(-2),ConstantRisk(-1)]}
    x=np.zeros((2,len(FEATURE_NAMES)),np.float32)
    risks=predict_risks(payload,x,np.array([[True,True,True],[False,False,True]]))
    assert risks[0].argmin()==0
    assert risks[1].argmin()==2
    assert np.isinf(risks[1,:2]).all()


def test_three_expert_squared_error_training_has_separate_checkpoint(tmp_path):
    from training.three_expert_router import fit_models, path_for, ARCH_MSE
    rng = np.random.default_rng(500)
    x = rng.random((280, len(FEATURE_NAMES))).astype(np.float32)
    err = np.empty((280, 3), np.float64)
    err[:, 0] = .015 + x[:, 0] * .01
    err[:, 1] = .005 + x[:, 1] * .004
    err[:, 2] = .03 + x[:, 2] * .02
    available = np.ones_like(err, bool)
    core = np.repeat([0, 1], 140)
    cases = [('spatial_patch', (i % 5) + 1) for i in range(280)]
    fit = fit_models(x, err, available, core, ('SWIR4', 'SWIR5'),
                     objective='mse', case_meta=cases)
    assert fit['risk_target'] == 'mse'
    assert fit['architecture'] == ARCH_MSE
    assert len(fit['_holdout_details']) == 140
    assert fit['holdout_router_rmse'] > 0
    assert fit['holdout_fusion_rmse'] < fit['holdout_expert_rmse']['region']
    risks = predict_risks(fit, x[core == 1], available[core == 1])
    assert np.isfinite(risks).all()
    assert (risks >= 0).all()
    assert path_for(tmp_path, 'mse') != path_for(tmp_path, 'log_rmse')


def test_three_expert_mse_prediction_honors_feasibility():
    from training.three_expert_router import ARCH_MSE
    x = np.zeros((2, len(FEATURE_NAMES)), np.float32)
    payload = {'architecture': ARCH_MSE, 'risk_target': 'mse',
               'models': [ConstantRisk(.04), ConstantRisk(.01), ConstantRisk(.25)]}
    risks = predict_risks(payload, x, [[True, True, True], [False, False, True]])
    np.testing.assert_allclose(risks[0], [.2, .1, .5])
    assert risks[0].argmin() == 1
    assert risks[1].argmin() == 2
    assert np.isinf(risks[1, :2]).all()


def test_aligned_region_refinement_uses_only_measured_neighbors():
    from training.three_expert_router import refine_region
    nband = 48
    cube = np.full((9, 9, nband), .42, np.float32)
    mask = np.zeros_like(cube, bool)
    mask[4, 4, :] = True
    coords = np.column_stack((np.full(nband, 4), np.full(nband, 4), np.arange(nband)))
    raw = np.full(nband, .91, np.float32)
    corrected, count = refine_region(cube, mask, coords, raw)
    assert count == nband
    np.testing.assert_array_equal(corrected, np.full(nband, .42, np.float32))
    assert np.array_equal(raw, np.full(nband, .91, np.float32))
    changed = cube.copy()
    changed[mask] = 10000.
    corrected2, count2 = refine_region(changed, mask, coords, raw)
    assert count2 == count
    np.testing.assert_array_equal(corrected2, corrected)


def test_aligned_region_preserves_raw_when_support_is_missing():
    from training.three_expert_router import refine_region
    cube = np.full((9, 9, 48), .42, np.float32)
    mask = np.zeros_like(cube, bool)
    mask[3:6, 3:6] = True
    coords = np.column_stack((np.full(48, 4), np.full(48, 4), np.arange(48)))
    result, count = refine_region(cube, mask, coords, np.full(48, .85, np.float32))
    assert count == 0
    np.testing.assert_array_equal(result, np.full(48, .85, np.float32))


def test_aligned_learned_region_choice_not_v2(tmp_path, monkeypatch):
    from training.three_expert_router import path_for
    import src.pipeline as pipeline
    cube, mask, _ = sample()
    mask[4, 4, :] = True
    cube[:, :, :] = np.float32(.40)
    coords = np.asarray([[4, 4, k] for k in range(40)], np.int64)
    monkeypatch.setattr(pipeline.config, 'PROCESSED_DATA_ROOT', tmp_path)
    monkeypatch.setattr(pipeline.config, 'TEST_DATASET_FOLDER', 'SWIR6')
    folder = tmp_path / 'SWIR6'
    folder.mkdir()
    np.save(folder / 'wavelengths_nm.npy', np.arange(40, dtype=np.float32))
    monkeypatch.setattr(pipeline, 'reconstruct_final_region', mock_region)
    monkeypatch.setattr(pipeline, 'predict_mask_candidate_v2',
        lambda *a, **k: (_ for _ in ()).throw(AssertionError('V2 must not route')))
    model = bundle(2)
    model.router3_checkpoint['region_expert'] = 'refined'
    out = predict_mask_learned_router3(model, cube, mask, coords)
    assert set(out['candidate_method']) == {'router3_region'}
    np.testing.assert_allclose(out['prediction'], np.float32(.40), atol=1e-7)
    assert out['candidate_policy_audit']['router3_region_refined_voxels'] == 40
    assert path_for(tmp_path, region_expert='refined') != path_for(tmp_path)


def test_aligned_training_label_uses_same_refined_region_expert(monkeypatch):
    import training.three_expert_router as router
    monkeypatch.setattr(router, '_lookup_region',
        lambda model, cube, mask, coords, device, batch:
            np.full(len(coords), .90, dtype=np.float32))
    cube = np.full((65, 65, 48), .42, np.float32)
    invalid = np.zeros_like(cube, bool)
    common = (cube, invalid, None, 'whole_spectrum_pixels', 1,
              np.zeros(48, np.float32), np.ones(48, np.float32),
              np.arange(48, dtype=np.float32), None, None, None,
              torch.device('cpu'), 4)
    raw = router.make_training_case(*common[:2], np.random.default_rng(4), *common[3:],
                                    region_expert='raw')
    refined = router.make_training_case(*common[:2], np.random.default_rng(4), *common[3:],
                                        region_expert='refined')
    assert raw[2].tolist() == refined[2].tolist() == [False, False, True]
    assert raw[1][2] > .45
    assert refined[1][2] < 1e-6
    np.testing.assert_array_equal(raw[0], refined[0])


def test_router3_shadow_split_on_main_eight_train_two_test():
    from argparse import Namespace
    from experiment_config import resolve_experiment, spectral_validation_cores
    from training.three_expert_router import shadow_experiment
    spec = resolve_experiment(Namespace(preset='swir_8x2', train_datasets=None,
        test_datasets=None, dataset=None, active_test=None, experiment_name=None))
    val = spectral_validation_cores(spec)
    shadow_name, shadow_train = shadow_experiment(spec, val)
    assert val == ('SWIR4', 'SWIR9')
    assert shadow_train == ('SWIR1', 'SWIR2', 'SWIR3', 'SWIR6', 'SWIR7', 'SWIR8')
    assert not set(shadow_train) & set(spec.tests)
    assert shadow_name == spec.experiment_name + '__router3_region_oof'


def test_router3_shadow_split_preserves_five_core_protocols():
    from argparse import Namespace
    from experiment_config import resolve_experiment, spectral_validation_cores
    from training.three_expert_router import shadow_experiment
    for preset, expected in (
        ('swir_region_a_to_b', ('SWIR1', 'SWIR2', 'SWIR3')),
        ('swir_region_b_to_a', ('SWIR6', 'SWIR7', 'SWIR8')),
    ):
        spec = resolve_experiment(Namespace(preset=preset, train_datasets=None,
            test_datasets=None, dataset=None, active_test=None, experiment_name=None))
        _name, shadow_train = shadow_experiment(spec, spectral_validation_cores(spec))
        assert shadow_train == expected
