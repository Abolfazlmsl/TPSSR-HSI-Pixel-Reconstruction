from experiment_config import ExperimentSpec, selected_lineaware_region, selected_results_folder


def _spec(preset):
    if preset == 'swir_8x2':
        return ExperimentSpec(('SWIR1','SWIR2','SWIR3','SWIR4','SWIR6','SWIR7','SWIR8','SWIR9'),
                              ('SWIR5','SWIR10'), 'SWIR5', 'swir_8x2', 'SWIR')
    if preset == 'vnir_8x2':
        return ExperimentSpec(('VNIR1','VNIR2','VNIR3','VNIR4','VNIR6','VNIR7','VNIR8','VNIR9'),
                              ('VNIR5','VNIR10'), 'VNIR5', 'vnir_8x2', 'VNIR')
    return ExperimentSpec(('SWIR1','SWIR2','SWIR3','SWIR4','SWIR5'),
                          ('SWIR6','SWIR7','SWIR8','SWIR9','SWIR10'), 'SWIR6', 'a_to_b', 'SWIR')


def test_selected_lineaware_variant_and_result_isolation():
    spec = _spec('swir_8x2')
    assert selected_lineaware_region(spec, 'learned_router3_aligned')
    assert selected_results_folder(spec, 'learned_router3_aligned') == 'tpssr_learned_router3_aligned_lineaware'
    assert selected_results_folder(spec, 'learned_router3_rmse') == 'tpssr_learned_router3_rmse'


def test_other_presets_keep_their_own_region():
    for key in ('vnir_8x2','a_to_b'):
        spec = _spec(key)
        assert not selected_lineaware_region(spec, 'learned_router3_aligned')
        assert selected_results_folder(spec, 'learned_router3_aligned') == 'tpssr_learned_router3_aligned'


def test_new_router_results_are_separate_for_all_swir_splits():
    from training.three_expert_router import path_for, shadow_experiment
    from experiment_config import spectral_validation_cores

    for split in ('swir_8x2', 'a_to_b'):
        spec = _spec(split)
        assert selected_lineaware_region(spec, 'learned_router3_lineaware')
        assert selected_results_folder(spec, 'learned_router3_lineaware') == 'tpssr_learned_router3_lineaware'
        validation = spectral_validation_cores(spec)
        shadow_name, shadow_train = shadow_experiment(spec, validation)
        assert shadow_name.endswith('__router3_region_oof')
        assert not set(shadow_train).intersection(set(spec.tests) | set(validation))
    b2a = ExperimentSpec(('SWIR6','SWIR7','SWIR8','SWIR9','SWIR10'),
                          ('SWIR1','SWIR2','SWIR3','SWIR4','SWIR5'),
                          'SWIR1', 'b_to_a', 'SWIR')
    assert selected_lineaware_region(b2a, 'learned_router3_lineaware')
    assert selected_lineaware_region(_spec('vnir_8x2'), 'learned_router3_lineaware')
    base = 'some_models'
    assert path_for(base, region_expert='lineaware').name == 'best_model.joblib'
    assert path_for(base, region_expert='lineaware').parent.name == 'learned_router3_lineaware'
    assert path_for(base, objective='mse', region_expert='lineaware').parent.name == 'learned_router3_lineaware_mse'
    assert path_for(base, region_expert='refined').parent.name == 'learned_router3_aligned'


def test_lineaware_paths_use_primary_model_for_every_active_core(monkeypatch, tmp_path):
    import config
    from src.lineaware_region import checkpoint_paths
    base = tmp_path / 'trained_models'
    monkeypatch.setattr(config, 'MODELS_DIR', base)
    monkeypatch.setattr(config, 'TRAIN_DATASET_FOLDERS',
                        ['SWIR6','SWIR7','SWIR8','SWIR9','SWIR10'])
    monkeypatch.setattr(config, 'TEST_DATASET_FOLDERS',
                        ['SWIR1','SWIR2','SWIR3','SWIR4','SWIR5'])
    monkeypatch.delenv('TPSSR_B2A_LINEAWARE_TEST', raising=False)
    for test_name in ('SWIR1','SWIR3','SWIR5'):
        monkeypatch.setattr(config, 'TEST_DATASET_FOLDER', test_name)
        geometry, topology = checkpoint_paths(full_split=True)
        assert geometry == base / 'tpssr/region_lineaware_trial_v1/geometry_swir1/best_model.pt'
        assert topology == base / 'tpssr/region_lineaware_trial_v1/topology_swir1/best_model.pt'


def test_new_router_checkpoint_requires_lineaware_region_fingerprint(tmp_path):
    import joblib
    import pytest
    from training.spectral_gate import sha
    from training.three_expert_router import ARCH, EXPERTS, FEATURE_NAMES, load_checkpoint

    paths = [tmp_path / f'weights_{i}.bin' for i in range(4)]
    for i, path in enumerate(paths):
        path.write_bytes(f'weights_{i}'.encode())
    ck = tmp_path / 'new_router.joblib'
    payload = dict(architecture=ARCH, feature_names=FEATURE_NAMES,
                   expert_names=EXPERTS, risk_target='log_rmse',
                   region_expert='lineaware', source_cores=['SWIR9','SWIR10'],
                   gate_sha256=sha(paths[0]), fusion_sha256=sha(paths[1]),
                   region_geometry_sha256=sha(paths[2]),
                   region_topology_sha256=sha(paths[3]))
    joblib.dump(payload, ck)
    checked = load_checkpoint(ck, *paths, ('SWIR9','SWIR10'),
                              region_expert='lineaware')
    assert checked['region_expert'] == 'lineaware'
    with pytest.raises(RuntimeError, match='Region definition mismatch'):
        load_checkpoint(ck, *paths, ('SWIR9','SWIR10'), region_expert='refined')
    paths[2].write_bytes(b'changed geometry weights')
    with pytest.raises(RuntimeError, match='upstream model mismatch'):
        load_checkpoint(ck, *paths, ('SWIR9','SWIR10'), region_expert='lineaware')


def test_lineaware_mse_router_is_distinct_and_uses_pinned_checkpoints():
    from training.three_expert_router import ARCH_MSE, EXPERTS, FEATURE_NAMES, load_checkpoint
    from experiment_config import selected_results_folder
    import joblib
    from training.spectral_gate import sha
    from pathlib import Path
    import tempfile
    spec = ExperimentSpec(('SWIR6','SWIR7','SWIR8','SWIR9','SWIR10'),
                          ('SWIR1','SWIR2','SWIR3','SWIR4','SWIR5'), 'SWIR1', 'b_to_a', 'SWIR')
    assert selected_lineaware_region(spec, 'learned_router3_lineaware_mse')
    assert selected_results_folder(spec, 'learned_router3_lineaware_mse') == 'tpssr_learned_router3_lineaware_mse'
    with tempfile.TemporaryDirectory() as folder:
        base = Path(folder)
        paths = [base / f'weights{i}' for i in range(4)]
        for i, path in enumerate(paths):
            path.write_bytes(str(i).encode())
        ck = base / 'best.joblib'
        joblib.dump(dict(architecture=ARCH_MSE, risk_target='mse', region_expert='lineaware',
                         expert_names=EXPERTS, feature_names=FEATURE_NAMES,
                         source_cores=['SWIR9','SWIR10'],
                         gate_sha256=sha(paths[0]),fusion_sha256=sha(paths[1]),
                         region_geometry_sha256=sha(paths[2]),region_topology_sha256=sha(paths[3])),ck)
        loaded = load_checkpoint(ck, *paths, ('SWIR9','SWIR10'), objective='mse', region_expert='lineaware')
        assert loaded['risk_target'] == 'mse'


def test_squared_risk_refit_uses_source_cache_only(tmp_path, monkeypatch):
    import types
    import joblib
    import numpy as np
    import config
    from training import three_expert_router as module
    from training.spectral_gate import sha

    base = tmp_path / 'trained_models' / 'swir_region_b_to_a'
    tag = 'swir1'
    root = base / 'tpssr'
    paths = {
        'gate': root / 'spectral_gate/band/best_model.pt',
        'fusion': root / 'neighbor_fusion/best_model.pt',
        'geom': root / f'region_lineaware_trial_v1/geometry_{tag}/best_model.pt',
        'topo': root / f'region_lineaware_trial_v1/topology_{tag}/best_model.pt',
    }
    for key, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key.encode())
    old = module.path_for(base, region_expert='lineaware')
    old.parent.mkdir(parents=True)
    old_payload = dict(architecture=module.ARCH, risk_target='log_rmse',
                       expert_names=module.EXPERTS, feature_names=module.FEATURE_NAMES,
                       region_expert='lineaware', source_cores=['SWIR9','SWIR10'],
                       gate_sha256=sha(paths['gate']), fusion_sha256=sha(paths['fusion']),
                       region_geometry_sha256=sha(paths['geom']),
                       region_topology_sha256=sha(paths['topo']),
                       shadow_region_geometry_sha256='shadow_geom',
                       shadow_region_topology_sha256='shadow_topo',
                       shadow_training_cores=['SWIR6','SWIR7','SWIR8'],
                       shadow_experiment='shadow_name',
                       holdout_router_rmse=0.1, holdout_router_vs_reference_pct=5.)
    joblib.dump(old_payload, old)
    rng = np.random.default_rng(3)
    n = 400
    features = rng.uniform(0, 1, (n, len(module.FEATURE_NAMES))).astype(np.float32)
    err = rng.uniform(.0001, .03, (n, 3))
    present = np.ones((n, 3), bool)
    core = np.r_[np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)]
    np.savez_compressed(old.parent / 'calibration_labels.npz', x=features, err=err,
                        present=present, core=core,
                        scenario=np.resize(np.array(['isolated_voxels','whole_spectrum_pixels']),n),
                        severity=np.ones(n, np.int16),
                        gate_sha256=sha(paths['gate']), fusion_sha256=sha(paths['fusion']),
                        shadow_geometry_sha256='shadow_geom',
                        shadow_topology_sha256='shadow_topo',
                        final_geometry_sha256=sha(paths['geom']),
                        final_topology_sha256=sha(paths['topo']), region_expert='lineaware')
    spec = types.SimpleNamespace(model_tag=tag)
    monkeypatch.setattr(config, 'MODELS_DIR', base)
    monkeypatch.setattr(module, 'resolve_experiment', lambda _: spec)
    monkeypatch.setattr(module, 'spectral_validation_cores', lambda _: ('SWIR9','SWIR10'))
    module.train_from_cache(types.SimpleNamespace(region_expert='lineaware', objective='mse'))
    target = module.path_for(base, objective='mse', region_expert='lineaware')
    assert target.is_file()
    saved = module.load_checkpoint(target, *paths.values(), ('SWIR9','SWIR10'),
                                   objective='mse', region_expert='lineaware')
    assert saved['selection_policy'] == 'predicted_mean_squared_error'
    assert (target.parent / 'holdout_expert_by_scenario.csv').is_file()
    assert old.is_file()


def test_binary_router_only_sends_spectrally_unsupported_pixels_to_region():
    import numpy as np
    from training.three_expert_router import (ARCH_BINARY, BINARY_EXPERTS, FEATURE_NAMES,
                                              predict_binary_risks)
    class Predict:
        def __init__(self, c): self.c = c
        def predict(self, x): return np.full(len(x), np.log(self.c))
    payload=dict(architecture=ARCH_BINARY, expert_names=BINARY_EXPERTS,
                 feature_names=FEATURE_NAMES, models=[Predict(0.05), Predict(0.02)])
    x=np.zeros((4, len(FEATURE_NAMES)),np.float32)
    avail=np.array([[True,True,True],[True,False,True],
                    [False,True,True],[False,False,True]],bool)
    risks=predict_binary_risks(payload,x,avail)
    assert np.array_equal(np.argmin(risks,axis=1), [1,0,1,2])
    assert np.isinf(risks[:3,2]).all()
    assert risks[3,2] == 0.0


def test_binary_router_allows_region_oracle_wins_without_routing_there():
    import numpy as np
    from training.three_expert_router import (ARCH_BINARY, FEATURE_NAMES,
                         predict_binary_risks, fit_binary_models)
    rng=np.random.default_rng(7)
    n=600
    x=rng.normal(size=(n,len(FEATURE_NAMES))).astype(np.float32)
    err=np.zeros((n,3),np.float64)
    err[:,0]=0.02 + 0.01*(x[:,0] > 0)
    err[:,1]=0.02 + 0.01*(x[:,0] < 0)
    err[:,2]=0.001  # Region oracle is best, but binary policy excludes it.
    present=np.ones((n,3),bool)
    present[::17,:2]=False
    err[::17,:2]=np.nan
    core=np.array([0]*300+[1]*300)
    payload=fit_binary_models(x,err,present,core,('SWIR9','SWIR10'))
    assert payload['architecture']==ARCH_BINARY
    risks=predict_binary_risks(payload,x[core==1],present[core==1])
    selections=np.argmin(risks,axis=1)
    assert np.array_equal(selections==2,~present[core==1,:2].any(axis=1))
    assert np.isfinite(risks.min(axis=1)).all()


def test_binary_checkpoint_validates_frozen_models(tmp_path):
    import joblib
    import numpy as np
    import pytest
    from training.spectral_gate import sha
    from training.three_expert_router import (ARCH_BINARY, BINARY_EXPERTS,
                 FEATURE_NAMES, load_binary_checkpoint)
    files=[tmp_path/f'weight_{i}.bin' for i in range(4)]
    for i, f in enumerate(files): f.write_bytes(f'expert{i}'.encode())
    checkpoint=tmp_path/'binary.joblib'
    payload=dict(architecture=ARCH_BINARY, expert_names=BINARY_EXPERTS,
        feature_names=FEATURE_NAMES, source_cores=['SWIR9','SWIR10'],
        region_expert='lineaware', gate_sha256=sha(files[0]),
        fusion_sha256=sha(files[1]), region_geometry_sha256=sha(files[2]),
        region_topology_sha256=sha(files[3]))
    joblib.dump(payload, checkpoint)
    assert load_binary_checkpoint(checkpoint,*files,('SWIR9','SWIR10'))['architecture']==ARCH_BINARY
    with pytest.raises(RuntimeError,match='source split'):
        load_binary_checkpoint(checkpoint,*files,('SWIR4','SWIR9'))
    files[3].write_bytes(b'different-region-checkpoint')
    with pytest.raises(RuntimeError,match='frozen expert mismatch'):
        load_binary_checkpoint(checkpoint,*files,('SWIR9','SWIR10'))


def test_binary_inference_routes_from_observed_band_support(monkeypatch, tmp_path):
    import numpy as np
    from types import SimpleNamespace
    import config
    import src.pipeline as pipe
    import training.three_expert_router as reroute
    import training.neighbor_fusion as fusion_module

    class ConstantRisk:
        def __init__(self, risk): self.risk=risk
        def predict(self, x): return np.full(len(x), np.log(self.risk))

    wave_dir=tmp_path/'SWIR1';wave_dir.mkdir()
    np.save(wave_dir/'wavelengths_nm.npy',np.array([1000,1100,1200],np.float32))
    monkeypatch.setattr(config,'PROCESSED_DATA_ROOT',tmp_path)
    monkeypatch.setattr(config,'TEST_DATASET_FOLDER','SWIR1')
    monkeypatch.setattr(fusion_module,'crop_at',lambda c,m,y,x,radius: (c,m,y,x))
    def features(c,m,cy,cx,*_rest):
        nvalid=np.sum(~m[cy,cx] & np.isfinite(c[cy,cx]))
        return (np.zeros(len(reroute.FEATURE_NAMES),np.float32),
                np.full(3,0.4,np.float32) if nvalid>=2 else None,
                np.full(3,0.6,np.float32) if nvalid>=2 else None)
    monkeypatch.setattr(reroute,'features_for_pixel',features)
    monkeypatch.setattr(reroute,'refine_region',lambda _c,_m,_coords,preds:(preds,0))
    def region(_t,_tc,_g,_cube,_mask,_device,eval_coords,**_kwargs):
        return ({int(b):(np.array([[int(y),int(x)] for y,x,_b in eval_coords if int(_b)==int(b)]),
                           np.full(sum(int(_b)==int(b) for _y,_x,_b in eval_coords),0.8,np.float32))
                 for b in np.unique(eval_coords[:,2])},None)
    monkeypatch.setattr(pipe,'reconstruct_final_region',region)
    payload=dict(architecture=reroute.ARCH_BINARY,expert_names=reroute.BINARY_EXPERTS,
                 feature_names=reroute.FEATURE_NAMES,models=[ConstantRisk(0.1),ConstantRisk(0.02)],
                 region_expert='lineaware')
    models=SimpleNamespace(device='cpu',fusion_model=object(),router3_checkpoint=payload,
        hybrid_mean=np.ones(3),hybrid_std=np.ones(3),hybrid_model=object(),
        topology_model=object(),topology_checkpoint={},geometry_model=object(),
        region_batch=16,verify_region_batch=False)
    mask=np.array([[[True,True,True],[True,False,False]]],bool)
    cube=np.array([[[np.nan,np.nan,np.nan],[np.nan,0.3,0.4]]],np.float32)
    coords=np.array([[0,0,0],[0,0,1],[0,0,2],[0,1,0]],np.int64)
    output=pipe.predict_mask_learned_router3(models,cube,mask,coords)
    assert np.allclose(output['prediction'],[0.8,0.8,0.8,0.6])
    assert output['candidate_policy_audit']['router3_region_voxels']==3
    assert output['candidate_policy_audit']['router3_fusion_voxels']==1


def test_rankcv_three_expert_crossfit_selects_only_eligible_experts():
    import numpy as np
    from training.three_expert_router import (
        ARCH_RANK_CV, FEATURE_NAMES, _crossfit_rank, _fit_rank_family,
        _route_score, predict_risks, path_for,
    )
    rng = np.random.default_rng(733)
    n = 340
    x = rng.normal(size=(n, len(FEATURE_NAMES))).astype(np.float32)
    missing_fraction = np.linspace(.025, 1.0, n, dtype=np.float32)
    x[:, FEATURE_NAMES.index('missing_fraction')] = missing_fraction
    core = np.arange(n) % 2
    errors = np.column_stack([
        .023 + .060 * missing_fraction,
        .005 + .052 * missing_fraction**3,
        .065 - .050 * missing_fraction,
    ])
    errors += rng.normal(0, .001, size=errors.shape)
    errors = np.maximum(errors, 1e-5)
    present = np.ones((n,3), bool)
    present[missing_fraction >= .996, :2] = False
    errors[~present] = np.nan
    winner, ledger = _crossfit_rank(x, errors, present, core,
        families=('log_risk_regularized', 'pairwise_log_ratio'))
    assert winner in ('log_risk_regularized', 'pairwise_log_ratio')
    assert len(ledger) == 2
    assert all(np.isfinite(v['pooled_oof_rmse']) for v in ledger)
    payload = _fit_rank_family(winner, x, errors, present)
    assert payload['architecture'] == ARCH_RANK_CV
    risk = predict_risks(payload, x, present)
    assert risk.shape == (n, 3)
    assert (np.isinf(risk[~present])).all()
    assert np.isfinite(risk[present]).all()
    selection = _route_score(payload, x, errors, present)
    assert np.all(present[np.arange(n), selection['choice']])
    assert np.isfinite(selection['rmse'])
    assert path_for('model_root', 'rank_cv', 'lineaware').parent.name == 'learned_router3_lineaware_rankcv'


def test_rankcv_checkpoint_schema_and_pins(tmp_path):
    import joblib
    import pytest
    from training.three_expert_router import ARCH_RANK_CV, FEATURE_NAMES, EXPERTS, load_checkpoint
    from training.spectral_gate import sha
    weights = [tmp_path / f'w{i}.pt' for i in range(4)]
    for i, path in enumerate(weights):
        path.write_bytes(f'model_{i}'.encode())
    ck = tmp_path / 'rank_cv.joblib'
    payload = dict(architecture=ARCH_RANK_CV, risk_target='log_rmse',
                   feature_names=FEATURE_NAMES, expert_names=EXPERTS,
                   region_expert='lineaware', source_cores=['SWIR9','SWIR10'],
                   gate_sha256=sha(weights[0]), fusion_sha256=sha(weights[1]),
                   region_geometry_sha256=sha(weights[2]),
                   region_topology_sha256=sha(weights[3]))
    joblib.dump(payload, ck)
    assert load_checkpoint(ck, *weights, ('SWIR9','SWIR10'),
                           objective='rank_cv',region_expert='lineaware')['architecture'] == ARCH_RANK_CV
    with pytest.raises(RuntimeError,match='schema mismatch'):
        load_checkpoint(ck,*weights,('SWIR9','SWIR10'),
                        objective='log_rmse',region_expert='lineaware')
    weights[3].write_bytes(b'tampered')
    with pytest.raises(RuntimeError,match='upstream model mismatch'):
        load_checkpoint(ck,*weights,('SWIR9','SWIR10'),
                        objective='rank_cv',region_expert='lineaware')


def test_rankcv_is_a_distinct_swir_only_results_mode():
    from experiment_config import selected_results_folder, selected_lineaware_region
    spec=_spec('swir_8x2')
    assert selected_lineaware_region(spec, 'learned_router3_lineaware_rankcv')
    assert selected_results_folder(spec, 'learned_router3_lineaware_rankcv') == 'tpssr_learned_router3_lineaware_rankcv'
    assert selected_lineaware_region(_spec('vnir_8x2'), 'learned_router3_lineaware_rankcv')
