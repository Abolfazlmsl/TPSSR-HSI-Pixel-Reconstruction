"""Source-only three-expert router: spectral interpolation, neighbor fusion, Region.

Region calibration uses a separate geometry/topology model trained on source
backbone-training cores, excluding the two router-calibration acquisitions.
The final Region checkpoint is used only at destination inference.
"""
from __future__ import annotations

import argparse
import json
import os
import math
from pathlib import Path
import sys

CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import joblib
import numpy as np
import torch
import torch.distributed as dist
from sklearn.ensemble import HistGradientBoostingRegressor

import config
from experiment_config import add_experiment_args, resolve_experiment, spectral_validation_cores
from training.spectral_data import load_core_data
from training.spectral_gate import sha
from training.neighbor_fusion import crop_at, make_features, checkpoint_path, load_fusion_checkpoint
from src.training_corruptions import SCENARIOS, generate_training_mask
from src.spatial_inference import load_models, reconstruct_mask
from src.routing_policy import local_neighbor_median

ARCH = 'three_expert_quality_router_v1'
ARCH_MSE = 'three_expert_mse_router_v2'
ARCH_RANK_CV = 'three_expert_source_core_crossfit_ranking_v1'
ARCH_REGRET_CV = 'three_expert_source_core_cost_sensitive_cv_v1'
EXPERTS = ('interp', 'fusion', 'region')
FEATURE_NAMES = (
    'observed_fraction', 'missing_fraction', 'longest_gap_fraction',
    'nearest_anchor_mean', 'masked_band_centroid', 'missing_runs',
    'mean_band_loss', 'max_band_loss', 'local_loss_3', 'local_loss_15',
    'donor_fraction', 'donor_discrepancy', 'donor_distance',
    'support_distance', 'support_fraction', 'observed_variation',
    'interp_variation', 'fusion_disagreement', 'fusion_max_disagreement',
    'fusion_correction_variation',
)


def path_for(base, objective="log_rmse", region_expert="raw"):
    if region_expert not in ("raw", "refined", "lineaware"):
        raise ValueError("Unknown Region expert variant")
    if objective in ('rank_cv', 'regret_cv') and region_expert != 'lineaware':
        raise ValueError('RankCV is supported only for Line-aware Region')
    if region_expert == 'lineaware' and objective not in ('log_rmse', 'mse', 'rank_cv', 'regret_cv'):
        raise ValueError('Unsupported Line-aware Router objective')
    folder = ("learned_router3_lineaware_regretcv" if region_expert == 'lineaware' and objective == 'regret_cv'
              else "learned_router3_lineaware_rankcv" if region_expert == "lineaware" and objective == "rank_cv"
              else "learned_router3_lineaware_mse" if region_expert == 'lineaware' and objective == 'mse'
              else "learned_router3_lineaware" if region_expert == "lineaware"
              else "learned_router3_aligned" if region_expert == "refined"
              else "learned_router3_rmse" if objective == "mse" else "learned_router3")
    if objective not in ("log_rmse", "mse", "rank_cv", "regret_cv"):
        raise ValueError("Unknown Router3 objective")
    return Path(base) / "tpssr" / folder / "best_model.joblib"


def shadow_experiment(spec, val_cores):
    if (len(val_cores) != 2 or len(set(val_cores)) != 2 or
            not set(val_cores).issubset(spec.train)):
        raise ValueError('Three-expert calibration requires two distinct source-only calibration cores')
    train = tuple(n for n in spec.train if n not in val_cores)
    if len(train) not in (3, 6):
        raise ValueError('Shadow Region expects three or six fitting cores')
    if set(train) & set(spec.tests):
        raise ValueError('Held-out test cores cannot be used to fit Shadow Region')
    return spec.experiment_name + '__router3_region_oof', train


def shadow_geometry_path(spec, val_cores):
    name, _ = shadow_experiment(spec, val_cores)
    return Path(config.PROJECT_DIR) / 'trained_models' / name / 'tpssr' / f'geometry_{val_cores[0].lower()}' / 'best_model.pt'


def shadow_topology_path(spec, val_cores):
    name, _ = shadow_experiment(spec, val_cores)
    return Path(config.PROJECT_DIR) / 'trained_models' / name / 'tpssr' / f'topology_{val_cores[0].lower()}' / 'best_model.pt'


def load_shadow_region(spec, val_cores, device):
    name, train = shadow_experiment(spec, val_cores)
    # load_models is checkpoint-path-aware via config; do not change the main models.
    old_dir, old_tag = config.MODELS_DIR, config.TEST_DATASET_FOLDER
    try:
        config.MODELS_DIR = Path(config.PROJECT_DIR) / 'trained_models' / name
        config.TEST_DATASET_FOLDER = val_cores[0]
        loaded = load_models(device)
    finally:
        config.MODELS_DIR, config.TEST_DATASET_FOLDER = old_dir, old_tag
    topology, tck, geometry, gck, tpath, gpath = loaded
    _verify_shadow_geometry(spec, val_cores, train, gck, tck, gpath)
    return topology, tck, geometry, Path(tpath), Path(gpath)


def _verify_shadow_geometry(spec, val_cores, train, geometry_ck, topology_ck, geometry_path,
                            region_expert='raw'):
    """Verify the source split from Geometry's training-only metadata.

    Geometry checkpoints store the normalization metadata path and arrays,
    but not a training_datasets field. The train split is recorded in the
    separate geometry_train_stats_*.npz generated by train.py.
    """
    shadow_name, expected_train = shadow_experiment(spec, val_cores)
    if tuple(train) != expected_train:
        raise RuntimeError('Unexpected shadow Region training split')
    root = Path(config.PROJECT_DIR) / 'trained_models' / shadow_name / 'tpssr'
    expected_geom_root = root / 'region_lineaware_trial_v1' if region_expert == 'lineaware' else root
    expected_geom = expected_geom_root / f'geometry_{val_cores[0].lower()}' / 'best_model.pt'
    expected_stats = root / f'geometry_train_stats_{val_cores[0].lower()}.npz'
    if Path(geometry_path).resolve() != expected_geom.resolve():
        raise RuntimeError('Shadow Geometry checkpoint path mismatch')
    if Path(str(geometry_ck.get('normalization_metadata_source', ''))).resolve() != expected_stats.resolve():
        raise RuntimeError('Shadow Geometry normalization provenance mismatch')
    if not expected_stats.is_file():
        raise RuntimeError(f'Shadow Geometry training statistics not found: {expected_stats}')
    if Path(str(topology_ck.get('source_geometry_checkpoint', ''))).resolve() != expected_geom.resolve():
        raise RuntimeError('Shadow Topology was not trained from this Geometry checkpoint')
    if 'training_datasets' in geometry_ck and tuple(geometry_ck['training_datasets']) != expected_train:
        raise RuntimeError('Shadow Geometry checkpoint training split mismatch')
    try:
        with np.load(expected_stats, allow_pickle=False) as stats:
            required = ('train_datasets', 'n_bands', 'band_mean', 'band_std', 'wavelengths_nm')
            if any(field not in stats for field in required):
                raise RuntimeError('Incomplete Shadow Geometry training statistics')
            observed_train = tuple(str(name) for name in stats['train_datasets'].tolist())
            if observed_train != expected_train:
                raise RuntimeError(f'Shadow Geometry training split mismatch: {observed_train}')
            if int(stats['n_bands']) != int(geometry_ck['n_bands']):
                raise RuntimeError('Shadow Geometry band count mismatch')
            for key in ('band_mean', 'band_std', 'wavelengths_nm'):
                a = np.asarray(stats[key], dtype=np.float32)
                b = np.asarray(geometry_ck[key], dtype=np.float32)
                if a.shape != b.shape or not np.array_equal(a, b):
                    raise RuntimeError(f'Shadow Geometry training statistics mismatch: {key}')
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(f'Unable to verify shadow training statistics: {expected_stats}') from exc


def shadow_lineaware_paths(spec, val_cores):
    name, _ = shadow_experiment(spec, val_cores)
    root = (Path(config.PROJECT_DIR) / 'trained_models' / name / 'tpssr' /
            'region_lineaware_trial_v1')
    tag = val_cores[0].lower()
    return (root / f'geometry_{tag}' / 'best_model.pt',
            root / f'topology_{tag}' / 'best_model.pt')


def load_shadow_lineaware_region(spec, val_cores, wave, device):
    from src.lineaware_region import load_region_from_paths
    name, train = shadow_experiment(spec, val_cores)
    geom_path, topo_path = shadow_lineaware_paths(spec, val_cores)
    loaded = load_region_from_paths(device, wave, geom_path, topo_path)
    topology, tck, geometry, gck, tpath, gpath = loaded
    _verify_shadow_geometry(spec, val_cores, train, gck, tck, gpath,
                            region_expert='lineaware')
    return topology, tck, geometry, Path(tpath), Path(gpath)


def _longest(m):
    edges = np.diff(np.pad(np.asarray(m, np.int8), (1, 1)))
    return int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max(initial=0))


def features_for_pixel(cube, mask, cy, cx, mean, std, wave, fusion_model=None,
                       hybrid_model=None, device=None, compute_fusion=True):
    """Generate one inference-identical feature row from erased observations.

    cube is a 65x65 or larger observed HSI crop; mask includes all corruption.
    The center truth must never be accessed in this function.
    """
    b = int(cube.shape[-1]); yy, xx = int(cy), int(cx)
    missing = np.asarray(mask[yy, xx], bool) | ~np.isfinite(cube[yy, xx])
    obs = np.flatnonzero(~missing)
    central = np.asarray(cube[yy, xx], np.float32)
    n = int(missing.sum())
    if not n:
        raise ValueError('Expected a corrupted target spectrum')
    anchor = np.zeros(b, np.float32)
    fusion = None
    sample = None
    if obs.size >= 2:
        anchor = np.interp(np.arange(b), obs, central[obs]).astype(np.float32)
        if compute_fusion:
            patch, bad, py, px = crop_at(cube, mask, yy, xx)
            sample = make_features(patch, bad, py, px, mean, std, wave)
            if sample is None:
                raise RuntimeError('Expected a feasible learned fusion feature tuple')
            if fusion_model is None or hybrid_model is None or device is None:
                raise ValueError('Frozen spectral models required for feature generation')
            with torch.inference_mode():
                tensors = [torch.as_tensor(z[None], device=device) for z in sample]
                hp = hybrid_model(tensors[1])
                fp = fusion_model(tensors[0], hp, tensors[2], tensors[3], tensors[4])
                fusion = (fp[0].float().cpu().numpy() * std + mean).astype(np.float32)
    margin = min(7, yy, xx, cube.shape[0] - yy - 1, cube.shape[1] - xx - 1)
    short = np.asarray(mask[yy-1:yy+2, xx-1:xx+2], bool) if margin >= 1 else mask[yy:yy+1, xx:xx+1]
    wide = mask[max(0,yy-7):min(mask.shape[0], yy+8), max(0,xx-7):min(mask.shape[1], xx+8)]
    band_fraction = mask.mean(axis=(0,1))
    if obs.size:
        near = np.abs(np.arange(b)[:,None] - obs[None,:]).min(axis=1)
    else:
        near = np.full(b, b, dtype=np.int64)
    runs = np.sum(np.diff(np.pad(missing.astype(np.int8),(1,1))) == 1)
    if sample is None:
        donors = 0.0; similarity = 1.0; donor_dist = 1.0
        support_dist = 1.0; support_frac = 0.0
    else:
        valid = sample[4] > .5
        meta = sample[3][valid]
        donors = float(valid.mean())
        similarity = float(meta[:,0].mean()) if len(meta) else 1.0
        donor_dist = float(meta[:,1].mean()) if len(meta) else 1.0
        support_dist = float(sample[0][0,9])
        support_frac = float(sample[0][0,10])
    difference = fusion[missing]-anchor[missing] if fusion is not None else np.zeros(n, np.float32)
    row = np.asarray([
        obs.size/b, n/b, _longest(missing)/b,
        float(near[missing].mean()/b), float(np.flatnonzero(missing).mean()/max(b-1,1)),
        float(runs/max(b,1)),
        float(band_fraction[missing].mean()), float(band_fraction[missing].max()),
        float(short[:,:,missing].mean()),float(wide[:,:,missing].mean()),
        donors, similarity, donor_dist, support_dist, support_frac,
        float(central[obs].std()) if len(obs) else 0.,
        float(anchor[missing].std()) if len(obs) >= 2 else 0.,
        float(np.abs(difference).mean()),float(np.abs(difference).max()),
        float(difference.std()),
    ], dtype=np.float32)
    if row.shape != (len(FEATURE_NAMES),) or not np.isfinite(row).all():
        raise RuntimeError('Invalid 3-expert router features')
    return row, anchor if obs.size >= 2 else None, fusion


def _rank_pairwise_log_scores(payload, x):
    # Pairwise log-RMSE differences directly model which expert is better.
    # Recenter them around the independently estimated log-RMSE mean so
    # inference and existing diagnostics stay on the expected RMSE scale.
    a = np.column_stack([model.predict(x) for model in payload['models']])
    d01, d02, d12 = [np.clip(model.predict(x), -6., 6.)
                     for model in payload['pair_models']]
    relative = np.column_stack(((d01 + d02) / 3.,
                                (-d01 + d12) / 3.,
                                (-d02 - d12) / 3.))
    return np.exp(np.clip(a.mean(axis=1, keepdims=True) + relative, -18., 5.))


def predict_risks(payload, x, availability=None):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError('Router feature mismatch')
    if payload.get('architecture') == ARCH_REGRET_CV and payload.get('model_family') == 'cost_pairwise':
        probabilities = []
        for model in payload['pair_models']:
            proba = model.predict_proba(x)
            classes = np.asarray(model.classes_, int)
            p0 = proba[:, np.where(classes == 1)[0][0]] if 1 in classes else np.zeros(len(x))
            probabilities.append(p0)
        p01, p02, p12 = probabilities  # Probability first-listed expert has less squared loss.
        risks = np.column_stack((2.0-p01-p02, p01+1.0-p12, p02+p12))
    elif payload.get('architecture') == ARCH_REGRET_CV and payload.get('model_family') == 'cost_oracle':
        model = payload['oracle_model']
        proba = model.predict_proba(x)
        p = np.zeros((len(x), 3), dtype=np.float64)
        p[:, np.asarray(model.classes_, int)] = proba
        risks = 1.0-p
    elif payload.get('architecture') == ARCH_RANK_CV and payload.get('model_family') == 'pairwise_log_ratio':
        risks = _rank_pairwise_log_scores(payload, x)
    else:
        risks = np.column_stack([m.predict(x) for m in payload['models']])
    # Return a common RMSE scale for inference and audit. Ranking is by
    # predicted mean squared error for the new objective.
    if payload.get('architecture') == ARCH_REGRET_CV and payload.get('risk_target') == 'cost_sensitive_ranking':
        pass  # Ranking scores (not calibrated RMSE predictions).
    elif payload.get('architecture') == ARCH_RANK_CV and payload.get('model_family') == 'pairwise_log_ratio':
        pass  # already converted to positive RMSE values
    elif payload.get('risk_target', 'log_rmse') == 'mse':
        risks = np.sqrt(np.maximum(risks, 0.0))
    else:
        risks = np.exp(np.clip(risks, -18, 5))
    if availability is not None:
        allow = np.asarray(availability, bool)
        if allow.shape != risks.shape:
            raise ValueError('Router expert-availability shape mismatch')
        risks[~allow] = np.inf
    if not np.isfinite(np.min(risks, axis=1)).all():
        raise RuntimeError('No eligible reconstruction expert')
    return risks


def fit_models(x, err, present, core, source_cores, objective="log_rmse", case_meta=None):
    """Fit three independent conditional risk regressors, holding out one core."""
    x=np.asarray(x,np.float32); err=np.asarray(err,np.float64)
    present=np.asarray(present,bool); core=np.asarray(core,np.int64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or err.shape != (len(x),3):
        raise ValueError('Invalid three-expert calibration matrix')
    if set(np.unique(core)) != {0,1}:
        raise RuntimeError('Both distinct calibration cores must be present')
    if objective not in ('log_rmse', 'mse'):
        raise ValueError('Unsupported risk objective')
    if present.shape != err.shape or core.shape != (len(x),):
        raise ValueError('Invalid risk availability / core vectors')
    models=[]
    for k in range(3):
        fit = (core == 0) & present[:,k] & np.isfinite(err[:,k])
        if fit.sum() < 50:
            raise RuntimeError(f'Insufficient calibration data for expert {EXPERTS[k]}: {fit.sum()}')
        estimator = HistGradientBoostingRegressor(
            max_iter=130, learning_rate=.045, max_leaf_nodes=12,
            min_samples_leaf=12, l2_regularization=1.5, random_state=8207+k)
        target = np.square(err[fit,k]) if objective == 'mse' else np.log(np.maximum(err[fit,k], 1e-6))
        estimator.fit(x[fit], target)
        models.append(estimator)
    payload={'architecture':ARCH_MSE if objective == 'mse' else ARCH,
             'risk_target':objective,'feature_names':FEATURE_NAMES,'expert_names':EXPERTS,
             'models':models,'fit_cores':[source_cores[0]],'holdout_cores':[source_cores[1]],
             'source_cores':list(source_cores), 'fit_cases':int((core==0).sum()),
             'holdout_cases':int((core==1).sum())}
    hold = core == 1
    risk = predict_risks(payload, x[hold], present[hold])
    selected = np.argmin(risk,axis=1)
    valid = np.asarray(err[hold], float)
    row = np.arange(len(selected))
    errors_chosen = valid[row,selected]
    if not np.isfinite(errors_chosen).all():
        raise RuntimeError('Nonfinite error for a selected expert')
    payload['holdout_router_rmse'] = float(np.sqrt(np.mean(errors_chosen**2)))
    payload['holdout_expert_rmse'] = {k:float(np.sqrt(np.nanmean(np.where(present[hold,j],valid[:,j]**2,np.nan))))
                                      for j,k in enumerate(EXPERTS)}
    payload['holdout_selected'] = {k:int((selected == j).sum()) for j,k in enumerate(EXPERTS)}
    # A full-coverage diagnostic reference: Fusion where possible, otherwise
    # Interpolation or Region. This is for evaluation ONLY, not an inference route.
    reference = np.where(present[hold, 1], 1, np.where(present[hold, 0], 0, 2))
    if not np.all(present[hold][row, reference]):
        raise RuntimeError('Missing reference predictions on calibration holdout')
    reference_rmse = float(np.sqrt(np.mean(valid[row, reference] ** 2)))
    payload['holdout_full_coverage_reference_rmse'] = reference_rmse
    payload['holdout_router_vs_reference_pct'] = float(
        100 * (payload['holdout_router_rmse'] / max(reference_rmse, 1e-12) - 1))
    payload['holdout_region_required'] = int(np.sum(~present[hold, :2].any(axis=1)))
    payload['holdout_region_optional_selected'] = int(np.sum(
        (selected == 2) & present[hold, :2].any(axis=1)))
    # Audit only; never use holdout ground truth to change the routing decision.
    baseline_eligible = present[hold, 1]
    if baseline_eligible.all():
        baseline = valid[:, 1]
        payload['holdout_fusion_rmse'] = float(np.sqrt(np.mean(baseline**2)))
        payload['holdout_router_vs_fusion_pct'] = float(100*(payload['holdout_router_rmse']/payload['holdout_fusion_rmse']-1))
        payload['holdout_router_wins_vs_fusion'] = int(np.sum(errors_chosen < baseline))
        payload['holdout_router_losses_vs_fusion'] = int(np.sum(errors_chosen > baseline))
        payload['holdout_router_ties_vs_fusion'] = int(np.sum(errors_chosen == baseline))
    # Oracle is a descriptive lower bound, never an inference-time expert.
    oracle = np.min(np.where(present[hold],valid,np.inf),axis=1)
    payload['holdout_oracle_rmse'] = float(np.sqrt(np.mean(oracle**2)))
    details=[]
    indices = np.flatnonzero(hold)
    for h,idx in enumerate(indices):
        info=case_meta[idx] if case_meta is not None else ('','')
        details.append(dict(scenario=info[0],severity=int(info[1]) if info[1] != '' else '',
            selected_expert=EXPERTS[selected[h]],
            actual_selected_rmse=float(errors_chosen[h]),
            predicted_interp_rmse=float(risk[h,0]),
            predicted_fusion_rmse=float(risk[h,1]),
            predicted_region_rmse=float(risk[h,2]),
            actual_interp_rmse=float(valid[h,0]) if present[hold][h,0] else None,
            actual_fusion_rmse=float(valid[h,1]) if present[hold][h,1] else None,
            actual_region_rmse=float(valid[h,2]) if present[hold][h,2] else None,
            oracle_expert=EXPERTS[int(np.argmin(np.where(present[hold][h],valid[h],np.inf)))],
            oracle_rmse=float(oracle[h])))
    payload['_holdout_details'] = details
    payload['validation_note'] = ('Shadow Region was trained only on the source training cores excluding '
                                  'the two router calibration acquisitions. Frozen spectral checkpoints were '
                                  'selected using the two source calibration cores, so the spectral experts '
                                  'are not fully out-of-fold. Destination test cores were never used to fit '
                                  'the router.')
    return payload


def load_checkpoint(path, gate_path, fusion_path, region_geometry_path, region_topology_path, expected_cores, objective="log_rmse", region_expert="raw"):
    payload=joblib.load(path)
    expected_arch = (ARCH_REGRET_CV if objective == 'regret_cv' else
                     ARCH_RANK_CV if objective == 'rank_cv' else
                     ARCH_MSE if objective == 'mse' else ARCH)
    if (payload.get('architecture') != expected_arch or
        tuple(payload.get('feature_names',())) != FEATURE_NAMES or
        tuple(payload.get('expert_names',())) != EXPERTS):
        raise RuntimeError('Three-expert router schema mismatch')
    for name, p in (('gate_sha256', gate_path),('fusion_sha256',fusion_path),
                    ('region_geometry_sha256',region_geometry_path),
                    ('region_topology_sha256',region_topology_path)):
        if payload.get(name) != sha(p):
            raise RuntimeError('Router checkpoint upstream model mismatch: ' + name)
    expected_target = ('cost_sensitive_ranking' if (objective == 'regret_cv' and
                       payload.get('model_family') in ('cost_pairwise', 'cost_oracle'))
                       else 'log_rmse' if objective in ('rank_cv', 'regret_cv') else objective)
    if payload.get('risk_target', 'log_rmse') != expected_target:
        raise RuntimeError('Three-expert router risk objective mismatch')
    if payload.get('region_expert', 'raw') != region_expert:
        raise RuntimeError('Three-expert Region definition mismatch')
    if tuple(payload.get('source_cores',())) != tuple(expected_cores):
        raise RuntimeError('Three-expert router provenance mismatch')
    return payload


def refine_region(cube, mask, coords, raw_predictions):
    """Apply the observed-neighbor refinement to the Region expert itself.

    This changes Region's prediction, never the learned branch selection.
    The implementation shared by calibration and deployment reads only
    measured neighbors and does not access masked target values.
    """
    coords = np.asarray(coords, np.int64)
    result = np.asarray(raw_predictions, np.float32).copy()
    if coords.shape != (len(result), 3):
        raise ValueError('Region prediction/coordinate shape mismatch')
    if len(coords) == 0:
        return result, 0
    median, _ = local_neighbor_median(cube, mask, coords,
                                       np.full(len(coords), 2, dtype=np.int32))
    use = np.isfinite(median)
    result[use] = median[use]
    return result, int(use.sum())


def _lookup_region(region_model, cube, mask, coords, device, batch):
    topology, ck, geometry = region_model
    output,_=reconstruct_mask(topology,ck,geometry,cube,mask,device,
                              eval_coords=coords,batch_size=batch,verify_batch=False)
    table = {(int(y),int(x),int(b)):float(v) for b,(yx,vs) in output.items()
             for (y,x),v in zip(yx,vs)}
    return np.asarray([table.get(tuple(map(int,c)),np.nan) for c in coords],np.float32)


def make_training_case(cube, invalid, rng, scenario, severity, mean, std, wave,
                       hybrid, fusion, region, device, region_batch, region_expert="raw"):
    h,w,b=cube.shape
    for _ in range(75):
        y=int(rng.integers(32,h-32)) if h>64 else int(rng.integers(h))
        x=int(rng.integers(32,w-32)) if w>64 else int(rng.integers(w))
        y0=max(0,y-32); y1=min(h,y+33); x0=max(0,x-32); x1=min(w,x+33)
        roi=np.asarray(cube[y0:y1,x0:x1],np.float32).copy()
        orig_bad=np.asarray(invalid[y0:y1,x0:x1],bool) | ~np.isfinite(roi)
        cy,cx=y-y0,x-x0
        if int((~orig_bad[cy,cx]).sum()) < max(35,int(.90*b)):
            continue
        synthetic,_,_=generate_training_mask(roi.shape,rng,scenario=scenario,
                        severity=severity,center=(cy,cx),band_center=int(rng.integers(b)))
        target=synthetic[cy,cx] & ~orig_bad[cy,cx]
        if not target.any():
            continue
        all_bad=orig_bad | synthetic
        erased=roi.copy();erased[all_bad]=np.nan
        with torch.inference_mode():
            feat,interp,fp=features_for_pixel(erased,all_bad,cy,cx,mean,std,wave,
                                                 fusion,hybrid,device)
        bands=np.flatnonzero(target)
        coords=np.column_stack((np.full(len(bands),cy),np.full(len(bands),cx),bands))
        rp=_lookup_region(region,erased,all_bad,coords,device,region_batch)
        if region_expert in ('refined', 'lineaware'):
            rp, _ = refine_region(erased, all_bad, coords, rp)
        if not np.isfinite(rp).all():
            continue
        truth=roi[cy,cx,bands]
        risks=np.asarray([np.sqrt(np.mean((interp[bands]-truth)**2)) if interp is not None else np.nan,
                          np.sqrt(np.mean((fp[bands]-truth)**2)) if fp is not None else np.nan,
                          np.sqrt(np.mean((rp-truth)**2))],np.float64)
        return feat,risks,np.isfinite(risks)
    raise RuntimeError(f'Could not sample {scenario} S{severity}')


def train_from_cache(args):
    """Refit a source-only Line-aware Router using cached counterfactual losses.

    The old log-RMSE model and source-only labels remain immutable. This fit
    targets expected squared reconstruction error instead of log-RMSE.
    """
    import csv
    if args.region_expert != 'lineaware' or args.objective != 'mse':
        raise ValueError('Cached calibration currently supports only Line-aware squared risk')
    spec = resolve_experiment(args)
    val = tuple(spectral_validation_cores(spec))
    old = path_for(config.MODELS_DIR, 'log_rmse', 'lineaware')
    target = path_for(config.MODELS_DIR, 'mse', 'lineaware')
    if target.exists():
        raise FileExistsError('Refusing to overwrite the Line-aware squared-risk checkpoint: ' + str(target))
    if not old.is_file():
        raise FileNotFoundError('Train router3_lineaware before using the cached squared-risk refit: ' + str(old))
    base = Path(config.MODELS_DIR) / 'tpssr'
    tag = spec.model_tag
    gate = base / 'spectral_gate' / 'band' / 'best_model.pt'
    fusion = base / 'neighbor_fusion' / 'best_model.pt'
    final_root = base / 'region_lineaware_trial_v1'
    final_geom = final_root / f'geometry_{tag}' / 'best_model.pt'
    final_topo = final_root / f'topology_{tag}' / 'best_model.pt'
    previous = load_checkpoint(old, gate, fusion, final_geom, final_topo, val,
                               objective='log_rmse', region_expert='lineaware')
    cache = old.parent / 'calibration_labels.npz'
    if not cache.is_file():
        raise FileNotFoundError('Source-only cached labels not found: ' + str(cache))
    from_cache = {
        'gate_sha256': sha(gate),
        'fusion_sha256': sha(fusion),
        'shadow_geometry_sha256': previous['shadow_region_geometry_sha256'],
        'shadow_topology_sha256': previous['shadow_region_topology_sha256'],
        'final_geometry_sha256': sha(final_geom),
        'final_topology_sha256': sha(final_topo),
        'region_expert': 'lineaware',
    }
    with np.load(cache, allow_pickle=False) as stored:
        for key, expected in from_cache.items():
            if key not in stored or str(stored[key].item()) != expected:
                raise RuntimeError(f'Cached Router3 labels do not match the frozen model: {key}')
        features = np.asarray(stored['x'], np.float32)
        err = np.asarray(stored['err'], np.float64)
        present = np.asarray(stored['present'], bool)
        core = np.asarray(stored['core'], np.int64)
        scenarios = np.asarray(stored['scenario']).astype(str)
        severities = np.asarray(stored['severity'], np.int64)
    n = len(features)
    if not (n >= 200 and err.shape == (n, 3) and present.shape == (n, 3)
            and core.shape == (n,) and scenarios.shape == (n,)
            and severities.shape == (n,) and np.isfinite(features).all()
            and set(np.unique(core)) == {0, 1} and np.isfinite(err[present]).all()
            and (err[present] >= 0).all()
            and not np.any(np.isfinite(err[~present]))):
        raise RuntimeError('Invalid cached three-expert calibration labels')
    fitted = fit_models(features, err, present, core, val, objective='mse',
                        case_meta=list(zip(scenarios, severities)))
    for field in ('gate_sha256', 'fusion_sha256', 'region_geometry_sha256',
                  'region_topology_sha256', 'shadow_region_geometry_sha256',
                  'shadow_region_topology_sha256', 'shadow_training_cores',
                  'shadow_experiment'):
        fitted[field] = previous[field]
    fitted['region_expert'] = 'lineaware'
    fitted['calibration_source'] = str(cache)
    fitted['calibration_source_sha256'] = sha(cache)
    fitted['selection_policy'] = 'predicted_mean_squared_error'
    fitted['comparison_log_rmse_router_holdout'] = previous['holdout_router_rmse']
    fitted['comparison_log_rmse_router_vs_reference_pct'] = previous.get('holdout_router_vs_reference_pct')
    target.parent.mkdir(parents=True, exist_ok=True)
    audit = fitted.pop('_holdout_details')
    with (target.parent / 'holdout_case_audit.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(audit[0])); writer.writeheader(); writer.writerows(audit)
    by_scenario = []
    for scenario in sorted(set(scenarios[core == 1])):
        rows = [row for row in audit if row['scenario'] == scenario]
        records = {'scenario': scenario, 'cases': len(rows),
                   'selected_region': sum(row['selected_expert'] == 'region' for row in rows),
                   'oracle_region': sum(row['oracle_expert'] == 'region' for row in rows)}
        paired = [row for row in rows if row['actual_region_rmse'] is not None
                  and (row['actual_interp_rmse'] is not None or row['actual_fusion_rmse'] is not None)]
        wins = sum(row['actual_region_rmse'] < min(
            value for value in (row['actual_interp_rmse'],row['actual_fusion_rmse'])
            if value is not None) for row in paired)
        records['region_spectral_pair_cases'] = len(paired)
        records['region_better_than_best_spectral'] = wins
        records['region_worse_or_tied_vs_best_spectral'] = len(paired) - wins
        for expert in EXPERTS:
            values = [row[f'actual_{expert}_rmse'] for row in rows
                      if row[f'actual_{expert}_rmse'] is not None]
            records[f'{expert}_eligible'] = len(values)
            records[f'{expert}_rmse'] = float(np.sqrt(np.mean(np.square(values)))) if values else float('nan')
        by_scenario.append(records)
    with (target.parent / 'holdout_expert_by_scenario.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(by_scenario[0])); writer.writeheader(); writer.writerows(by_scenario)
    tmp = target.with_suffix('.tmp')
    joblib.dump(fitted, tmp); os.replace(tmp, target)
    summary = {key: value for key, value in fitted.items() if key != 'models'}
    (target.parent / 'TRAINING_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Source-held-out squared-risk Router RMSE:', fitted['holdout_router_rmse'], flush=True)
    print('Previous source-held-out Router RMSE:', previous['holdout_router_rmse'], flush=True)
    print('Target checkpoint:', target, flush=True)


def train(args):
    if getattr(args, 'from_cache', False):
        return train_from_cache(args)
    from src.pipeline import _load_accepted_spectral_hybrid_v2
    spec=resolve_experiment(args)
    if args.region_expert in ('refined', 'lineaware') and args.objective != 'log_rmse':
        raise ValueError('Aligned Region router currently uses log_rmse calibration')
    val=tuple(spectral_validation_cores(spec))
    shadow_name, three = shadow_experiment(spec,val)
    rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0))
    if int(os.environ.get('WORLD_SIZE',1)) != 2 or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError('Run three-expert calibration with torchrun on 2 CUDA GPUs')
    torch.cuda.set_device(local); device=torch.device('cuda',local)
    torch.set_num_threads(1)
    dist.init_process_group('nccl')
    try:
        _, _, _, val_c,val_i,wave=load_core_data(spec,val)
        hybrid,mean,std,_,gate_path,backbone_path=_load_accepted_spectral_hybrid_v2(device)
        fusion_path=checkpoint_path(config.MODELS_DIR)
        fusion,saved=load_fusion_checkpoint(fusion_path,device,gate_path,wave,mean,std)
        if tuple(saved['validation_cores']) != val or saved.get('backbone_sha256') != sha(backbone_path):
            raise RuntimeError('Fusion source validation provenance mismatch')
        if args.region_expert == 'lineaware':
            topology,tck,geometry,shadow_tpath,shadow_gpath = (
                load_shadow_lineaware_region(spec, val, wave, device))
        else:
            topology,tck,geometry,shadow_tpath,shadow_gpath=load_shadow_region(spec,val,device)
        for model in (hybrid,fusion,topology,geometry):
            model.eval()
        # The deployed pair must match the exact definition used for labels.
        if args.region_expert == 'lineaware':
            final_root = Path(config.MODELS_DIR) / 'tpssr' / 'region_lineaware_trial_v1'
        else:
            final_root = Path(config.MODELS_DIR) / 'tpssr'
        tag = str(config.TEST_DATASET_FOLDER).lower()
        final_gpath = final_root / f'geometry_{tag}' / 'best_model.pt'
        final_tpath = final_root / f'topology_{tag}' / 'best_model.pt'
        if not final_gpath.is_file() or not final_tpath.is_file():
            raise FileNotFoundError('Final Region checkpoint pair missing: ' + str(final_root))
        samples=[]
        # Distributed deterministic sampling, exactly balanced across core and damage families.
        n_per_core=args.samples_per_case
        jobs=[(ci,case,severity,k) for ci in (0,1)
              for case in SCENARIOS for severity in range(1,6) for k in range(n_per_core)]
        for j,(ci,case,severity,k) in enumerate(jobs):
            if j%2 != rank:continue
            rng=np.random.default_rng(args.seed+10331*j)
            row,rmses,eligible=make_training_case(val_c[ci],val_i[ci],rng,
                case,severity,mean,std,wave,hybrid,fusion,(topology,tck,geometry),
                device,args.region_batch,region_expert=args.region_expert)
            samples.append((ci,row,rmses,eligible,case,severity))
            if len(samples)%25==0:
                print(f'[GPU{rank}] three-expert labels: {len(samples)}',flush=True)
        gathered=[None,None]
        dist.all_gather_object(gathered,samples)
        if rank == 0:
            combined=[item for group in gathered for item in group]
            ck=path_for(config.MODELS_DIR, objective=args.objective, region_expert=args.region_expert)
            if ck.exists():
                raise FileExistsError('Three-expert Router exists; it will not be overwritten')
            cores=np.asarray([s[0] for s in combined],np.int64)
            features=np.stack([s[1] for s in combined]);err=np.stack([s[2] for s in combined])
            eligible=np.stack([s[3] for s in combined])
            fitted=fit_models(features,err,eligible,cores,val,objective=args.objective,
                              case_meta=[(s[4],s[5]) for s in combined])
            fitted.update(region_expert=args.region_expert,
                gate_sha256=sha(gate_path),fusion_sha256=sha(fusion_path),
                region_geometry_sha256=sha(final_gpath),
                region_topology_sha256=sha(final_tpath),
                shadow_region_geometry_sha256=sha(shadow_gpath),shadow_region_topology_sha256=sha(shadow_tpath),
                shadow_training_cores=list(three),shadow_experiment=shadow_name)
            ck.parent.mkdir(parents=True,exist_ok=True)
            audit=fitted.pop('_holdout_details')
            # Store all calibration labels once to avoid recomputing expensive
            # Region predictions for subsequent source-only diagnostic work.
            cache=ck.parent/'calibration_labels.npz'
            with cache.open('wb') as stream:
                np.savez_compressed(stream,x=features,err=err,present=eligible,core=cores,
                    scenario=np.asarray([s[4] for s in combined]),
                    severity=np.asarray([s[5] for s in combined],np.int16),
                    gate_sha256=np.asarray(sha(gate_path)),
                    fusion_sha256=np.asarray(sha(fusion_path)),
                    shadow_geometry_sha256=np.asarray(sha(shadow_gpath)),
                    shadow_topology_sha256=np.asarray(sha(shadow_tpath)),
                    region_expert=np.asarray(args.region_expert),
                    final_geometry_sha256=np.asarray(sha(final_gpath)),
                    final_topology_sha256=np.asarray(sha(final_tpath)))
            import csv
            report=ck.parent/'holdout_case_audit.csv'
            with report.open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(audit[0]))
                writer.writeheader();writer.writerows(audit)
            tmp=ck.with_suffix('.tmp');joblib.dump(fitted,tmp);os.replace(tmp,ck)
            summary={k:v for k,v in fitted.items() if k!='models'}
            (ck.parent/'TRAINING_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
            print('Three-expert router trained. Held-out SWIR core result:', summary['holdout_router_rmse'],flush=True)
            print('Checkpoint:',ck,flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()




# Optional two-way spectral selector. Region is reserved for spectra with
# fewer than two observed anchors; the decision is based on measured support,
# not synthetic scenario labels or a fitted rule.
ARCH_BINARY = 'spectral_binary_router_with_region_fallback_v1'
BINARY_EXPERTS = ('interp', 'fusion')


def binary_path(base):
    return Path(base) / 'tpssr' / 'learned_router2_lineaware' / 'best_model.joblib'


def predict_binary_risks(payload, x, available):
    if (payload.get('architecture') != ARCH_BINARY or
            tuple(payload.get('expert_names', ())) != BINARY_EXPERTS or
            tuple(payload.get('feature_names', ())) != FEATURE_NAMES):
        raise RuntimeError('Binary Router checkpoint schema mismatch')
    x = np.asarray(x, np.float32)
    available = np.asarray(available, bool)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or available.shape != (len(x), 3):
        raise ValueError('Binary Router input shape mismatch')
    spectral = np.column_stack([np.exp(np.clip(model.predict(x), -18, 5))
                                for model in payload['models']])
    out = np.full((len(x), 3), np.inf, np.float64)
    out[:, :2] = np.where(available[:, :2], spectral, np.inf)
    fallback = ~available[:, :2].any(axis=1)
    if np.any(fallback & ~available[:, 2]):
        raise RuntimeError('No Region prediction available for unsupported spectra')
    out[fallback, 2] = 0.0
    return out


def load_binary_checkpoint(path, gate, fusion, geometry, topology, source_cores):
    payload = joblib.load(path)
    if (payload.get('architecture') != ARCH_BINARY or
            payload.get('region_expert') != 'lineaware' or
            tuple(payload.get('feature_names', ())) != FEATURE_NAMES or
            tuple(payload.get('expert_names', ())) != BINARY_EXPERTS or
            tuple(payload.get('source_cores', ())) != tuple(source_cores)):
        raise RuntimeError('Binary Router schema or source split mismatch')
    for name, path in (('gate_sha256', gate), ('fusion_sha256', fusion),
                       ('region_geometry_sha256', geometry),
                       ('region_topology_sha256', topology)):
        if payload.get(name) != sha(path):
            raise RuntimeError('Binary Router frozen expert mismatch: ' + name)
    return payload


def fit_binary_models(x, err, present, core, source_cores):
    x = np.asarray(x, np.float32)
    err = np.asarray(err, np.float64)
    present = np.asarray(present, bool)
    core = np.asarray(core, np.int64)
    if (x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or
            err.shape != (len(x), 3) or present.shape != err.shape or
            core.shape != (len(x),) or set(np.unique(core)) != {0, 1} or
            not present[:, 2].all()):
        raise ValueError('Malformed binary calibration labels')
    models = []
    for j in range(2):
        fitting = (core == 0) & present[:, j] & np.isfinite(err[:, j])
        if fitting.sum() < 50:
            raise RuntimeError('Insufficient spectral fitting examples')
        est = HistGradientBoostingRegressor(max_iter=130, learning_rate=.045,
             max_leaf_nodes=12, min_samples_leaf=12, l2_regularization=1.5,
             random_state=8207 + j)
        est.fit(x[fitting], np.log(np.maximum(err[fitting, j], 1e-6)))
        models.append(est)
    return {'architecture': ARCH_BINARY, 'feature_names': FEATURE_NAMES,
            'expert_names': BINARY_EXPERTS, 'models': models,
            'region_expert': 'lineaware', 'source_cores': list(source_cores),
            'fit_cores': [source_cores[0]], 'holdout_cores': [source_cores[1]],
            'selection_policy': 'region_only_when_spectral_unsupported_else_learned_binary',
            'observed_anchor_minimum': 2}


def train_binary_from_cache(args):
    """Fit the spectral selector and audit genuine expert counterfactuals on holdout."""
    import csv
    if args.region_expert != 'lineaware':
        raise ValueError('Binary Router requires Line-aware Region')
    spec = resolve_experiment(args)
    val = tuple(spectral_validation_cores(spec))
    original = path_for(config.MODELS_DIR, 'log_rmse', 'lineaware')
    target = binary_path(config.MODELS_DIR)
    if target.exists():
        raise FileExistsError('Binary Router already exists: ' + str(target))
    root = Path(config.MODELS_DIR) / 'tpssr'
    tag = spec.model_tag
    gate = root / 'spectral_gate/band/best_model.pt'
    fusion = root / 'neighbor_fusion/best_model.pt'
    geom = root / 'region_lineaware_trial_v1' / f'geometry_{tag}' / 'best_model.pt'
    topo = root / 'region_lineaware_trial_v1' / f'topology_{tag}' / 'best_model.pt'
    old = load_checkpoint(original, gate, fusion, geom, topo, val,
                          objective='log_rmse', region_expert='lineaware')
    cache = original.parent / 'calibration_labels.npz'
    if not cache.is_file():
        raise FileNotFoundError('Run router3_lineaware first to generate calibration labels: '+str(cache))
    expected = {'gate_sha256': sha(gate), 'fusion_sha256': sha(fusion),
                'shadow_geometry_sha256': old['shadow_region_geometry_sha256'],
                'shadow_topology_sha256': old['shadow_region_topology_sha256'],
                'final_geometry_sha256': sha(geom), 'final_topology_sha256': sha(topo),
                'region_expert': 'lineaware'}
    with np.load(cache, allow_pickle=False) as data:
        for field, value in expected.items():
            if field not in data or str(data[field].item()) != value:
                raise RuntimeError('Cached labels mismatch: ' + field)
        x = np.asarray(data['x'], np.float32)
        err = np.asarray(data['err'], np.float64)
        present = np.asarray(data['present'], bool)
        core = np.asarray(data['core'], np.int64)
        scenarios = np.asarray(data['scenario']).astype(str)
        severity = np.asarray(data['severity'], np.int64)
    if (len(x) < 200 or scenarios.shape != (len(x),) or severity.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(err[present]).all() or
            np.any(np.isfinite(err[~present])) or np.any(err[present] < 0)):
        raise RuntimeError('Malformed cached calibration data')
    payload = fit_binary_models(x, err, present, core, val)
    payload.update(gate_sha256=sha(gate), fusion_sha256=sha(fusion),
                   region_geometry_sha256=sha(geom), region_topology_sha256=sha(topo),
                   calibration_source=str(cache), calibration_source_sha256=sha(cache),
                   shadow_region_geometry_sha256=old['shadow_region_geometry_sha256'],
                   shadow_region_topology_sha256=old['shadow_region_topology_sha256'])
    hold = core == 1
    risks = predict_binary_risks(payload, x[hold], present[hold])
    route = np.argmin(risks, axis=1)
    e = err[hold]; a = present[hold]
    idx = np.arange(len(e))
    chosen = e[idx, route]
    if not np.isfinite(chosen).all():
        raise RuntimeError('Binary Router selected an unavailable expert')
    original_risks = predict_risks(old, x[hold], a)
    old_route = np.argmin(original_risks, axis=1)
    full_oracle = np.argmin(np.where(a, e, np.inf), axis=1)
    spectral_available = a[:, :2].any(axis=1)
    better_region = spectral_available & (e[:, 2] < np.min(np.where(a[:, :2], e[:, :2], np.inf), axis=1))
    expected_fallback = ~spectral_available
    if not np.array_equal(route == 2, expected_fallback):
        raise RuntimeError('Binary Router violated Region fallback restriction')
    def pooled(values):
        return float(np.sqrt(np.mean(np.square(values)))) if len(values) else None
    report = []
    for k in range(len(e)):
        report.append(dict(scenario=str(scenarios[hold][k]), severity=int(severity[hold][k]),
             observed_fraction=float(x[hold][k, 0]),
             spectral_available=bool(spectral_available[k]),
             region_forced=bool(expected_fallback[k]),
             region_beats_best_spectral=bool(better_region[k]),
             actual_interp_rmse=float(e[k, 0]) if a[k, 0] else '',
             actual_fusion_rmse=float(e[k, 1]) if a[k, 1] else '',
             actual_region_rmse=float(e[k, 2]),
             selected_binary=BINARY_EXPERTS[route[k]] if route[k] < 2 else 'region',
             actual_binary_rmse=float(chosen[k]),
             selected_router3=EXPERTS[old_route[k]],
             actual_router3_rmse=float(e[k, old_route[k]]),
             oracle_expert=EXPERTS[full_oracle[k]]))
    summary=[]
    for scenario in sorted(set(scenarios[hold])):
        rows = [i for i, r in enumerate(report) if r['scenario'] == scenario]
        both = [i for i in rows if spectral_available[i]]
        binary_losses = sum(e[i, route[i]] > e[i, old_route[i]] for i in rows)
        summary.append(dict(scenario=scenario, cases=len(rows), spectral_supported=len(both),
            no_spectral_support=len(rows)-len(both),
            region_best_when_spectral_supported=int(better_region[both].sum()),
            region_win_fraction=float(better_region[both].mean()) if both else None,
            region_rmse_same_supported=pooled(e[both, 2]),
            best_spectral_oracle_rmse_same_supported=pooled(
                np.min(np.where(a[both,:2], e[both,:2], np.inf),axis=1)) if both else None,
            binary_rmse=pooled(chosen[rows]),
            previous_router3_rmse=pooled(e[rows,old_route[rows]]),
            binary_worse_cases=binary_losses))
    detailed=[]
    for scenario in sorted(set(scenarios[hold])):
        for sev in range(1, 6):
            rows=[i for i,r in enumerate(report) if r['scenario']==scenario and r['severity']==sev]
            if not rows:
                continue
            supported=[i for i in rows if spectral_available[i]]
            detailed.append(dict(scenario=scenario, severity=sev, cases=len(rows),
                spectral_supported=len(supported),
                region_required=len(rows)-len(supported),
                region_best_among_supported=int(better_region[supported].sum()),
                region_win_fraction=float(better_region[supported].mean()) if supported else None,
                binary_rmse=pooled(chosen[rows]),
                previous_router3_rmse=pooled(e[rows,old_route[rows]])))
    payload.update(holdout_binary_rmse=pooled(chosen),
        holdout_previous_router3_rmse=pooled(e[idx, old_route]),
        holdout_full_oracle_rmse=pooled(e[idx, full_oracle]),
        holdout_spectral_supported=int(spectral_available.sum()),
        holdout_region_best_when_spectral_supported=int(better_region.sum()),
        holdout_region_forced=int(expected_fallback.sum()),
        holdout_cases=len(chosen))
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent/'holdout_binary_case_audit.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(report[0]));writer.writeheader();writer.writerows(report)
    with (target.parent/'holdout_region_risk_by_scenario.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    with (target.parent/'holdout_region_risk_by_scenario_severity.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(detailed[0]));writer.writeheader();writer.writerows(detailed)
    temp=target.with_suffix('.tmp')
    joblib.dump(payload,temp);os.replace(temp,target)
    (target.parent/'TRAINING_SUMMARY.json').write_text(json.dumps(
        {k:v for k,v in payload.items() if k!='models'},indent=2),encoding='utf-8')
    print('Binary held-out RMSE:',payload['holdout_binary_rmse'],flush=True)
    print('Previous Router3 same-label held-out RMSE:',payload['holdout_previous_router3_rmse'],flush=True)
    print('Region oracle wins with spectral support:', payload['holdout_region_best_when_spectral_supported'],
          '/',payload['holdout_spectral_supported'],flush=True)
    print('Region-required cases:',payload['holdout_region_forced'],flush=True)
    print('Case and scenario audits:',target.parent,flush=True)
    print('Checkpoint:',target,flush=True)


# Source-core cross-fitting and direct three-expert ranking. Cached labels are
# generated by the frozen models; destination test data are never read here.

def _fit_rank_family(family, x, err, present):
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.dummy import DummyRegressor
    models = []
    for k in range(3):
        fit = present[:, k] & np.isfinite(err[:, k])
        if not np.any(fit):
            raise RuntimeError('Cannot fit a router with an absent expert: ' + EXPERTS[k])
        if family == 'log_risk_original':
            estimator = HistGradientBoostingRegressor(max_iter=130, learning_rate=.045,
                max_leaf_nodes=12, min_samples_leaf=12, l2_regularization=1.5,
                random_state=8207+k, early_stopping=False)
        elif family in ('log_risk_regularized', 'pairwise_log_ratio'):
            estimator = HistGradientBoostingRegressor(max_iter=170, learning_rate=.035,
                max_leaf_nodes=7, min_samples_leaf=25, l2_regularization=4.0,
                random_state=8207+k, early_stopping=False)
        elif family == 'log_risk_extratrees':
            estimator = ExtraTreesRegressor(n_estimators=120, min_samples_leaf=10,
                max_features=.85, n_jobs=1, random_state=8207+k)
        else:
            raise ValueError('Unknown rank family ' + family)
        target = np.log(np.maximum(err[fit, k], 1e-6))
        if fit.sum() < 35:
            estimator = DummyRegressor(strategy='median')
        estimator.fit(x[fit], target)
        models.append(estimator)
    payload = dict(architecture=ARCH_RANK_CV, model_family=family,
                   risk_target='log_rmse', feature_names=FEATURE_NAMES,
                   expert_names=EXPERTS, models=models)
    if family == 'pairwise_log_ratio':
        pair_models = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            fit = present[:,i] & present[:,j] & np.isfinite(err[:,i]) & np.isfinite(err[:,j])
            if fit.sum() < 35:
                raise RuntimeError(f'Insufficient matched risk labels for pair {i}/{j}')
            y = np.clip(np.log(np.maximum(err[fit,i],1e-6)) -
                        np.log(np.maximum(err[fit,j],1e-6)), -6., 6.)
            model = HistGradientBoostingRegressor(max_iter=190, learning_rate=.035,
                max_leaf_nodes=8, min_samples_leaf=24, l2_regularization=4.0,
                early_stopping=False, random_state=9811+i*3+j)
            model.fit(x[fit], y)
            pair_models.append(model)
        payload['pair_models'] = pair_models
    return payload


def _route_score(payload, x, err, present):
    risk = predict_risks(payload, x, present)
    choice = np.argmin(risk, axis=1)
    chosen = err[np.arange(len(choice)), choice]
    if not np.all(np.isfinite(chosen)):
        raise RuntimeError('Rank Router selected an unavailable expert')
    # Each training item represents a corrupted pixel. Weight its squared
    # error by its masked-band fraction, matching pooled masked-voxel RMSE.
    weight = np.asarray(x[:,FEATURE_NAMES.index('missing_fraction')], float)
    if not np.all(np.isfinite(weight)) or np.any(weight <= 0):
        raise RuntimeError('Non-positive number of missing bands in calibration labels')
    return dict(rmse=float(np.sqrt(np.average(chosen ** 2, weights=weight))),
                case_rmse=float(np.sqrt(np.mean(chosen ** 2))),
                choice=choice, risk=risk, chosen=chosen, weight=weight)


def _crossfit_rank(x, err, present, core, families=None):
    families = tuple(families or ('log_risk_original','log_risk_regularized',
                                  'log_risk_extratrees','pairwise_log_ratio'))
    outcomes = []
    for family in families:
        folds = []
        try:
            for held in (0, 1):
                tr, va = core != held, core == held
                candidate = _fit_rank_family(family, x[tr], err[tr], present[tr])
                score = _route_score(candidate, x[va], err[va], present[va])
                folds.append((held, score))
        except (RuntimeError, ValueError) as exc:
            outcomes.append(dict(family=family, error=str(exc)))
            continue
        numerator = sum(float((r['weight'] * r['chosen']**2).sum()) for _, r in folds)
        denominator = sum(float(r['weight'].sum()) for _, r in folds)
        family_score = float(np.sqrt(numerator/denominator))
        outcomes.append(dict(family=family, pooled_oof_rmse=family_score,
                             folds={str(h):dict(rmse=r['rmse'], case_rmse=r['case_rmse'])
                                    for h,r in folds}))
    valid = [o for o in outcomes if 'pooled_oof_rmse' in o]
    if not valid:
        raise RuntimeError('No three-expert ranking candidate could be fitted')
    # Choose a single family using source-only leave-one-acquisition-out CV.
    winner = min(valid, key=lambda o: (o['pooled_oof_rmse'], o['family']))['family']
    return winner, outcomes


def train_rank_from_cache(args):
    import csv
    if args.region_expert != 'lineaware':
        raise ValueError('RankCV requires the frozen Line-aware Region')
    spec = resolve_experiment(args)
    validation = tuple(spectral_validation_cores(spec))
    old = path_for(config.MODELS_DIR, 'log_rmse', 'lineaware')
    target = path_for(config.MODELS_DIR, 'rank_cv', 'lineaware')
    if target.exists():
        raise FileExistsError('Refusing to overwrite a fitted RankCV checkpoint: ' + str(target))
    base = Path(config.MODELS_DIR) / 'tpssr'
    tag = spec.model_tag
    gate = base / 'spectral_gate' / 'band' / 'best_model.pt'
    fusion = base / 'neighbor_fusion' / 'best_model.pt'
    final_root = base / 'region_lineaware_trial_v1'
    geometry = final_root / f'geometry_{tag}' / 'best_model.pt'
    topology = final_root / f'topology_{tag}' / 'best_model.pt'
    previous = load_checkpoint(old, gate, fusion, geometry, topology, validation,
                               objective='log_rmse', region_expert='lineaware')
    cache = old.parent / 'calibration_labels.npz'
    if not cache.is_file():
        raise FileNotFoundError('Source-only calibration cache missing: ' + str(cache))
    expected = dict(gate_sha256=sha(gate), fusion_sha256=sha(fusion),
        shadow_geometry_sha256=previous['shadow_region_geometry_sha256'],
        shadow_topology_sha256=previous['shadow_region_topology_sha256'],
        final_geometry_sha256=sha(geometry), final_topology_sha256=sha(topology),
        region_expert='lineaware')
    with np.load(cache, allow_pickle=False) as saved:
        for key, value in expected.items():
            if key not in saved or str(saved[key].item()) != value:
                raise RuntimeError('Cached expert labels mismatch frozen checkpoints: ' + key)
        x = np.asarray(saved['x'],np.float32)
        err = np.asarray(saved['err'],np.float64)
        present = np.asarray(saved['present'],bool)
        core = np.asarray(saved['core'],np.int64)
        scenarios = np.asarray(saved['scenario']).astype(str)
        severities = np.asarray(saved['severity'],np.int64)
    if (x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or err.shape != (len(x),3)
            or present.shape != err.shape or core.shape != (len(x),)
            or scenarios.shape != (len(x),) or severities.shape != (len(x),)
            or set(np.unique(core)) != {0,1} or min(np.bincount(core)) < 100
            or not np.isfinite(x).all() or not np.isfinite(err[present]).all()
            or np.any(np.isfinite(err[~present])) or np.any(err[present] < 0)
            or not np.all(present.any(axis=1))):
        raise RuntimeError('Invalid source calibration cache')
    winner, leaderboard = _crossfit_rank(x, err, present, core)
    # Compare on the SAME source holdout as the previous Router (core 1).
    va = core == 1
    old_holdout = _route_score(previous, x[va], err[va], present[va])
    selected_holdout_model = _fit_rank_family(winner, x[core == 0],
        err[core == 0], present[core == 0])
    new_holdout = _route_score(selected_holdout_model, x[va], err[va], present[va])
    # After source-only CV selects the model family, use both source
    # calibration acquisitions to fit the deployment router. The reported
    # OOF scores are model-selection diagnostics, not final test estimates.
    final = _fit_rank_family(winner, x, err, present)
    final.update(region_expert='lineaware',
        source_cores=list(validation), fit_cores=list(validation),
        crossfit_cores=list(validation), selected_family=winner,
        gate_sha256=sha(gate), fusion_sha256=sha(fusion),
        region_geometry_sha256=sha(geometry), region_topology_sha256=sha(topology),
        shadow_region_geometry_sha256=previous['shadow_region_geometry_sha256'],
        shadow_region_topology_sha256=previous['shadow_region_topology_sha256'],
        calibration_source_sha256=sha(cache), training_samples=int(len(x)),
        leaderboard=leaderboard,
        oof_previous_router_core1_rmse=old_holdout['rmse'],
        oof_new_router_core1_rmse=new_holdout['rmse'],
        oof_previous_router_core1_case_rmse=old_holdout['case_rmse'],
        oof_new_router_core1_case_rmse=new_holdout['case_rmse'],
        validation_note=('Leave-one-source-acquisition-out cross-fitting selects a model family; '
                         'the final model is refit on both source calibration acquisitions. '
                         'These CV scores are not an independent test estimate. '
                         'No destination cores were used for training.'))
    target.parent.mkdir(parents=True, exist_ok=True)
    report = []
    for i, index in enumerate(np.flatnonzero(va)):
        row = dict(scenario=str(scenarios[index]), severity=int(severities[index]),
                   missing_fraction=float(x[index,FEATURE_NAMES.index('missing_fraction')]),
                   previous=EXPERTS[int(old_holdout['choice'][i])],
                   rankcv=EXPERTS[int(new_holdout['choice'][i])],
                   previous_rmse=float(old_holdout['chosen'][i]),
                   rankcv_rmse=float(new_holdout['chosen'][i]),
                   oracle_rmse=float(np.min(err[index,present[index]])))
        report.append(row)
    with (target.parent/'core1_paired_audit.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(report[0]))
        writer.writeheader();writer.writerows(report)
    temp=target.with_suffix('.tmp')
    joblib.dump(final, temp);os.replace(temp,target)
    summary={k:v for k,v in final.items() if k not in ('models','pair_models')}
    (target.parent/'TRAINING_SUMMARY.json').write_text(
        json.dumps(summary,indent=2),encoding='utf-8')
    print('RankCV selected:',winner,flush=True)
    print('Prior Router source-core1 pooled RMSE:',old_holdout['rmse'],flush=True)
    print('New Router source-core1 pooled RMSE:',new_holdout['rmse'],flush=True)
    for entry in leaderboard:
        print('Candidate:',entry,flush=True)
    if new_holdout['rmse'] >= old_holdout['rmse']:
        print('WARNING: source core1 RMSE did not improve; evaluate before running 200-mask test.',flush=True)
    print('Checkpoint:',target,flush=True)




# Cost-sensitive three-expert selection fitted on source calibration labels.
# This deliberately uses a distinct checkpoint rather than modifying RankCV.
def _regret_weights(squared_gap, fraction):
    gap = np.maximum(np.asarray(squared_gap, np.float64), 0.)
    fraction = np.asarray(fraction, np.float64)
    positive = gap[np.isfinite(gap) & (gap > 0)]
    if len(positive) == 0:
        return np.ones(len(gap), np.float64)
    cap = float(np.quantile(positive, .95))
    floor = max(float(np.median(positive)) * .01, 1e-14)
    weights = np.maximum(np.minimum(gap, cap), floor) * np.maximum(fraction, 1e-4)
    return weights / max(float(weights.mean()), 1e-14)


def _fit_regret_family(family, x, err, present):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.dummy import DummyClassifier
    if family == 'log_risk_extratrees':
        original = _fit_rank_family('log_risk_extratrees', x, err, present)
        original['architecture'] = ARCH_REGRET_CV
        return original
    if family not in ('cost_oracle', 'cost_pairwise'):
        raise ValueError('Unrecognized cost-sensitive Router family: ' + family)
    fraction = x[:, FEATURE_NAMES.index('missing_fraction')]
    payload = dict(architecture=ARCH_REGRET_CV, model_family=family,
                   risk_target='cost_sensitive_ranking', feature_names=FEATURE_NAMES,
                   expert_names=EXPERTS, models=[])
    if family == 'cost_oracle':
        eligible = present.sum(axis=1) >= 2
        labels = np.argmin(np.where(present[eligible], err[eligible], np.inf), axis=1)
        sorted_losses = np.sort(np.where(present[eligible], err[eligible] ** 2, np.inf), axis=1)
        gap = sorted_losses[:, 1] - sorted_losses[:, 0]
        weight = _regret_weights(gap, fraction[eligible])
        if len(np.unique(labels)) < 2:
            estimator = DummyClassifier(strategy='most_frequent')
        else:
            estimator = ExtraTreesClassifier(n_estimators=180, min_samples_leaf=6,
                max_features=.9, n_jobs=1, random_state=9457)
        estimator.fit(x[eligible], labels, sample_weight=weight)
        payload['oracle_model'] = estimator
    else:
        pair_models = []
        for i,j in ((0,1),(0,2),(1,2)):
            eligible = present[:,i] & present[:,j]
            if int(eligible.sum()) < 25:
                raise RuntimeError(f'Not enough shared labels for expert pair {i}/{j}')
            target = (err[eligible,i]**2 < err[eligible,j]**2).astype(np.int32)
            gap = np.abs(err[eligible,i]**2 - err[eligible,j]**2)
            weight = _regret_weights(gap, fraction[eligible])
            if len(np.unique(target)) < 2:
                estimator = DummyClassifier(strategy='most_frequent')
            else:
                estimator = ExtraTreesClassifier(n_estimators=180, min_samples_leaf=6,
                    max_features=.9, n_jobs=1, random_state=9571 + i*3 + j)
            estimator.fit(x[eligible], target, sample_weight=weight)
            pair_models.append(estimator)
        payload['pair_models'] = pair_models
    return payload


def _crossfit_regret(x, err, present, core):
    families = ('log_risk_extratrees','cost_oracle','cost_pairwise')
    results=[]
    for family in families:
        records=[]
        try:
            for held in (0,1):
                model = _fit_regret_family(family, x[core!=held], err[core!=held], present[core!=held])
                scored = _route_score(model, x[core==held], err[core==held], present[core==held])
                records.append(scored)
            chosen=np.concatenate([r['chosen'] for r in records])
            weight=np.concatenate([r['weight'] for r in records])
            results.append(dict(family=family, mean_case_rmse=float(chosen.mean()),
                pooled_rmse=float(np.sqrt(np.average(chosen**2,weights=weight))),
                fold_mean_case_rmse=[float(r['chosen'].mean()) for r in records]))
        except (ValueError,RuntimeError) as exc:
            results.append(dict(family=family,error=str(exc)))
    valid=[item for item in results if 'mean_case_rmse' in item]
    if not valid:
        raise RuntimeError('All RegretCV candidates failed source-only cross-validation')
    # Match the project's mean-of-mask RMSE headline. Pooled squared error is a tie-break.
    selected=min(valid, key=lambda v:(v['mean_case_rmse'],v['pooled_rmse'],v['family']))
    return selected['family'],results


def train_regret_from_cache(args):
    import csv
    if args.region_expert != 'lineaware':
        raise ValueError('RegretCV requires Line-aware Region')
    spec = resolve_experiment(args)
    validation = tuple(spectral_validation_cores(spec))
    rank_path = path_for(config.MODELS_DIR, objective='rank_cv', region_expert='lineaware')
    target = path_for(config.MODELS_DIR, objective='regret_cv', region_expert='lineaware')
    if target.exists():
        raise FileExistsError('Refusing to overwrite trained RegretCV checkpoint: '+str(target))
    base = Path(config.MODELS_DIR) / 'tpssr'
    tag = spec.model_tag
    gate = base / 'spectral_gate' / 'band' / 'best_model.pt'
    fusion = base / 'neighbor_fusion' / 'best_model.pt'
    root = base / 'region_lineaware_trial_v1'
    geom = root / f'geometry_{tag}' / 'best_model.pt'
    topo = root / f'topology_{tag}' / 'best_model.pt'
    previous = load_checkpoint(rank_path,gate,fusion,geom,topo,validation,
                               objective='rank_cv',region_expert='lineaware')
    original = path_for(config.MODELS_DIR,objective='log_rmse',region_expert='lineaware')
    labels_path = original.parent/'calibration_labels.npz'
    expected = dict(gate_sha256=sha(gate), fusion_sha256=sha(fusion),
        shadow_geometry_sha256=previous['shadow_region_geometry_sha256'],
        shadow_topology_sha256=previous['shadow_region_topology_sha256'],
        final_geometry_sha256=sha(geom), final_topology_sha256=sha(topo),
        region_expert='lineaware')
    with np.load(labels_path,allow_pickle=False) as saved:
        for field,value in expected.items():
            if field not in saved or str(saved[field].item()) != value:
                raise RuntimeError('Stale source-only calibration labels: '+field)
        x=np.asarray(saved['x'],np.float32)
        err=np.asarray(saved['err'],np.float64)
        present=np.asarray(saved['present'],bool)
        core=np.asarray(saved['core'],np.int64)
        scenario=np.asarray(saved['scenario']).astype(str)
        severity=np.asarray(saved['severity'],np.int64)
    if (x.ndim!=2 or x.shape[1]!=len(FEATURE_NAMES) or err.shape!=(len(x),3)
            or present.shape!=err.shape or core.shape!=(len(x),)
            or set(np.unique(core))!={0,1} or min(np.bincount(core)) < 100
            or scenario.shape!=(len(x),) or severity.shape!=(len(x),)
            or not np.isfinite(x).all() or not present.any(axis=1).all()
            or not np.isfinite(err[present]).all() or np.any(np.isfinite(err[~present]))
            or np.any(err[present]<0)):
        raise RuntimeError('Invalid source-only cached expert labels')
    chosen, leaderboard=_crossfit_regret(x,err,present,core)
    # Only source-core validation decides the fitted model family.
    va=core==1
    # Fit the reference family on source core 0 only: comparing against the
    # full two-core RankCV checkpoint here would leak source core 1 into its score.
    old_cv=_fit_rank_family(previous['model_family'],x[~va],err[~va],present[~va])
    old_score=_route_score(old_cv,x[va],err[va],present[va])
    fresh=_fit_regret_family(chosen,x[~va],err[~va],present[~va])
    new_score=_route_score(fresh,x[va],err[va],present[va])
    final=_fit_regret_family(chosen,x,err,present)
    final.update(region_expert='lineaware',source_cores=list(validation),
        fit_cores=list(validation),crossfit_cores=list(validation),
        gate_sha256=sha(gate),fusion_sha256=sha(fusion),
        region_geometry_sha256=sha(geom),region_topology_sha256=sha(topo),
        shadow_region_geometry_sha256=previous['shadow_region_geometry_sha256'],
        shadow_region_topology_sha256=previous['shadow_region_topology_sha256'],
        calibration_source_sha256=sha(labels_path),training_samples=int(len(x)),
        selected_family=chosen,leaderboard=leaderboard,
        source_core1_previous_mean_rmse=float(old_score['chosen'].mean()),
        source_core1_new_mean_rmse=float(new_score['chosen'].mean()),
        source_core1_previous_pooled_rmse=float(old_score['rmse']),
        source_core1_new_pooled_rmse=float(new_score['rmse']),
        validation_note=('Source-only SWIR calibration with leave-one-acquisition-out model choice. '
            'Mean corrupted-pixel RMSE used for selection as a proxy for mean-mask RMSE. '
            'Checkpoint refitted on both source cores; never uses destination test cache for training. '
            'Two-fold CV is small and selection can be unstable. Cost-based ranking scores are not calibrated RMSE.'))
    target.parent.mkdir(parents=True,exist_ok=True)
    detail=[]
    for n,idx in enumerate(np.flatnonzero(va)):
        detail.append(dict(scenario=str(scenario[idx]),severity=int(severity[idx]),
            missing_fraction=float(x[idx,1]),
            previous=EXPERTS[int(old_score['choice'][n])],
            regretcv=EXPERTS[int(new_score['choice'][n])],
            previous_rmse=float(old_score['chosen'][n]),
            regretcv_rmse=float(new_score['chosen'][n]),
            oracle_rmse=float(np.min(err[idx,present[idx]]))))
    with (target.parent/'core1_paired_audit.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(detail[0]));writer.writeheader();writer.writerows(detail)
    tmp=target.with_suffix('.tmp')
    joblib.dump(final,tmp);os.replace(tmp,target)
    summary={key:value for key,value in final.items() if key not in ('models','pair_models','oracle_model')}
    (target.parent/'TRAINING_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('RegretCV source-CV selected:',chosen,flush=True)
    print('RankCV source core-1 mean:',final['source_core1_previous_mean_rmse'],flush=True)
    print('RegretCV source core-1 mean:',final['source_core1_new_mean_rmse'],flush=True)
    for entry in leaderboard:
        print('Candidate:',entry,flush=True)
    print('Checkpoint:',target,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_experiment_args(parser)
    parser.add_argument('--samples-per-case',type=int,default=10,
                        help='Samples per (scenario,severity,source-core); default totals 1000 labels')
    parser.add_argument('--region-batch',type=int,default=16)
    parser.add_argument('--seed',type=int,default=8207)
    parser.add_argument('--objective', choices=('log_rmse','mse'),default='log_rmse')
    parser.add_argument('--binary-from-cache', action='store_true', help='Fit two-expert spectral Router, audit held-out Region risks')
    parser.add_argument('--from-cache', action='store_true', help='Refit from verified source-only calibration labels')
    parser.add_argument('--rank-cv-from-cache', action='store_true', help='Crossfit a three-way rank router on source cached expert losses')
    parser.add_argument('--regret-cv-from-cache', action='store_true', help='Fit cost-sensitive three-expert router on source-only labels')
    parser.add_argument('--region-expert', choices=('raw','refined','lineaware'),default='raw')
    args=parser.parse_args()
    if args.samples_per_case<2:parser.error('Need at least 2 samples for each scenario and severity')
    if args.regret_cv_from_cache:
        train_regret_from_cache(args)
    elif args.rank_cv_from_cache:
        train_rank_from_cache(args)
    elif args.binary_from_cache:
        train_binary_from_cache(args)
    else:
        train(args)

if __name__=='__main__':main()
