"""Conservative learned selection between interpolation and frozen neighbor fusion.

The Region policy remains fixed: these models cannot supply unbiased Region
training targets because Region already learned from every source core.
Source validation cores supply synthetic spectral comparisons; one core fits
this selector and a different core performs an internal safety check.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.ensemble import HistGradientBoostingRegressor

from training.neighbor_fusion import (
    FusionDataset, checkpoint_path, load_fusion_checkpoint,
)
from training.spectral_data import load_core_data
from training.spectral_gate import sha
from experiment_config import add_experiment_args, resolve_experiment, spectral_validation_cores

FEATURE_NAMES = (
    'observed_fraction', 'missing_fraction', 'longest_gap_fraction',
    'mean_nearest_anchor_distance', 'max_nearest_anchor_distance',
    'missing_band_centroid', 'mean_global_missing_fraction',
    'max_global_missing_fraction', 'donor_fraction', 'donor_similarity',
    'donor_distance', 'distance_to_spatial_support', 'spatial_support_fraction',
    'anchor_variation', 'observed_spectrum_variation',
    'fusion_anchor_disagreement', 'fusion_anchor_max_disagreement',
    'fusion_correction_variation',
)
ARCH = 'spectral_selector_gbr_v1'
LOG_MARGIN = 0.25  # Require a substantial predicted advantage before changing fusion.


def feature_row(feat, donors, donor_meta, donor_valid, fusion_prediction):
    """Observed-only feature vector; truth is never an argument.

    feat, donors, donor_meta and donor_valid come from make_features().
    fusion_prediction is from the frozen fusion model on the erased input.
    """
    feat = np.asarray(feat, np.float64)
    fusion_prediction = np.asarray(fusion_prediction, np.float64)
    mask = feat[:, 2] > 0.5
    if not mask.any():
        raise ValueError('Selector needs at least one masked band')
    anchor = feat[:, 1]
    diff = fusion_prediction[mask] - anchor[mask]
    valid = np.asarray(donor_valid, bool)
    meta = np.asarray(donor_meta, np.float64)[valid]
    center = float(np.mean(feat[mask, 3]))
    row = np.array([
        feat[0, 4], float(mask.mean()), feat[0, 5],
        float(np.mean(feat[mask, 7])), float(np.max(feat[mask, 7])),
        center,
        float(np.mean(feat[mask, 6])), float(np.max(feat[mask, 6])),
        feat[0, 8], float(np.mean(meta[:, 0])) if len(meta) else 1.0,
        float(np.mean(meta[:, 1])) if len(meta) else 1.0,
        feat[0, 9], feat[0, 10],
        float(np.std(anchor[mask])), float(np.std(feat[~mask, 0])) if (~mask).any() else 0.,
        float(np.mean(np.abs(diff))), float(np.max(np.abs(diff))),
        float(np.std(diff)),
    ], dtype=np.float32)
    if row.shape != (len(FEATURE_NAMES),) or not np.isfinite(row).all():
        raise RuntimeError('Nonfinite learned router feature row')
    return row


def choice(model, features, enabled=True):
    """True selects interpolation; False retains the accepted fusion output."""
    features = np.asarray(features, np.float32)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError('Invalid learned router feature matrix')
    if not enabled:
        return np.zeros(len(features), dtype=bool)
    return np.asarray(model.predict(features), np.float64) > LOG_MARGIN


def router_file(base):
    return Path(base) / 'tpssr' / 'learned_router' / 'best_model.joblib'


def load_router_checkpoint(path, gate_path, fusion_path, expected_source_cores):
    payload = joblib.load(path)
    if payload.get('architecture') != ARCH or tuple(payload.get('feature_names', ())) != FEATURE_NAMES:
        raise RuntimeError('Learned Router checkpoint/schema mismatch')
    if payload.get('gate_sha256') != sha(gate_path) or payload.get('fusion_sha256') != sha(fusion_path):
        raise RuntimeError('Learned Router checkpoint does not match frozen experts')
    if tuple(payload.get('source_cores', ())) != tuple(expected_source_cores):
        raise RuntimeError('Learned Router source calibration cores do not match')
    if set(payload.get('fit_cores', ())) & set(payload.get('holdout_cores', ())):
        raise RuntimeError('Learned Router calibration/holdout overlap')
    if not set(payload['fit_cores'] + payload['holdout_cores']).issubset(expected_source_cores):
        raise RuntimeError('Learned Router provenance invalid')
    return payload


def fit_selector(x, sq_interp, sq_fusion, vox, core, source_cores, gate_path, fusion_path):
    """Fit only on first source-validation core; last core is diagnostics only."""
    if len(source_cores) != 2:
        raise RuntimeError('Expected exactly two separate source calibration cores')
    fit = core == 0
    held = core == 1
    if min(int(fit.sum()), int(held.sum())) < 150:
        raise RuntimeError('Not enough distinct-core router calibration cases')
    y = np.log((sq_fusion / vox + 1e-7) / (sq_interp / vox + 1e-7)).clip(-4, 4)
    model = HistGradientBoostingRegressor(
        max_iter=120, learning_rate=.04, max_leaf_nodes=10,
        min_samples_leaf=30, l2_regularization=1.0,
        random_state=7410,
    )
    model.fit(x[fit], y[fit])
    selected = choice(model, x[held])
    base = math.sqrt(float(sq_fusion[held].sum() / vox[held].sum()))
    new_sq = np.where(selected, sq_interp[held], sq_fusion[held])
    candidate = math.sqrt(float(new_sq.sum() / vox[held].sum()))
    # A failed internal check cannot silently deploy a worse-than-fusion router.
    enabled = bool(candidate < base - 1e-7 and selected.any())
    metadata = {
        'architecture': ARCH, 'feature_names': FEATURE_NAMES, 'model': model,
        'enabled': enabled, 'source_cores': list(source_cores),
        'fit_cores': [source_cores[0]], 'holdout_cores': [source_cores[1]],
        'fit_cases': int(fit.sum()), 'holdout_cases': int(held.sum()),
        'selected_holdout_cases': int(selected.sum()),
        'holdout_fusion_rmse': base, 'holdout_candidate_rmse': candidate,
        'gate_sha256': sha(gate_path), 'fusion_sha256': sha(fusion_path),
        'decision_margin_log_ratio': LOG_MARGIN,
        'validation_note': ('Source-only internal selection check, not an independent final test. '
                            'Fusion checkpoint was previously selected using these source validation cores. '
                            'Region is intentionally unchanged, not learned.'),
    }
    return metadata


def train(args):
    from src.pipeline import _load_accepted_spectral_hybrid_v2
    import config
    spec = resolve_experiment(args)
    val_cores = tuple(spectral_validation_cores(spec))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    if world != 2 or args.gpus != 2 or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError('Learned Router calibration requires torchrun on two CUDA GPUs')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(1)
    dist.init_process_group('nccl')
    try:
        torch.manual_seed(args.seed + rank)
        _, _, _, val_c, val_i, wave = load_core_data(spec, val_cores)
        if len(val_cores) != 2:
            raise RuntimeError('Need two distinct entire source validation cores')
        hybrid, mean, std, _, gate_path, backbone_path = _load_accepted_spectral_hybrid_v2(device)
        for p in hybrid.parameters():
            p.requires_grad_(False)
        fusion_path = checkpoint_path(config.MODELS_DIR)
        fusion, saved = load_fusion_checkpoint(fusion_path, device, gate_path, wave, mean, std)
        if tuple(saved['validation_cores']) != val_cores or saved.get('backbone_sha256') != sha(backbone_path):
            raise RuntimeError('Frozen Fusion provenance mismatch')
        for p in fusion.parameters():
            p.requires_grad_(False)
        dataset = FusionDataset(val_cores, val_c, val_i, mean, std, wave,
                                args.samples, args.seed + 62000, False)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            sampler=DistributedSampler(dataset, num_replicas=2,
                                                       rank=rank, shuffle=False),
                            num_workers=0)
        features, errors_s, errors_f, volumes, cores = [], [], [], [], []
        reference = torch.as_tensor(std, device=device).view(1, -1)
        with torch.inference_mode():
            for batch in loader:
                batch = [t.to(device) for t in batch]
                hybrid_pred = hybrid(batch[1])
                pred = fusion(batch[0], hybrid_pred, batch[2], batch[3], batch[4])
                anchor = batch[0][..., 1]
                truth = batch[5]
                missing = batch[6]
                se_s = (((anchor - truth) * reference).square() * missing).sum(1)
                se_f = (((pred - truth) * reference).square() * missing).sum(1)
                nn = missing.sum(1)
                feats = batch[0].cpu().numpy()
                meta = batch[3].cpu().numpy()
                valid = batch[4].cpu().numpy()
                fusion_predictions = pred.cpu().numpy()
                ci = batch[7].cpu().numpy()
                for k in range(len(ci)):
                    if not nn[k].item():
                        continue
                    # Global-band guarded pixels are not routed by this model.
                    if np.max(feats[k, :, 6][feats[k, :, 2] > .5]) >= .995:
                        continue
                    features.append(feature_row(feats[k], None, meta[k], valid[k], fusion_predictions[k]))
                    errors_s.append(float(se_s[k]))
                    errors_f.append(float(se_f[k]))
                    volumes.append(float(nn[k]))
                    cores.append(int(ci[k]))
        payload = (np.asarray(features, np.float32).reshape(-1, len(FEATURE_NAMES)),
                   np.asarray(errors_s, np.float64), np.asarray(errors_f, np.float64),
                   np.asarray(volumes, np.float64), np.asarray(cores, np.int64))
        gathered = [None, None]
        dist.all_gather_object(gathered, payload)
        if rank == 0:
            x, es, ef, nv, c = [np.concatenate([part[i] for part in gathered]) for i in range(5)]
            ck = router_file(config.MODELS_DIR)
            if ck.exists():
                raise FileExistsError('Learned Router already exists; never replace a previous run implicitly')
            fitted = fit_selector(x, es, ef, nv, c, val_cores, gate_path, fusion_path)
            ck.parent.mkdir(parents=True, exist_ok=True)
            tmp = ck.with_suffix('.joblib.tmp')
            joblib.dump(fitted, tmp)
            os.replace(tmp, ck)
            summary = {k: v for k, v in fitted.items() if k != 'model'}
            (ck.parent / 'TRAINING_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
            print('Learned Router internal heldout:',
                  f'Fusion RMSE {fitted["holdout_fusion_rmse"]:.6f}',
                  f'candidate RMSE {fitted["holdout_candidate_rmse"]:.6f}',
                  f'enabled={fitted["enabled"]}', flush=True)
            print('Checkpoint:', ck, flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_args(parser)
    parser.add_argument('--gpus', type=int, choices=[2], default=2)
    parser.add_argument('--samples', type=int, default=3200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=7410)
    args = parser.parse_args()
    if args.samples < 600 or args.samples % 2 or args.batch_size < 1:
        parser.error('samples must be even and >= 600; batch-size must be positive')
    train(args)


if __name__ == '__main__':
    main()
