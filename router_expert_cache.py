#!/usr/bin/env python3
"""Cache three frozen TPSSR experts once, then replay trained routers on fixed masks.

This is a held-out EVALUATION cache.  Its stored ground truth must never be
used to train or tune a router.  Router fitting belongs to source-only RankCV.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
CORE = ROOT / 'core'
from _launcher_common import env_for
from experiment_config import add_experiment_args, experiment_cli_args, resolve_experiment, spectral_validation_cores
from mask_budget import planned_counts

CACHE_VERSION = 1
DEFAULT_ROUTERS = 'learned_router3_lineaware_rankcv'
SUPPORTED_ROUTERS = {
    'learned_router3_lineaware': ('log_rmse', 'lineaware'),
    'learned_router3_lineaware_mse': ('mse', 'lineaware'),
    'learned_router3_lineaware_rankcv': ('rank_cv', 'lineaware'),
    'learned_router3_lineaware_regretcv': ('regret_cv', 'lineaware'),
    'learned_router2_lineaware': ('binary', 'lineaware'),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_args(parser)
    parser.add_argument('--stage', required=True, choices=['build', 'replay', 'score', 'verify'])
    parser.add_argument('--gpus', type=int, choices=[1, 2], default=2)
    parser.add_argument('--max-masks', type=int, default=200)
    parser.add_argument('--region-batch', type=int, default=32)
    parser.add_argument('--verify-cases', type=int, default=1,
                        help='Per GPU and held-out core, verify one or more cases with Region routing')
    parser.add_argument('--no-verify-region-batch', action='store_true')
    parser.add_argument('--routers', default=DEFAULT_ROUTERS,
                        help='Comma-separated names from the supported Line-aware routers')
    parser.add_argument('--force', action='store_true', help='Rebuild previously completed cache files')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def cache_root(spec):
    return ROOT / 'results' / spec.experiment_name / '_router_expert_cache_v1'


def cache_path(spec, dataset, mask_path):
    return cache_root(spec) / dataset / (mask_path.stem + '.npz')


def input_stat(path):
    stat = Path(path).stat()
    return [int(stat.st_size), int(stat.st_mtime_ns)]


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def atomic_npz(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial.npz')
    try:
        np.savez_compressed(tmp, **data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    tmp = path.with_suffix('.partial.csv')
    with tmp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def describe_case_meta(spec, dataset, mask_path, cube_path, pipeline, case_index, coords, mask, pixels, hashes):
    return {
        'cache_version': CACHE_VERSION, 'experiment': spec.experiment_name,
        'dataset': dataset, 'train': list(spec.train), 'tests': list(spec.tests),
        'mask_file': mask_path.name, 'mask_stat': input_stat(mask_path),
        'cube_stat': input_stat(cube_path),
        'case_index': int(case_index), 'n_pixels': int(len(pixels)),
        'n_voxels': int(len(coords)), 'cube_shape': list(mask.shape),
        **hashes,
        'region_expert': 'lineaware',
        'feature_schema': 'three_expert_quality_router_v1',
        'region_tiling_protocol': 'fixed_union_of_sampled_coordinates_v1',
        'evaluation_only': True,
    }


def build_case(pipeline, cube, mask, coords, wave, mask_path, spec, dataset, meta, index, cube_path, hashes):
    import torch
    from training.neighbor_fusion import crop_at
    from training.three_expert_router import features_for_pixel, refine_region, FEATURE_NAMES
    from src.spatial_inference import reconstruct_mask

    cube = np.asarray(cube)
    mask = np.asarray(mask, bool)
    coords = np.asarray(coords, np.int64)
    pixels, reverse = np.unique(coords[:, :2], axis=0, return_inverse=True)
    features = np.empty((len(pixels), len(FEATURE_NAMES)), dtype=np.float32)
    predictions = np.full((len(coords), 3), np.nan, dtype=np.float32)
    observed = np.zeros((len(pixels), 3), dtype=bool)
    interp_all, fusion_all = [], []

    # Exactly the feature/expert-input path used by predict_mask_learned_router3.
    for i, (yy, xx) in enumerate(pixels):
        patch, damaged, cy, cx = crop_at(cube, mask, int(yy), int(xx), radius=35)
        feat, interp, fusion = features_for_pixel(
            patch, damaged, cy, cx, pipeline.hybrid_mean, pipeline.hybrid_std,
            wave, pipeline.fusion_model, pipeline.hybrid_model, pipeline.device,
        )
        features[i] = feat
        interp_all.append(interp)
        fusion_all.append(fusion)
        observed[i] = (interp is not None, fusion is not None, True)

    sort_idx = np.argsort(reverse, kind='stable')
    counts = np.bincount(reverse, minlength=len(pixels))
    offsets = np.concatenate(([0], np.cumsum(counts)))
    for i in range(len(pixels)):
        which = sort_idx[offsets[i]:offsets[i + 1]]
        bands = coords[which, 2]
        if observed[i, 0]:
            predictions[which, 0] = np.asarray(interp_all[i], np.float32)[bands]
        if observed[i, 1]:
            predictions[which, 1] = np.asarray(fusion_all[i], np.float32)[bands]

    # Counterfactual Region: compute for EVERY evaluation coordinate, not
    # only for the voxels chosen by the currently installed Router.
    with torch.inference_mode():
        reconstructed, _ = reconstruct_mask(
            pipeline.topology_model, pipeline.topology_checkpoint,
            pipeline.geometry_model, cube, mask, pipeline.device,
            eval_coords=coords, batch_size=pipeline.region_batch,
            verify_batch=pipeline.verify_region_batch,
            progress_desc=f'{dataset} {mask_path.stem} Region cache',
        )
    lookup = {(int(y), int(x), int(b)): float(value)
              for b, (locations, values) in reconstructed.items()
              for (y, x), value in zip(locations, values)}
    predictions[:, 2] = [lookup.get(tuple(map(int, c)), np.nan) for c in coords]
    predictions[:, 2], _ = refine_region(cube, mask, coords, predictions[:, 2])

    eligible = observed[reverse]
    if not np.isfinite(predictions[:, 2]).all():
        raise RuntimeError('Region counterfactual has missing predictions: refusing incomplete cache')
    if not np.isfinite(predictions[eligible]).all():
        raise RuntimeError('Spectral/Fusion counterfactual has nonfinite values')
    truth = np.asarray(cube[coords[:, 0], coords[:, 1], coords[:, 2]], np.float32)
    if not np.isfinite(truth).all():
        raise RuntimeError('Nonfinite test truth in evaluation coordinates; cache cannot exactly reproduce RMSE')
    case_meta = describe_case_meta(spec, dataset, mask_path, cube_path, pipeline, index, coords, mask, pixels, hashes)
    case_meta.update(scenario=str(meta['scenario']), severity=int(meta['severity']), repeat=int(meta['repeat']))
    return {
        'metadata': np.asarray(json.dumps(case_meta, sort_keys=True)),
        'coords': coords.astype(np.int32),
        'reverse': reverse.astype(np.int32),
        'features': features,
        'eligible': observed,
        'predictions': predictions,
        'truth': truth,
    }


def validate_case(path, spec, dataset, mask_path, cube_path, *, expected_version=CACHE_VERSION):
    with np.load(path, allow_pickle=False) as loaded:
        meta = json.loads(str(loaded['metadata']))
        if (meta.get('cache_version') != expected_version or
            meta.get('experiment') != spec.experiment_name or
            meta.get('dataset') != dataset or
            meta.get('mask_file') != mask_path.name or
            meta.get('mask_stat') != input_stat(mask_path) or
            meta.get('cube_stat') != input_stat(cube_path) or
            meta.get('evaluation_only') is not True or
            meta.get('region_expert') != 'lineaware' or
            meta.get('region_tiling_protocol') != 'fixed_union_of_sampled_coordinates_v1'):
            raise RuntimeError(f'Stale or incompatible expert cache: {path}')
        from training.three_expert_router import FEATURE_NAMES
        if (loaded['features'].ndim != 2 or
            loaded['features'].shape[1] != len(FEATURE_NAMES) or
            loaded['reverse'].shape != (len(loaded['truth']),) or
            loaded['predictions'].shape != (len(loaded['truth']), 3) or
            loaded['eligible'].shape != (len(loaded['features']), 3) or
            loaded['coords'].shape != (len(loaded['truth']), 3) or
            meta.get('n_voxels') != len(loaded['truth'])):
            raise RuntimeError(f'Invalid cache array shapes: {path}')
    return meta


def load_router_payload(name, model_dir, core_names, cache_meta):
    import joblib
    from training.three_expert_router import (path_for, binary_path, FEATURE_NAMES,
                                               EXPERTS, ARCH, ARCH_MSE, ARCH_RANK_CV, ARCH_REGRET_CV)
    objective, _ = SUPPORTED_ROUTERS[name]
    if objective == 'binary':
        ck_path = binary_path(model_dir)
    else:
        ck_path = path_for(model_dir, objective=objective, region_expert='lineaware')
    if not ck_path.exists():
        raise FileNotFoundError(f'Router checkpoint missing: {ck_path}')
    payload = joblib.load(ck_path)
    architectures = {
        'log_rmse': ARCH, 'mse': ARCH_MSE, 'rank_cv': ARCH_RANK_CV,
        'regret_cv': ARCH_REGRET_CV,
        'binary': 'spectral_binary_router_with_region_fallback_v1',
    }
    if payload.get('architecture') != architectures[objective]:
        raise RuntimeError(f'Wrong Router architecture: {name}')
    if objective != 'binary':
        if (tuple(payload.get('feature_names', ())) != FEATURE_NAMES or
            tuple(payload.get('expert_names', ())) != EXPERTS or
            payload.get('region_expert') != 'lineaware' or
            tuple(payload.get('source_cores', ())) != tuple(core_names)):
            raise RuntimeError(f'Router features/provenance mismatch: {name}')
    else:
        if (tuple(payload.get('feature_names', ())) != FEATURE_NAMES or
            tuple(payload.get('expert_names', ())) != ('interp', 'fusion') or
            payload.get('region_expert') != 'lineaware' or
            tuple(payload.get('source_cores', ())) != tuple(core_names)):
            raise RuntimeError(f'Binary Router features/provenance mismatch: {name}')
    for field in ('gate_sha256', 'fusion_sha256', 'region_geometry_sha256', 'region_topology_sha256'):
        if payload.get(field) != cache_meta.get(field):
            raise RuntimeError(f'Router {name} does not match cached Expert weights: {field}')
    return payload


def predict_from_cache(payload, features, eligible):
    from training.three_expert_router import predict_risks, predict_binary_risks
    features = np.asarray(features, np.float32)
    eligible = np.asarray(eligible, bool)
    if payload['architecture'] == 'spectral_binary_router_with_region_fallback_v1':
        risk = predict_binary_risks(payload, features, eligible)
    else:
        risk = predict_risks(payload, features, eligible)
    return np.argmin(risk, axis=1)


def reconstruct_from_cache(record, payload):
    features = np.asarray(record['features'], np.float32)
    eligible = np.asarray(record['eligible'], bool)
    rev = np.asarray(record['reverse'], np.int64)
    pred = np.asarray(record['predictions'], np.float32)
    truth = np.asarray(record['truth'], np.float32)
    choice = predict_from_cache(payload, features, eligible)
    selected = choice[rev]
    result = pred[np.arange(len(pred)), selected]
    if not np.isfinite(result).all():
        raise RuntimeError('Selected Expert has missing predictions in cache')
    return truth, result, selected


def case_score(truth, prediction):
    diff = np.asarray(prediction, np.float64) - np.asarray(truth, np.float64)
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    mae = float(np.mean(np.abs(diff)))
    return rmse, mae, float(np.square(diff).sum()), int(diff.size)


def worker_build(args, spec, dataset):
    import torch
    from src.evaluation_utils import (load_dataset, split_blocks, load_test_mask,
        list_mask_files, benchmark_case_index, select_eval_coords)
    from src.pipeline import load_pipeline
    import config

    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', str(rank)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    pipeline = load_pipeline(device, region_batch=args.region_batch,
                             verify_region_batch=not args.no_verify_region_batch,
                             learned_fusion=True, learned_router3_lineaware_rankcv=True)
    from src.lineaware_region import checkpoint_paths
    geom_expected, topo_expected = checkpoint_paths(full_split=True)
    if Path(pipeline.geometry_path).resolve() != geom_expected.resolve() or Path(pipeline.topology_path).resolve() != topo_expected.resolve():
        raise RuntimeError('Not using the expected full-split Line-aware Region model')

    hashes = {
        'gate_sha256': fingerprint(pipeline.hybrid_gate_checkpoint_path),
        'fusion_sha256': fingerprint(pipeline.fusion_checkpoint_path),
        'region_geometry_sha256': fingerprint(pipeline.geometry_path),
        'region_topology_sha256': fingerprint(pipeline.topology_path),
    }
    cube, _, split, wave = load_dataset(dataset)
    _, _, test_cube = split_blocks(cube, split)
    cube_path = config.PROCESSED_DATA_ROOT / dataset / 'calibrated_reflectance.npy'
    mask_paths = list_mask_files()
    print(f'Cache build {dataset}: {len(mask_paths)} masks, GPU rank {rank}/{world}', flush=True)
    for i, mask_path in enumerate(mask_paths):
        if i % world != rank:
            continue
        out_path = cache_path(spec, dataset, mask_path)
        if out_path.exists() and not args.force:
            old = validate_case(out_path, spec, dataset, mask_path, cube_path)
            if any(old.get(field) != value for field, value in hashes.items()):
                raise RuntimeError(f'Cached Expert weights changed: {out_path}; rebuild with --force')
            print(f'[{dataset}][{rank}] skip completed {i+1}/{len(mask_paths)} {mask_path.name}', flush=True)
            continue
        t0 = time.monotonic()
        mask, meta = load_test_mask(mask_path)
        stable_id = benchmark_case_index(mask_path)
        coords = select_eval_coords(mask, int(config.SEED) + int(stable_id))
        if len(coords) == 0:
            raise RuntimeError(f'No evaluation coordinates: {mask_path}')
        print(f'[{dataset}][{rank}] caching {i+1}/{len(mask_paths)} {mask_path.name} {len(coords)} voxels', flush=True)
        arrays = build_case(pipeline, test_cube, mask, coords, wave, mask_path, spec, dataset,
                            meta, stable_id, cube_path, hashes)
        atomic_npz(out_path, arrays)
        print(f'[{dataset}][{rank}] cached {out_path.name} in {time.monotonic()-t0:.1f}s', flush=True)


def worker_verify(args, spec, dataset):
    """GPU spot-check for identical predictions between cache and fixed-eval test.

    This is diagnostic comparison, never a Router training signal.  Select
    cached masks in which the installed RankCV actually uses Region.
    """
    import torch
    import config
    from src.evaluation_utils import load_dataset, split_blocks, load_test_mask, list_mask_files
    from src.pipeline import load_pipeline, predict_mask_learned_router3
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', str(rank)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    pipeline = load_pipeline(device, region_batch=args.region_batch,
                             verify_region_batch=not args.no_verify_region_batch,
                             learned_fusion=True, learned_router3_lineaware_rankcv=True)
    from src.lineaware_region import checkpoint_paths
    geometry, topology = checkpoint_paths(full_split=True)
    if (Path(pipeline.geometry_path).resolve() != geometry.resolve() or
        Path(pipeline.topology_path).resolve() != topology.resolve()):
        raise RuntimeError('Parity check requires the full-split Line-aware Region')
    hashes = {
        'gate_sha256': fingerprint(pipeline.hybrid_gate_checkpoint_path),
        'fusion_sha256': fingerprint(pipeline.fusion_checkpoint_path),
        'region_geometry_sha256': fingerprint(pipeline.geometry_path),
        'region_topology_sha256': fingerprint(pipeline.topology_path),
    }
    cube, _, split, _ = load_dataset(dataset)
    _, _, test_cube = split_blocks(cube, split)
    cube_path = config.PROCESSED_DATA_ROOT / dataset / 'calibrated_reflectance.npy'
    comparisons = []
    from src.evaluation_utils import list_mask_files
    for i, mask_path in enumerate(list_mask_files()):
        if i % world != rank or len(comparisons) >= args.verify_cases:
            continue
        path = cache_path(spec, dataset, mask_path)
        meta = validate_case(path, spec, dataset, mask_path, cube_path)
        if any(meta.get(key) != value for key, value in hashes.items()):
            raise RuntimeError(f'Expert checkpoint mismatch for {mask_path}: cached predictions stale')
        with np.load(path, allow_pickle=False) as record:
            truth, cached, chosen = reconstruct_from_cache(record, pipeline.router3_checkpoint)
            # Only spend GPU time on cases that exercise Region and thus test
            # the coordinate-dependent tile parity we're fixing.
            if not np.any(chosen == 2):
                continue
            coords = np.asarray(record['coords'], np.int64)
            mask, _ = load_test_mask(mask_path)
            with torch.inference_mode():
                live = predict_mask_learned_router3(
                    pipeline, test_cube, mask, coords,
                    region_progress=f'{dataset} parity {mask_path.stem}',
                    region_full_eval=True)
            observed = np.asarray(live['prediction'], np.float32)
            if not np.array_equal(live['candidate_method'],
                                  np.asarray(['router3_interp', 'router3_fusion', 'router3_region'])[chosen]):
                raise RuntimeError(f'Router choices differ: {mask_path}')
            delta = np.abs(np.asarray(cached, np.float64) - observed)
            maximum = float(np.max(delta))
            if not np.allclose(cached, observed, rtol=5e-4, atol=5e-5):
                raise RuntimeError(f'Parity failure on {dataset} {mask_path.name}: max abs {maximum:.8g}')
            comparisons.append(dict(dataset=dataset, mask_file=mask_path.name,
                                    max_abs_error=maximum, n_voxels=len(coords),
                                    n_region_voxels=int(np.sum(chosen == 2)),
                                    status='ok'))
            print(f'[{dataset}] GPU{rank} fixed-union parity OK {mask_path.name}: '
                  f'max abs {maximum:.8g}, Region {np.sum(chosen==2)} voxels', flush=True)
    if not comparisons:
        raise RuntimeError(f'No Region-selected masks available for parity checking on {dataset}/GPU{rank}')
    atomic_csv(cache_root(spec) / f'parity_{dataset}_rank{rank}.csv', comparisons)


def worker_replay(args, spec, dataset):
    import config
    from src.evaluation_utils import list_mask_files
    model_dir = config.MODELS_DIR
    root = cache_root(spec)
    names = [part.strip() for part in args.routers.split(',') if part.strip()]
    for name in names:
        if name not in SUPPORTED_ROUTERS:
            raise ValueError(f'Unsupported router {name}; choose from {list(SUPPORTED_ROUTERS)}')
    paths = list_mask_files()
    cube_path = config.PROCESSED_DATA_ROOT / dataset / 'calibrated_reflectance.npy'
    detailed = args.stage == 'score'
    if detailed:
        from src.evaluation_utils import load_dataset, split_blocks, load_test_mask, select_eval_coords, benchmark_case_index
        from src.pipeline import sparse_band_predictions
        from src.spatial_inference import evaluate_reconstruction
        cube, _, split, _ = load_dataset(dataset)
        _, _, test_cube = split_blocks(cube, split)
    if not paths:
        raise RuntimeError('No test masks')
    first = cache_path(spec, dataset, paths[0])
    if not first.is_file():
        raise RuntimeError(f'Incomplete counterfactual cache, missing: {first}')
    signature = validate_case(first, spec, dataset, paths[0], cube_path)
    core_names = spectral_validation_cores(spec)
    payloads = {name: load_router_payload(name, model_dir, core_names, signature) for name in names}
    rows = []
    for i, mask_path in enumerate(paths, 1):
        path = cache_path(spec, dataset, mask_path)
        if not path.exists():
            raise RuntimeError(f'Incomplete expert cache ({i}/{len(paths)}), missing {path}')
        meta = validate_case(path, spec, dataset, mask_path, cube_path)
        for field in ('gate_sha256', 'fusion_sha256', 'region_geometry_sha256', 'region_topology_sha256'):
            if meta[field] != signature[field]:
                raise RuntimeError(f'Different Expert versions found inside cache: {mask_path}')
        if detailed:
            mask, actual_meta = load_test_mask(mask_path)
            sampled = select_eval_coords(mask, int(config.SEED) + int(benchmark_case_index(mask_path)))
            if (str(actual_meta['scenario']) != meta['scenario'] or
                int(actual_meta['severity']) != int(meta['severity']) or
                int(actual_meta['repeat']) != int(meta['repeat'])):
                raise RuntimeError(f'Case metadata changed: {mask_path}')
        with np.load(path, allow_pickle=False) as record:
            if detailed:
                cached_coords = np.asarray(record['coords'], np.int64)
                if not np.array_equal(cached_coords, sampled):
                    raise RuntimeError(f'Evaluation coordinates changed: {mask_path}')
                current_truth = np.asarray(test_cube[cached_coords[:, 0], cached_coords[:, 1], cached_coords[:, 2]], np.float32)
                if not np.array_equal(current_truth, record['truth'], equal_nan=True):
                    raise RuntimeError(f'Original test values differ from expert cache: {mask_path}')
            for name, payload in payloads.items():
                truth, prediction, chosen = reconstruct_from_cache(record, payload)
                rmse, mae, sse, count = case_score(truth, prediction)
                row = dict(router=name, dataset=dataset, mask_file=mask_path.name,
                    scenario=meta['scenario'], severity=meta['severity'], repeat=meta['repeat'],
                    rmse=rmse, mae=mae, squared_error=sse, count=count,
                    filled_voxels=int(np.isfinite(prediction).sum()),
                    interp_voxels=int((chosen == 0).sum()),
                    fusion_voxels=int((chosen == 1).sum()),
                    region_voxels=int((chosen == 2).sum()))
                if detailed:
                    exact_metrics = evaluate_reconstruction(
                        test_cube, sparse_band_predictions(cached_coords, prediction),
                        mask, eval_coords=cached_coords)
                    if not np.isclose(exact_metrics['rmse'], rmse, rtol=1e-5, atol=1e-7):
                        raise RuntimeError(f'Cache and official metric RMSE disagree: {mask_path}')
                    row.update(exact_metrics)
                    row['region_tiling_protocol'] = 'fixed_union_of_sampled_coordinates_v1'
                rows.append(row)
        print(f'[{dataset}] {args.stage} mask {i}/{len(paths)} {mask_path.name}', flush=True)
    path = root / f'{args.stage}_{dataset}.csv'
    atomic_csv(path, rows)
    print('Saved', path)


def replay_summary(spec, names):
    root = cache_root(spec)
    all_rows = []
    for dataset in spec.tests:
        path = root / f'replay_{dataset}.csv'
        with path.open(newline='', encoding='utf-8') as stream:
            all_rows.extend(csv.DictReader(stream))
    summary = []
    for name in names:
        values = [r for r in all_rows if r['router'] == name]
        if not values:
            raise RuntimeError(f'No replay rows for {name}')
        sse = sum(float(row['squared_error']) for row in values)
        n = sum(int(row['count']) for row in values)
        summary.append(dict(router=name, cases=len(values),
            mean_case_rmse=float(np.mean([float(r['rmse']) for r in values])),
            pooled_voxel_rmse=float(np.sqrt(sse/n)),
            region_voxel_fraction=sum(int(r['region_voxels']) for r in values)/n,
            coverage=sum(int(r['filled_voxels']) for r in values)/n))
    atomic_csv(root / 'REPLAY_SUMMARY.csv', summary)
    atomic_csv(root / 'REPLAY_ALL_CASES.csv', all_rows)
    print('Saved', root / 'REPLAY_SUMMARY.csv')
    print('Counterfactual evaluation: Region uses fixed union-coordinate tiles. '           'For final publication, run test.py once with the chosen Router.', flush=True)
    for item in summary:
        print(f"{item['router']}: {item['cases']} masks, mean RMSE={item['mean_case_rmse']:.8f}, "
              f"pooled RMSE={item['pooled_voxel_rmse']:.8f}", flush=True)




def score_summary(spec, names):
    """Official masked-voxel metrics for every precomputed case, without GPU."""
    root = cache_root(spec)
    all_rows = []
    for dataset in spec.tests:
        with (root / f'score_{dataset}.csv').open(newline='', encoding='utf-8') as stream:
            all_rows.extend(csv.DictReader(stream))
    summary = []
    for name in names:
        cases = [row for row in all_rows if row['router'] == name]
        if not cases:
            raise RuntimeError(f'No scored cases for Router {name}')
        n = sum(int(row['count']) for row in cases)
        summary.append(dict(
            router=name, cases=len(cases),
            mean_case_rmse=float(np.mean([float(row['rmse']) for row in cases])),
            pooled_voxel_rmse=float(np.sqrt(sum(float(row['squared_error']) for row in cases) / n)),
            mean_mae=float(np.mean([float(row['mae']) for row in cases])),
            mean_r2=finite_mean(cases, 'r2'),
            mean_sam_deg=finite_mean(cases, 'sam_deg'),
            mean_ssim=finite_mean(cases, 'ssim'),
            coverage=sum(int(row['filled_voxels']) for row in cases) / n,
            region_voxel_fraction=sum(int(row['region_voxels']) for row in cases) / n,
        ))
    atomic_csv(root / 'SCORE_SUMMARY.csv', summary)
    atomic_csv(root / 'SCORE_ALL_CASES.csv', all_rows)
    print('Saved', root / 'SCORE_SUMMARY.csv')
    print('Metrics use exactly the same evaluate_reconstruction function as test.py; '
          'the Region fixed-union protocol still requires a small GPU parity check.', flush=True)
    for item in summary:
        print(f"{item['router']}: {item['cases']} masks, mean RMSE={item['mean_case_rmse']:.8f}, "
              f"SAM={item['mean_sam_deg']:.6f}, SSIM={item['mean_ssim']:.6f}", flush=True)


def finite_mean(rows, field):
    values = [float(row[field]) for row in rows]
    valid = [v for v in values if np.isfinite(v)]
    return float(np.mean(valid)) if valid else float('nan')

def launch(args, spec):
    counts = planned_counts(args.max_masks, len(spec.tests))
    offset = 0
    for i, dataset in enumerate(spec.tests):
        command = [sys.executable]
        common = [str(Path(__file__).resolve()), *experiment_cli_args(spec, active_test=dataset),
                  '--stage', args.stage, '--max-masks', str(args.max_masks),
                  '--region-batch', str(args.region_batch), '--verify-cases', str(args.verify_cases),
                '--routers', args.routers, '--worker']
        if args.no_verify_region_batch:
            common.append('--no-verify-region-batch')
        if args.force:
            common.append('--force')
        if args.stage in ('build', 'verify') and args.gpus == 2:
            command += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
        command += common
        env = env_for(spec, dataset)
        if args.max_masks > 0:
            env['HSI_TOTAL_TEST_MASKS'] = str(args.max_masks)
            env['HSI_EXPERIMENT_MASK_OFFSET'] = str(offset)
            env['HSI_EXPERIMENT_MASK_TOTAL'] = str(args.max_masks)
        print('RUN:', ' '.join(command), flush=True)
        subprocess.run(command, cwd=str(ROOT), env=env, check=True)
        if counts[i] is not None:
            offset += counts[i]
    if args.stage in ('replay', 'score'):
        names = [p.strip() for p in args.routers.split(',') if p.strip()]
        if args.stage == 'score':
            score_summary(spec, names)
        else:
            replay_summary(spec, names)
    elif args.stage == 'verify':
        print('GPU parity spot-check complete. The fixed-union Region protocol matches the cache on tested cases.', flush=True)
    else:
        print('Expert cache built. Run replay or score without GPU.', flush=True)


def main(argv=None):
    args = parse_args(argv)
    spec = resolve_experiment(args)
    if args.max_masks <= 0:
        raise ValueError('Use an explicit positive GLOBAL --max-masks budget for a comparable cache')
    if not args.worker:
        launch(args, spec)
        return
    # Workers are started with env_for to keep the original experiment config exact.
    os.environ.update(env_for(spec, spec.active_test))
    if int(args.max_masks) > 0:
        os.environ['HSI_TOTAL_TEST_MASKS'] = str(args.max_masks)
    if str(CORE) not in sys.path:
        sys.path.insert(0, str(CORE))
    if args.stage == 'build':
        worker_build(args, spec, spec.active_test)
    elif args.stage == 'verify':
        worker_verify(args, spec, spec.active_test)
    else:
        worker_replay(args, spec, spec.active_test)


if __name__ == '__main__':
    main()
