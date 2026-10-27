"""Train TPSSR using a spectral backbone and geometry-guided completion.

All parameters are fitted on the configured training acquisitions only.
The spectral confidence gate is fitted with its backbone frozen; geometry and
topology are trained sequentially. One model serves both held-out acquisitions.
An already-accepted checkpoint is never overwritten automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from _launcher_common import ROOT, CORE, env_for
from experiment_config import add_experiment_args, resolve_experiment, describe

METHOD_DIR = CORE / 'reconstruction_methods' / 'tpssr'
FAMILY = 'tpssr'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def geometry_training_metadata(spec, folder: Path) -> Path:
    """Cache finite/valid reflectance stats from TRAIN rows only, never test.

    Samples an evenly distributed spatial lattice in the first 80% of each
    training acquisition. Streaming single rows avoids materializing HSI cubes.
    """
    target = folder / f'geometry_train_stats_{spec.model_tag}.npz'
    bands = None
    reference_wave = None
    names = np.asarray(spec.train)
    if target.exists():
        with np.load(target, allow_pickle=False) as ck:
            if tuple(ck['train_datasets'].tolist()) != tuple(spec.train):
                raise RuntimeError(f'Existing Geometry statistics have a different train split: {target}')
            reference_wave = np.asarray(ck['wavelengths_nm'], dtype=np.float32)
            for name in spec.train:
                wave_path = ROOT / 'processed_data' / name / 'wavelengths_nm.npy'
                wave = np.load(wave_path)
                if len(wave) != len(reference_wave) or not np.allclose(wave, reference_wave, atol=0.5, rtol=0):
                    raise RuntimeError(f'Geometry cached stats wavelength mismatch on {name}')
        return target

    sums = sums2 = counts = None
    for name in spec.train:
        folder_data = ROOT / 'processed_data' / name
        cube = np.load(folder_data / 'calibrated_reflectance.npy', mmap_mode='r')
        invalid = np.load(folder_data / 'invalid_mask.npy', mmap_mode='r')
        wave = np.asarray(np.load(folder_data / 'wavelengths_nm.npy'), dtype=np.float32)
        if cube.shape != invalid.shape or cube.shape[2] != len(wave):
            raise RuntimeError(f'Corrupt or inconsistent processed HSI data: {name}')
        if reference_wave is None:
            reference_wave = wave
            bands = len(wave)
            sums = np.zeros(bands, dtype=np.float64)
            sums2 = np.zeros(bands, dtype=np.float64)
            counts = np.zeros(bands, dtype=np.int64)
        elif len(wave) != bands or not np.allclose(wave, reference_wave, atol=0.5, rtol=0):
            raise RuntimeError(f'Train spectral grids differ: {name}')

        n_train_rows = max(1, min(int(round(cube.shape[0] * 0.80)), cube.shape[0] - 1))
        ys = np.unique(np.linspace(0, n_train_rows - 1, min(128, n_train_rows), dtype=np.int64))
        xs = np.unique(np.linspace(0, cube.shape[1] - 1, min(128, cube.shape[1]), dtype=np.int64))
        for y in ys:
            values = np.asarray(cube[int(y), xs, :], dtype=np.float64)
            bad = np.asarray(invalid[int(y), xs, :], dtype=np.bool_)
            good = (~bad) & np.isfinite(values)
            safe = np.where(good, values, 0.0)
            sums += safe.sum(axis=0)
            sums2 += (safe * safe).sum(axis=0)
            counts += good.sum(axis=0)
        print(f'  Normalization sampled from training rows: {name} ({len(ys)}x{len(xs)} spatial samples)', flush=True)

    if bands is None or (counts < 100).any():
        raise RuntimeError('Not enough finite, non-invalid TRAIN pixels to compute Geometry normalization.')
    mean = sums / counts
    var = np.maximum(sums2 / counts - mean * mean, 1e-10)
    std = np.sqrt(var)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError('Non-finite train normalization statistics.')
    folder.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix('.npz.tmp')
    with temp.open('wb') as stream:
        np.savez(stream, n_bands=np.int64(bands), band_mean=mean.astype(np.float32),
                 band_std=std.astype(np.float32), wavelengths_nm=reference_wave,
                 train_datasets=names)
    os.replace(temp, target)
    return target


def launch(command, spec, extra_env=None, dry_run=False):
    print('RUN:', ' '.join(map(str, command)), flush=True)
    if dry_run:
        return
    env = env_for(spec, spec.primary_test)
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    env.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    subprocess.run([str(part) for part in command], cwd=str(ROOT), env=env, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_experiment_args(ap)
    ap.add_argument('--gpus', type=int, choices=[2], default=2)
    ap.add_argument('--dry-run', action='store_true', help='Show training plan without touching GPU/data')
    ap.add_argument('--stage', choices=('all', 'hybrid', 'region', 'fusion', 'router', 'router3', 'router3_rmse', 'router3_aligned', 'router3_lineaware', 'router3_lineaware_mse', 'router3_lineaware_rankcv', 'router3_lineaware_regretcv', 'router2_lineaware'), default='all',
                    help='Optionally restrict the training family; default trains all missing components')
    ap.add_argument('--router-samples', type=int, default=3200)
    ap.add_argument('--router3-samples-per-case', type=int, default=10)
    ap.add_argument('--router-batch-size', type=int, default=16)
    ap.add_argument('--fusion-epochs', type=int, default=8)
    ap.add_argument('--fusion-train-samples', type=int, default=6000)
    ap.add_argument('--fusion-val-samples', type=int, default=1000)
    ap.add_argument('--fusion-batch-size', type=int, default=16)
    ap.add_argument('--hybrid-epochs', type=int, default=5)
    ap.add_argument('--gate-epochs', type=int, default=8)
    ap.add_argument('--topology-epochs', type=int, default=22)
    ap.add_argument('--topology-min-epochs', type=int, default=14)
    ap.add_argument('--topology-patience', type=int, default=5)
    ap.add_argument('--topology-train-samples', type=int, default=4200)
    ap.add_argument('--topology-val-samples', type=int, default=900)
    args = ap.parse_args()
    spec = resolve_experiment(args)
    from experiment_config import spectral_validation_cores
    val_cores = spectral_validation_cores(spec) if args.stage != 'region' else ()
    if args.router3_samples_per_case < 2:
        ap.error('Need at least 2 samples per Router3 case and severity')
    if min(args.router_samples, args.router_batch_size, args.fusion_epochs, args.fusion_train_samples, args.fusion_val_samples, args.fusion_batch_size,
           args.hybrid_epochs, args.gate_epochs, args.topology_epochs,
           args.topology_min_epochs, args.topology_patience,
           args.topology_train_samples, args.topology_val_samples) < 1:
        ap.error('All epoch and sample counts must be positive.')

    base = ROOT / 'trained_models' / spec.experiment_name / FAMILY
    hybrid_root = base / 'spectral_backbone' / 'hybrid'
    backbone = hybrid_root / 'best_model.pt'
    diagnostic = hybrid_root / 'diagnostic_best_val.pt'
    gate_dir = base / 'spectral_gate' / 'band'
    gate = gate_dir / 'best_model.pt'
    tag = spec.model_tag
    geometry_candidates = (
        base / f'geometry_{tag}' / 'best_model.pt',
    )
    geometry_checkpoint = next((p for p in geometry_candidates if p.is_file()), None)
    topology = base / f'topology_{tag}' / 'best_model.pt'
    norm_path = base / f'geometry_train_stats_{tag}.npz'

    # Fail early if a previously trained downstream model has lost its upstream
    # weights; do not overwrite or retrain another model under the same tag.
    if topology.is_file() and geometry_checkpoint is None:
        raise RuntimeError('Topology exists but its Geometry source is missing. Restore Geometry rather than retraining it.')
    if gate.is_file() and not (backbone.is_file() or diagnostic.is_file()):
        raise RuntimeError('Confidence gate exists but its pinned spectral backbone is missing. Restore the V1 checkpoint.')

    print('\nTPSSR V2.3 TRAINING: spectral backbone + confidence gate + geometry-guided completion')
    print(describe(spec))
    print('One shared model; held-out TEST acquisitions are never used for training.')
    if val_cores:
        print('Spectral validation cores:', ', '.join(val_cores))
    print('Checkpoint tag:', tag, '(folder label only, not training data)')
    if args.stage in ('router3','router3_rmse','router3_aligned','router3_lineaware','router3_lineaware_mse','router3_lineaware_rankcv','router3_lineaware_regretcv'):
        print('Training a learned selector among all three experts; source-only Region shadow.\n')
    elif args.stage == 'router2_lineaware':
        print('Training a learned two-spectral-expert selector; Region for unsupported spectra.\n')
    elif args.stage == 'router':
        print('Training a conservative spectral selector; Region remains V2.\n')
    else:
        print('Original inference modes keep their accepted policies.\n')

    scripts = (
        ROOT / 'training/spectral_backbone.py', ROOT / 'training/spectral_gate.py',
        METHOD_DIR / 'train_geometry.py',
        METHOD_DIR / 'train_topology.py',
    )
    for script in scripts:
        if not script.is_file():
            raise FileNotFoundError(f'Required V2.3 trainer is missing: {script}')
    if args.dry_run:
        print('DRY RUN: no dataset files are opened and no GPU jobs are submitted.\n')

    if args.stage == 'router2_lineaware':
        if not args.preset:
            raise RuntimeError('Supply an explicit SWIR --preset')
        from experiment_config import selected_lineaware_region
        if not selected_lineaware_region(spec, 'learned_router2_lineaware'):
            raise RuntimeError('Two-spectral-expert Router supports the three defined SWIR splits')
        from training.three_expert_router import binary_path
        checkpoint = binary_path(ROOT / 'trained_models' / spec.experiment_name)
        if checkpoint.is_file():
            print('SKIP Binary Router: checkpoint already exists', checkpoint)
        else:
            print('REFIT Binary Router from source-only cached counterfactual labels (CPU)')
            cmd = [sys.executable, '-m', 'training.three_expert_router',
                   '--preset', str(args.preset), '--region-expert', 'lineaware',
                   '--binary-from-cache']
            launch(cmd, spec, dry_run=args.dry_run)
        return

    if args.stage == 'router3_lineaware_regretcv':
        if not args.preset:
            raise RuntimeError('Use an explicit SWIR --preset')
        from experiment_config import selected_lineaware_region
        if not selected_lineaware_region(spec, 'learned_router3_lineaware_regretcv'):
            raise RuntimeError('RegretCV supports only the three defined SWIR splits')
        from training.three_expert_router import path_for
        checkpoint = path_for(ROOT / 'trained_models' / spec.experiment_name,
                              objective='regret_cv', region_expert='lineaware')
        if checkpoint.is_file():
            print('SKIP RegretCV Router: checkpoint already exists', checkpoint)
        else:
            print('Fit cost-sensitive Router3 on source-only cached labels (CPU)')
            cmd = [sys.executable, '-m', 'training.three_expert_router',
                   '--preset', str(args.preset), '--region-expert', 'lineaware',
                   '--regret-cv-from-cache']
            launch(cmd, spec, dry_run=args.dry_run)
        return

    if args.stage == 'router3_lineaware_rankcv':
        if not args.preset:
            raise RuntimeError('Use an explicit SWIR --preset')
        from experiment_config import selected_lineaware_region
        if not selected_lineaware_region(spec, 'learned_router3_lineaware_rankcv'):
            raise RuntimeError('RankCV supports only the three defined SWIR splits')
        from training.three_expert_router import path_for
        checkpoint = path_for(ROOT / 'trained_models' / spec.experiment_name,
                              objective='rank_cv', region_expert='lineaware')
        if checkpoint.is_file():
            print('SKIP RankCV Router: checkpoint already exists',checkpoint)
        else:
            print('CROSSFIT three-expert RankCV Router using verified source-only cached labels (CPU)')
            cmd = [sys.executable, '-m','training.three_expert_router',
                   '--preset',str(args.preset),'--region-expert','lineaware',
                   '--rank-cv-from-cache']
            launch(cmd,spec,dry_run=args.dry_run)
        return

    if args.stage == 'router3_lineaware_mse':
        if not args.preset:
            raise RuntimeError('Use an explicit SWIR --preset')
        from experiment_config import selected_lineaware_region
        if not selected_lineaware_region(spec, 'learned_router3_lineaware_mse'):
            raise RuntimeError('Squared-risk Line-aware Router supports the three SWIR splits only')
        from training.three_expert_router import path_for
        checkpoint = path_for(ROOT / 'trained_models' / spec.experiment_name,
                              objective='mse', region_expert='lineaware')
        if checkpoint.is_file():
            print('SKIP squared-risk Router: checkpoint already exists', checkpoint)
        else:
            print('REFIT squared-risk Router from verified source-only cached labels (CPU)')
            cmd = [sys.executable, '-m', 'training.three_expert_router',
                   '--preset', str(args.preset), '--region-expert', 'lineaware',
                   '--objective', 'mse', '--from-cache']
            launch(cmd, spec, dry_run=args.dry_run)
        return

    if args.stage in ('router3','router3_rmse','router3_aligned','router3_lineaware'):
        if not args.preset:
            raise RuntimeError('Use an explicit --preset for the three-expert routing run')
        from training.three_expert_router import shadow_experiment
        val = tuple(val_cores)
        shadow_name, shadow_train = shadow_experiment(spec, val)
        from training.three_expert_router import path_for
        from src.lineaware_region import TRIAL_NAME
        lineaware = args.stage == 'router3_lineaware'
        if lineaware:
            from experiment_config import selected_lineaware_region
            if not selected_lineaware_region(spec, 'learned_router3_lineaware'):
                raise RuntimeError('Router3 Line-aware requires one of the three defined SWIR splits')
        shadow_root = ROOT / 'trained_models' / shadow_name / FAMILY
        shadow_base = shadow_root / TRIAL_NAME if lineaware else shadow_root
        shadow_geom = shadow_base / f'geometry_{val[0].lower()}' / 'best_model.pt'
        shadow_top = shadow_base / f'topology_{val[0].lower()}' / 'best_model.pt'
        variant = 'lineaware' if lineaware else 'refined' if args.stage == 'router3_aligned' else 'raw'
        ck = path_for(ROOT / 'trained_models' / spec.experiment_name,
                      objective='mse' if args.stage == 'router3_rmse' else 'log_rmse',
                      region_expert=variant)
        fusion_ck = base / 'neighbor_fusion' / 'best_model.pt'
        if not args.dry_run and not all(p.is_file() for p in (gate,fusion_ck,geometry_candidates[0],topology)):
            raise RuntimeError('Three-expert Router requires all accepted frozen experts first.')
        if lineaware:
            final_base = base / TRIAL_NAME
            final_checkpoints = (final_base / f'geometry_{tag}' / 'best_model.pt',
                                 final_base / f'topology_{tag}' / 'best_model.pt')
            if not args.dry_run and not all(p.is_file() for p in final_checkpoints):
                raise FileNotFoundError('Train the full-split Line-aware Region first: ' + str(final_base))
        if not shadow_geom.is_file() or not shadow_top.is_file():
            print(f'TRAIN shadow Region on {len(shadow_train)} source cores only (2 GPUs)')
            if lineaware:
                cmd=[sys.executable,str(ROOT/'train_region_lineaware.py'),
                     '--train-datasets',*shadow_train,'--test-datasets',*val,
                     '--experiment-name',shadow_name,'--gpus','2']
            else:
                cmd=[sys.executable,str(ROOT/'train.py'),'--train-datasets',*shadow_train,
                     '--test-datasets',*val,'--experiment-name',shadow_name,
                     '--stage','region','--gpus','2',
                     '--topology-epochs',str(args.topology_epochs),
                     '--topology-min-epochs',str(args.topology_min_epochs),
                     '--topology-patience',str(args.topology_patience),
                     '--topology-train-samples',str(args.topology_train_samples),
                     '--topology-val-samples',str(args.topology_val_samples)]
            # Spawn a separate experiment; main model files remain untouched.
            print('RUN:', ' '.join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd,cwd=str(ROOT),check=True)
        else:
            print('SKIP shadow Region: both checkpoints exist')
        if ck.exists():
            print('SKIP three-expert Router: found', ck)
        else:
            print('TRAIN three-expert learned quality Router on source calibration cores (2 GPUs)')
            cmd=[sys.executable,'-m','torch.distributed.run','--standalone',
                 '--nproc_per_node=2','-m','training.three_expert_router',
                 '--preset',str(args.preset),'--samples-per-case',str(args.router3_samples_per_case),
                 '--objective', 'mse' if args.stage == 'router3_rmse' else 'log_rmse',
                 '--region-expert', variant]
            launch(cmd,spec,dry_run=args.dry_run)
        if not args.dry_run and not ck.is_file():
            raise RuntimeError('Three-expert Router calibration did not produce a checkpoint')
        return

    if args.stage == 'router':
        router_ck = base / 'learned_router' / 'best_model.joblib'
        fusion_ck = base / 'neighbor_fusion' / 'best_model.pt'
        if not args.dry_run and not (gate.is_file() and fusion_ck.is_file()):
            raise RuntimeError('Router requires accepted frozen Hybrid and Fusion checkpoints.')
        if router_ck.exists():
            print('SKIP learned router: found', router_ck)
        else:
            print('CALIBRATE learned spectral selector on source cores (2 GPUs)')
            cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node=2', '-m', 'training.learned_router',
                   '--preset', args.preset, '--gpus', '2',
                   '--samples', str(args.router_samples),
                   '--batch-size', str(args.router_batch_size)]
            launch(cmd, spec, dry_run=args.dry_run)
        if not args.dry_run and not router_ck.exists():
            raise RuntimeError('Learned Router calibration did not produce a checkpoint')
        return

    if args.stage == 'fusion':
        fusion_ck = base / 'neighbor_fusion' / 'best_model.pt'
        if not args.dry_run and (not gate.is_file() or not (backbone.is_file() or diagnostic.is_file())):
            raise RuntimeError('Fusion needs the existing accepted gate and matching frozen backbone.')
        if fusion_ck.is_file():
            print('SKIP learned neighbor fusion: found', fusion_ck)
        else:
            print('TRAIN learned neighbor fusion (frozen spectral models, 2 GPUs)')
            cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node=2', '-m', 'training.neighbor_fusion',
                   '--preset', args.preset, '--gpus', '2',
                   '--epochs', str(args.fusion_epochs),
                   '--train-samples', str(args.fusion_train_samples),
                   '--val-samples', str(args.fusion_val_samples),
                   '--batch-size', str(args.fusion_batch_size)]
            launch(cmd, spec, dry_run=args.dry_run)
        if not args.dry_run and not fusion_ck.is_file():
            raise RuntimeError('Fusion training did not produce best_model.pt')
        print('DRY RUN: fusion training not started.' if args.dry_run else f'DONE: fusion checkpoint {fusion_ck}')
        return

    if args.stage in ('all', 'hybrid'):
        if backbone.exists() or diagnostic.exists():
            chosen = backbone if backbone.exists() else diagnostic
            print('SKIP spectral backbone: found', chosen)
        else:
            print('STAGE 1/4: spectral backbone')
            cmd = [sys.executable, '-m', 'training.spectral_backbone', '--preset', args.preset, '--gpus', '2',
                   '--epochs', str(args.hybrid_epochs), '--val-cores', *val_cores]
            if (hybrid_root / 'last_model.pt').exists():
                cmd.append('--resume')
            launch(cmd, spec, dry_run=args.dry_run)
        if not args.dry_run and not (backbone.exists() or diagnostic.exists()):
            raise RuntimeError('Spectral training did not produce an accepted or diagnostic checkpoint.')

        if gate.exists():
            print('SKIP band confidence gate: found', gate)
        else:
            print('STAGE 2/4: confidence gate with a frozen backbone')
            cmd = [sys.executable, '-m', 'training.spectral_gate', '--preset', args.preset, '--gpus', '2',
                   '--epochs', str(args.gate_epochs), '--val-cores', *val_cores]
            if (gate_dir / 'last_model.pt').exists():
                cmd.append('--resume')
            launch(cmd, spec, dry_run=args.dry_run)
        if not args.dry_run and not gate.is_file():
            raise RuntimeError('Band confidence gate has no accepted best_model.pt. '
                               'The inference code refuses diagnostic-only gates; inspect gate validation.')

        if not args.dry_run:
            import torch
            ck = torch.load(gate, map_location='cpu', weights_only=False)
            if ck.get('gate_variant') != 'band':
                raise RuntimeError('Checkpoint is not a band confidence gate.')
            want_hash = ck.get('backbone_sha256')
            if not any(p.is_file() and sha256(p) == want_hash for p in (backbone, diagnostic)):
                raise RuntimeError('Confidence gate backbone SHA256 mismatch or backbone missing. '
                                   'Do NOT retrain the backbone in-place; restore its original checkpoint.')
            print('CHECKED: gate and spectral backbone SHA256 match.')

    if args.stage in ('all', 'region'):
        if topology.is_file() and geometry_checkpoint is None:
            raise RuntimeError('Topology checkpoint already exists but its Geometry source is missing. '
                               'Restore the Geometry checkpoint instead of silently training an incompatible new one.')
        if geometry_checkpoint is not None:
            print('SKIP geometry guidance: found', geometry_checkpoint)
        else:
            print('STAGE 3/4: geometry guidance')
            if not args.dry_run:
                norm_path = geometry_training_metadata(spec, base)
                print('Geometry normalization (TRAIN ONLY):', norm_path)
            cmd = [sys.executable, scripts[2], '--gpus', '2']
            launch(cmd, spec, extra_env={'TPSSR_GEOMETRY_TRAIN_STATS': str(norm_path)},
                   dry_run=args.dry_run)
        if not args.dry_run and not any(p.is_file() for p in geometry_candidates):
            raise RuntimeError('Geometry stage finished without an expected best_model.pt.')

        if topology.is_file():
            print('SKIP Topology Completion: found', topology)
        else:
            print('STAGE 4/4: topology completion with frozen geometry guidance')
            env = {
                'TPSSR_TOPOLOGY_EPOCHS': args.topology_epochs,
                'TPSSR_TOPOLOGY_MIN_EPOCHS': args.topology_min_epochs,
                'TPSSR_TOPOLOGY_PATIENCE': args.topology_patience,
                'TPSSR_TOPOLOGY_TRAIN_SAMPLES': args.topology_train_samples,
                'TPSSR_TOPOLOGY_VAL_SAMPLES': args.topology_val_samples,
            }
            cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node=2', scripts[3]]
            launch(cmd, spec, extra_env=env, dry_run=args.dry_run)
        if not args.dry_run and not topology.is_file():
            raise RuntimeError('Topology Completion did not produce best_model.pt.')

    print('\nDONE.' if not args.dry_run else '\nDRY-RUN PLAN FINISHED.')
    if not args.dry_run and args.stage == 'all':
        print('Inference: python test.py --preset ' + str(args.preset) + ' --gpus 2 '
              '--routing-mode tpssr --max-masks 200')


if __name__ == '__main__':
    main()
