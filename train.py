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
EXPECTED_TRAIN = ('SWIR1', 'SWIR2', 'SWIR3', 'SWIR4', 'SWIR6', 'SWIR7', 'SWIR8', 'SWIR9')
EXPECTED_TEST = ('SWIR5', 'SWIR10')


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
    ap.add_argument('--stage', choices=('all', 'hybrid', 'region'), default='all',
                    help='Optionally restrict the training family; default trains all missing components')
    ap.add_argument('--hybrid-epochs', type=int, default=5)
    ap.add_argument('--gate-epochs', type=int, default=8)
    ap.add_argument('--topology-epochs', type=int, default=22)
    ap.add_argument('--topology-min-epochs', type=int, default=14)
    ap.add_argument('--topology-patience', type=int, default=5)
    ap.add_argument('--topology-train-samples', type=int, default=4200)
    ap.add_argument('--topology-val-samples', type=int, default=900)
    args = ap.parse_args()
    spec = resolve_experiment(args)
    if tuple(spec.train) != EXPECTED_TRAIN or tuple(spec.tests) != EXPECTED_TEST:
        ap.error('The current frozen Spectral trainer/runtime supports only --preset swir_8x2. '
                 'Cross-region and VNIR require retraining/adapting their spectral trainers, not merely changing tags.')
    if min(args.hybrid_epochs, args.gate_epochs, args.topology_epochs,
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
    print('One shared model; SWIR5 and SWIR10 are never used for training.')
    print('Checkpoint tag:', tag, '(folder label only, not training data)')
    print('The inference policy needs NO trainable router.\n')

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

    if args.stage in ('all', 'hybrid'):
        if backbone.exists() or diagnostic.exists():
            chosen = backbone if backbone.exists() else diagnostic
            print('SKIP spectral backbone: found', chosen)
        else:
            print('STAGE 1/4: spectral backbone')
            cmd = [sys.executable, '-m', 'training.spectral_backbone', '--preset', 'swir_8x2', '--gpus', '2',
                   '--epochs', str(args.hybrid_epochs)]
            if (hybrid_root / 'last_model.pt').exists():
                cmd.append('--resume')
            launch(cmd, spec, dry_run=args.dry_run)
        if not args.dry_run and not (backbone.exists() or diagnostic.exists()):
            raise RuntimeError('Spectral training did not produce an accepted or diagnostic checkpoint.')

        if gate.exists():
            print('SKIP band confidence gate: found', gate)
        else:
            print('STAGE 2/4: confidence gate with a frozen backbone')
            cmd = [sys.executable, '-m', 'training.spectral_gate', '--preset', 'swir_8x2', '--gpus', '2',
                   '--epochs', str(args.gate_epochs)]
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
        print('Inference: python test.py --preset swir_8x2 --gpus 2 '
              '--routing-mode tpssr --max-masks 200')


if __name__ == '__main__':
    main()
