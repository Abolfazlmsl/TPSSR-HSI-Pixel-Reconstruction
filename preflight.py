#!/usr/bin/env python3
"""Check data and frozen checkpoint files before TPSSR inference.

The first held-out acquisition is only a checkpoint folder tag.
Each experiment shares a trained model across its held-out acquisitions.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
from experiment_config import add_experiment_args, resolve_experiment

ROOT = Path(__file__).resolve().parent

def sha256(path):
    dig = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            dig.update(block)
    return dig.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_args(parser)
    parser.add_argument("--data-only", action="store_true",
                        help="Verify all training/test arrays and test masks before GPU training")
    parser.add_argument("--router", choices=("learned_router3_aligned", "v23"),
                        default="learned_router3_aligned",
                        help="Checkpoint set to verify; defaults to the final learned router")
    a = parser.parse_args()
    spec = resolve_experiment(a)
    from experiment_config import spectral_validation_cores
    validation = spectral_validation_cores(spec)
    training = tuple(x for x in spec.train if x not in validation)
    if a.data_only:
        failures = []
        reference = None
        for name in (*spec.train, *spec.tests):
            data = ROOT / 'processed_data' / name
            required = ['calibrated_reflectance.npy', 'invalid_mask.npy', 'wavelengths_nm.npy']
            if any(not (data / item).is_file() for item in required):
                failures.append(f"{name}: missing processed arrays")
                continue
            try:
                cube = np.load(data / required[0], mmap_mode='r')
                invalid = np.load(data / required[1], mmap_mode='r')
                wave = np.load(data / required[2])
                if cube.ndim != 3 or cube.shape != invalid.shape or cube.shape[-1] != len(wave):
                    failures.append(f"{name}: inconsistent array shapes")
                if reference is None:
                    reference = wave
                elif len(reference) != len(wave) or not np.allclose(reference, wave, rtol=0, atol=0.5):
                    failures.append(f"{name}: incompatible wavelength grid")
                print(f"{name}: cube={cube.shape}; invalid={invalid.shape}; bands={len(wave)}")
            except Exception as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        for name in spec.tests:
            folder = ROOT / 'simulated_saturation' / spec.experiment_name / name / 'masks'
            count = sum(1 for _ in folder.glob('*.npz'))
            print(f"{name}: {count} simulated masks")
            if count < 40:
                failures.append(f"{name}: too few test masks (found {count})")
        print(f"DATA CHECK: {len(failures)} issue(s)")
        for failure in failures:
            print("ERROR:", failure)
        if failures:
            sys.exit(2)
        return
    root = ROOT / 'trained_models' / spec.experiment_name / 'tpssr'
    tag = spec.model_tag
    backbone_options = [
        root / 'spectral_backbone/hybrid/best_model.pt',
        root / 'spectral_backbone/hybrid/diagnostic_best_val.pt',
    ]
    gate = root / 'spectral_gate/band/best_model.pt'
    geometry_options = [
        root / f'geometry_{tag}/best_model.pt',
    ]
    topology = root / f'topology_{tag}/best_model.pt'
    groups = {
        'Spectral backbone checkpoint': backbone_options,
        'Band confidence gate checkpoint': [gate],
        'Geometry guidance checkpoint': geometry_options,
        'Topology completion checkpoint': [topology],
    }
    if a.router == 'learned_router3_aligned':
        groups['Learned neighbor fusion checkpoint'] = [
            root / 'neighbor_fusion' / 'best_model.pt']
        groups['Learned three-expert router checkpoint'] = [
            root / 'learned_router3_aligned' / 'best_model.joblib']
    for name in spec.tests:
        for short, file in (('reflectance', 'calibrated_reflectance.npy'),
                            ('invalid mask', 'invalid_mask.npy'),
                            ('wavelength grid', 'wavelengths_nm.npy')):
            groups[f'{name} {short}'] = [ROOT / 'processed_data' / name / file]
    missing = []
    print('TPSSR PRE-FLIGHT\nExperiment:', spec.experiment_name,
          '\nShared model tag:', tag, '\n')
    for label, options in groups.items():
        found = next((path for path in options if path.is_file()), None)
        print('OK     ' if found else 'MISSING', label,
              '->', found if found else ' OR '.join(str(x) for x in options))
        if found is None:
            missing.append(label)
    if not missing and gate.is_file():
        import torch
        g = torch.load(gate, map_location='cpu', weights_only=False)
        if g.get('architecture') != 'spectral_only_hybrid_confidence_gate_v2' or g.get('gate_variant') != 'band':
            missing.append('Invalid band confidence gate architecture')
        elif (tuple(g.get('run_config', {}).get('train_cores', ())) != training or
              tuple(g.get('run_config', {}).get('val_cores', ())) != validation):
            missing.append('Spectral checkpoint split metadata mismatch')
        elif not any(p.exists() and sha256(p) == g.get('backbone_sha256') for p in backbone_options):
            missing.append('Spectral backbone does not match gate SHA256')
        else:
            print('OK      SHA256 of frozen spectral backbone')
    print('\n', len(missing), 'issue(s).', sep='')
    if missing:
        sys.exit(2)

if __name__ == '__main__':
    main()
