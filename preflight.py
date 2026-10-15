#!/usr/bin/env python3
"""Verify the checkpoint and test-data layout used by TPSSR V2.3 inference.

SWIR5 is the stable checkpoint tag for the shared SWIR5/SWIR10 experiment.
The tests DO NOT need one separately trained region model per acquisition.
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
    a = parser.parse_args()
    spec = resolve_experiment(a)
    if tuple(spec.tests) != ('SWIR5', 'SWIR10') or len(spec.train) != 8:
        parser.error('This specific Hybrid gate checker supports only the main swir_8x2 split.')
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
    for name in spec.tests:
        for short, file in (('reflectance', 'calibrated_reflectance.npy'),
                            ('invalid mask', 'invalid_mask.npy'),
                            ('wavelength grid', 'wavelengths_nm.npy')):
            groups[f'{name} {short}'] = [ROOT / 'processed_data' / name / file]
    missing = []
    print('TPSSR V2.3 PRE-FLIGHT\nExperiment:', spec.experiment_name,
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
        elif not any(p.exists() and sha256(p) == g.get('backbone_sha256') for p in backbone_options):
            missing.append('Spectral backbone does not match gate SHA256')
        else:
            print('OK      SHA256 of frozen spectral backbone')
    print('\n', len(missing), 'issue(s).', sep='')
    if missing:
        sys.exit(2)

if __name__ == '__main__':
    main()
