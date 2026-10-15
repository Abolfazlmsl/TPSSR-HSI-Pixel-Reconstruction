#!/usr/bin/env python3
"""Check trained weights, test cubes and masks without running model inference."""
from __future__ import annotations
import argparse
from pathlib import Path
import os
from experiment_config import resolve_experiment

P=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--preset',default='swir_8x2')
a=p.parse_args()
spec=resolve_experiment(a)
model=P/'trained_models'/spec.experiment_name/'texture_restoration_router_rf_v2'
items={
 'Hybrid V1 backbone':model/'spectral_architecture_compare_v1/hybrid/diagnostic_best_val.pt',
 'Hybrid V2 band-wise gate':model/'spectral_hybrid_gate_v2/band/best_model.pt',
}
for core in spec.tests:
 items[f'{core} wavelengths'] = P/'processed_data'/core/'wavelengths_nm.npy'
 items[f'{core} reflectance'] = P/'processed_data'/core/'calibrated_reflectance.npy'
 items[f'{core} invalid mask'] = P/'processed_data'/core/'invalid_mask.npy'
 items[f'{core} topology model'] = model/f'new_v13_topology_first_{core.lower()}'/'best_model.pt'
 # V27 can have either of two supported checkpoint names.
 options=[model/f'region_v27_original_{core.lower()}_dev_2gpu'/'best_model.pt',
          model/f'region_v27_learned_correspondence_{core.lower()}_dev_2gpu'/'best_model.pt']
 if not any(x.is_file() for x in options):
  items[f'{core} V27 Stage-A (one option)']=options[0]
from collections import Counter
failed=[]
print('Experiment:',spec.experiment_name)
for name,path in items.items():
 found=path.is_file()
 print(('OK     ' if found else 'MISSING'),name,'->',path.relative_to(P))
 if not found:failed.append(name)
for core in spec.tests:
 m=P/'results'/spec.experiment_name/core/'masks'
 # Mask locations depend on the original config; list only if present.
print('\nModel/data checks:',len(items)-len(failed),'OK,',len(failed),'missing')
print('NOTE: this lightweight check does not validate checkpoint SHA256 or actual mask paths.')
if failed:raise SystemExit(2)
