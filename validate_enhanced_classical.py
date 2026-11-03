#!/usr/bin/env python3
"""Quick validation of enhanced PCA/RF on TRAINING acquisitions' validation blocks.

Samples one intentionally masked voxel per scenario/severity/case, with a real
spatial corruption footprint in a 49x49 context. Does NOT touch held-out tests.
The output is diagnostic, not a substitute for final 200-mask evaluation.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import os
import sys
import numpy as np
import joblib
from experiment_config import add_experiment_args, resolve_experiment
from _launcher_common import CORE, env_for


def main():
    p=argparse.ArgumentParser(description=__doc__)
    add_experiment_args(p)
    p.add_argument('--cases-per-combination',type=int,default=4)
    a=p.parse_args()
    if a.cases_per_combination < 1:
        raise ValueError('Need at least one validation case')
    spec=resolve_experiment(a)
    os.environ.update(env_for(spec,spec.active_test))
    sys.path.insert(0,str(CORE))
    import config
    from src.evaluation_utils import load_global_training_data
    from src.training_corruptions import SCENARIOS,generate_training_mask
    from src.enhanced_classical import reconstruct_spatial_pca,predict_multiscale_rf
    from reconstruction_methods.pca.test import reconstruct_pca
    from reconstruction_methods.random_forest.test import predict

    _,validation,_=load_global_training_data()
    models={}
    for name,path in (
        ('pca', config.MODELS_DIR/'pca'/'pca_model.joblib'),
        ('pca_spatial',config.MODELS_DIR/'pca'/'pca_spatial_model.joblib'),
        ('random_forest',config.MODELS_DIR/'random_forest'/'random_forest_model.joblib'),
        ('random_forest_multiscale',config.MODELS_DIR/'random_forest'/'random_forest_multiscale_model.joblib')):
        if path.is_file():
            item=joblib.load(path)
            if list(item.get('training_datasets',[])) != list(config.TRAIN_DATASET_FOLDERS):
                print(f'WARNING: {name} training acquisitions do not match this experiment. Ignoring.',flush=True)
                continue
            models[name]=item
    if not {'pca_spatial','random_forest_multiscale'}.issubset(models):
        raise FileNotFoundError('Train both enhanced methods before validation')
    rng=np.random.default_rng(config.SEED+17592)
    records=[]
    for scenario in SCENARIOS:
        for severity in range(1,6):
            for repeat in range(a.cases_per_combination):
                cube=validation[int(rng.integers(len(validation)))]
                H,W,B=cube.shape
                if H<49 or W<49 or B<33:
                    raise RuntimeError('Validation cube too small for multiscale features')
                sample=None
                for attempt in range(100):
                    y=int(rng.integers(24,H-24))
                    x=int(rng.integers(24,W-24))
                    b=int(rng.integers(16,B-16))
                    value=float(cube[y,x,b])
                    if np.isfinite(value):
                        sample=(y,x,b,value)
                        break
                if sample is None:
                    continue
                y,x,b,truth=sample
                patch=np.asarray(cube[y-24:y+25,x-24:x+25,:],np.float32)
                mask,_,_=generate_training_mask(patch.shape,rng,scenario=scenario,
                              severity=severity,center=(24,24),band_center=b)
                if not mask[24,24,b]:
                    raise RuntimeError('Center target not masked')
                coords=np.array([[24,24,b]],dtype=np.int64)
                for method,item in models.items():
                    if method=='pca':
                        result=reconstruct_pca(patch,mask,coords,item['model'],item['band_mean'])
                    elif method=='pca_spatial':
                        result=reconstruct_spatial_pca(patch,mask,coords,item['model'],item['band_mean'])
                    elif method=='random_forest':
                        result=predict(item['model'],item['band_mean'],patch,mask,coords)
                    else:
                        result=predict_multiscale_rf(item['model'],item['band_mean'],patch,mask,coords)
                    err=float(result[0])-truth
                    records.append(dict(scenario=scenario,severity=severity,repeat=repeat,
                              method=method,truth=truth,prediction=float(result[0]),
                              abs_error=abs(err),squared_error=err*err))
            print(f'[validation] {scenario} S{severity} complete',flush=True)
    out=config.MODELS_DIR/'classical_validation'
    out.mkdir(parents=True,exist_ok=True)
    path=out/'validation_center_voxel.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print('Center-voxel validation (not the final test results):')
    for method in models:
        vals=[r['squared_error'] for r in records if r['method']==method]
        print(f'  {method}: n={len(vals)}, RMSE={np.sqrt(np.mean(vals)):.6f}')
    print('Saved:',path)

if __name__=='__main__':
    main()
