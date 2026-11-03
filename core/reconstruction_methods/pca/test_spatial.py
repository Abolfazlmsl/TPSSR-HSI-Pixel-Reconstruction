"""Evaluate spatially initialized PCA on the original fixed benchmark masks."""
from __future__ import annotations
from pathlib import Path
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'config.py').exists())
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from tqdm import tqdm
import config
from src.enhanced_classical import reconstruct_spatial_pca
from src.evaluation_utils import (load_dataset,split_blocks,list_mask_files,load_test_mask,
    select_eval_coords,regression_metrics,spectral_angle_metrics,save_method_metrics,
    benchmark_case_index, experiment_progress)


def main():
    checkpoint = config.MODELS_DIR/'pca'/'pca_spatial_model.joblib'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Run the spatial PCA training first: {checkpoint}')
    item = joblib.load(checkpoint)
    cube, _, split, _ = load_dataset()
    _,_,test = split_blocks(cube,split)
    rows=[]
    files=list_mask_files()
    for index, path in enumerate(tqdm(files,desc='Spatial PCA masks'), start=1):
        mask, meta = load_test_mask(path)
        progress,total=experiment_progress(index,len(files))
        print(f'[PCA spatial] {progress}/{total}: {meta["scenario"]} S{meta["severity"]} R{meta["repeat"]}',flush=True)
        coords=select_eval_coords(mask,config.SEED+benchmark_case_index(path))
        if not len(coords):
            continue
        pred=reconstruct_spatial_pca(test, mask, coords, item['model'], item['band_mean'],
                                     iterations=int(item.get('iterations',10)))
        truth=np.asarray(test)[coords[:,0], coords[:,1], coords[:,2]]
        rows.append(dict(scenario=meta['scenario'],severity=meta['severity'],
            repeat=meta['repeat'],method='pca_spatial',
            **regression_metrics(truth,pred),**spectral_angle_metrics(test,coords,pred)))
    print('Saved:',save_method_metrics('pca_spatial',rows),flush=True)

if __name__=='__main__':
    main()
