#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TPSSR spectral-only Transformer experiment.

No neighboring pixels are read by the model or the dataset.  Train on six of
TPSSR's eight training acquisitions; hold out two ENTIRE training acquisitions
(one per region) for cross-core validation; never read SWIR5/SWIR10.

Standalone addition to project root (beside test.py), not a replacement.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset

ROOT = Path(__file__).resolve().parent
CORE = ROOT / 'core'
for path in (str(ROOT), str(CORE)):
    if path not in sys.path: sys.path.insert(0, path)

from experiment_config import add_experiment_args, resolve_experiment, describe
from _launcher_common import env_for

FAMILIES = (
    'isolated_voxels', 'single_detector_line', 'multiple_detector_lines',
    'partial_vertical_lines', 'spatial_patch', 'consecutive_spectral_bands',
    'spatio_spectral_block', 'band_wide_saturation', 'mixed_saturation',
)
# All-spectral missing pixels are deliberately ineligible for a within-pixel
# model. They MUST be handled by the existing Region expert.


def band_mask(rng: np.random.Generator, bands: int, family: str, severity: int):
    """One-pixel band-mask approximations of TPSSR's synthetic corruption families.

    This is a TRAINING generator only; it never samples the held-out test masks.
    Spatial patch geometry is irrelevant once restricted to a single spectrum.
    """
    sev = int(severity)
    m = np.zeros(bands, dtype=np.bool_)

    def block(width):
        width = max(1, min(int(width), bands))
        c = int(rng.integers(0, bands))
        start = max(0, min(c - width // 2, bands-width))
        m[start:start+width] = True

    if family == 'isolated_voxels':
        n = [1,2,4,8,16][sev-1]
        m[rng.choice(bands, min(n, bands), replace=False)] = True
    elif family == 'single_detector_line':
        block(1)
    elif family == 'multiple_detector_lines':
        n = [4,8,16,32,64][sev-1]
        m[rng.choice(bands, min(n, bands), replace=False)] = True
    elif family == 'partial_vertical_lines':
        n = [2,5,10,20,40][sev-1]
        m[rng.choice(bands, min(n, bands), replace=False)] = True
    elif family == 'spatial_patch':
        block([1,3,7,15,31][sev-1])
        if sev >= 3 and rng.random() < .3: block([1,3,7,15,31][sev-1])
    elif family == 'consecutive_spectral_bands':
        block([3,7,15,31,63][sev-1])
    elif family == 'spatio_spectral_block':
        block([5,11,21,41,63][sev-1])
    elif family == 'band_wide_saturation':
        block([1,3,7,15,31][sev-1])
    elif family == 'mixed_saturation':
        block([3,7,15,31,63][sev-1])
        block([1,3,7,15,31][sev-1])
        n = min([2,4,8,16,32][sev-1], bands)
        m[rng.choice(bands, n, replace=False)] = True
    else:
        raise ValueError(f'Unknown family {family}')
    return m


def interpolate_raw(raw, missing, finite):
    obs = (~missing) & finite
    idx = np.flatnonzero(obs)
    if len(idx) < 2:
        return None
    return np.interp(np.arange(len(raw)), idx, raw[idx]).astype(np.float32)


class SpectraDataset(Dataset):
    def __init__(self, names, cubes, invalids, bands, mean, std, wave, count, seed,
                 training, ncores_for_balance):
        self.names = tuple(names)
        self.cubes = cubes
        self.invalids = invalids
        self.bands = int(bands)
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)
        self.wave = (np.asarray(wave, np.float32) - wave[0]) / max(float(wave[-1]-wave[0]), 1e-6)
        self.count = int(count)
        self.seed = int(seed)
        self.training = bool(training)
        self.epoch = 0
        self.ncores_for_balance = int(ncores_for_balance)

    def set_epoch(self, e): self.epoch = int(e)
    def __len__(self): return self.count

    def __getitem__(self, idx):
        idx = int(idx)
        rng = np.random.default_rng(self.seed + idx*10007 + (self.epoch if self.training else 0)*10000019)
        if self.training:
            core_idx = idx % len(self.cubes)
            family_idx = int(rng.integers(len(FAMILIES)))
            severity = int(rng.integers(1,6))
        else:
            core_idx = idx % len(self.cubes)
            # deterministic, scenario/severity balanced within each validation core
            job = idx // len(self.cubes)
            family_idx = job % len(FAMILIES)
            severity = (job // len(FAMILIES)) % 5 + 1
        cube = self.cubes[core_idx]
        inv = self.invalids[core_idx]
        h,w,b = cube.shape
        if b != self.bands: raise RuntimeError('Band count mismatch')
        for _ in range(100):
            y = int(rng.integers(0,h)); x = int(rng.integers(0,w))
            raw = np.array(cube[y,x,:], dtype=np.float32)
            bad = np.asarray(inv[y,x,:], dtype=bool) | ~np.isfinite(raw)
            finite = ~bad
            if finite.sum() < max(3, int(.90*b)): continue
            m = band_mask(rng,b,FAMILIES[family_idx],severity) & finite
            if not m.any(): continue
            raw_anchor = interpolate_raw(raw,m,finite)
            if raw_anchor is None: continue
            observed = finite & ~m
            if observed.sum() < 2: continue
            target = np.where(finite, (raw-self.mean)/self.std, 0.).astype(np.float32)
            anch = (raw_anchor-self.mean)/self.std
            observed_values = np.where(observed, target, 0.).astype(np.float32)
            # IMPORTANT: hidden raw values NEVER enter any model input.
            feats = np.stack((observed_values, anch, m.astype(np.float32), self.wave), axis=-1).astype(np.float32)
            return {
                'feat': torch.from_numpy(feats),
                'target': torch.from_numpy(target),
                'anchor': torch.from_numpy(anch),
                'missing': torch.from_numpy(m.astype(np.float32)),
                'observed': torch.from_numpy(observed.astype(np.float32)),
                'core': torch.tensor(core_idx, dtype=torch.long),
                'family': torch.tensor(family_idx, dtype=torch.long),
                'severity': torch.tensor(severity-1, dtype=torch.long),
            }
        raise RuntimeError('Could not sample spectrum with sufficient finite bands; inspect invalid masks')


class SpectralTransformer(nn.Module):
    """Per-pixel masked Transformer: [N,B,4] -> reconstructed [N,B]."""
    def __init__(self, bands, width=64, layers=2, heads=4, dropout=.08):
        super().__init__()
        if width % heads: raise ValueError('width must be divisible by heads')
        self.bands = int(bands)
        self.input = nn.Sequential(nn.Linear(4,width), nn.LayerNorm(width))
        self.position = nn.Parameter(torch.randn(1,bands,width)*.01)
        layer = nn.TransformerEncoderLayer(width,heads,dim_feedforward=width*3,
                dropout=dropout,activation='gelu',batch_first=True,norm_first=True)
        self.encoder = nn.TransformerEncoder(layer,num_layers=layers,enable_nested_tensor=False)
        self.head = nn.Linear(width,2)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias,0.)

    def forward(self, feat):
        if feat.ndim != 3 or feat.shape[1] != self.bands or feat.shape[2] != 4:
            raise RuntimeError(f'Expected [N,{self.bands},4], got {tuple(feat.shape)}')
        x = self.encoder(self.input(feat)+self.position)
        out = self.head(x)
        # Small, learned residual; zero-residual initialization exactly equals raw interpolation.
        delta = .5 * torch.tanh(out[...,0]) * torch.sigmoid(out[...,1])
        return feat[...,1] + feat[...,2] * delta


def train_loss(pred, batch):
    m = batch['missing']
    e2 = (pred-batch['target']).square()
    denom = m.sum().clamp_min(1.)
    mse = (e2*m).sum()/denom
    anchor_err = (batch['anchor']-batch['target']).abs().detach()
    reg = (F.relu((pred-batch['target']).abs()-anchor_err).square()*m).sum()/denom
    preserv = ((pred-batch['anchor']).square()*m).sum()/denom
    return mse + .35*reg + .015*preserv


def checkpoint(model,args,epoch,metrics,mean,std,wave,train_cores,val_cores,optimizer=None):
    core = model.module if isinstance(model,DDP) else model
    payload=dict(architecture='spectral_only_transformer_v1',epoch=int(epoch),
        model_state_dict={k:v.detach().cpu() for k,v in core.state_dict().items()},
        model_params=dict(bands=core.bands,width=args.width,layers=args.layers,heads=args.heads),
        band_mean=np.asarray(mean,np.float32),band_std=np.asarray(std,np.float32),
        wavelengths_nm=np.asarray(wave,np.float32),train_cores=list(train_cores),
        validation_cores=list(val_cores),validation=metrics,
        policy='Same-pixel only; ineligible if <2 observed bands; Region fallback required.')
    if optimizer is not None: payload['optimizer_state_dict']=optimizer.state_dict()
    return payload


def distributed_sum(t):
    if dist.is_initialized():
        dist.all_reduce(t,op=dist.ReduceOp.SUM)
    return t


@torch.no_grad()
def validate(model,loader,device,ncores,std):
    model.eval()
    # Row indexed by (core * len(families)*5 + family*5 + severity) -> 
    # [model sumsq, raw sumsq, nvox, model abs, raw abs]
    sums = torch.zeros((ncores*len(FAMILIES)*5,5),device=device,dtype=torch.float64)
    for batch in loader:
        batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
        pred=model(batch['feat'])
        m=batch['missing'].double()
        d=(pred.double()-batch['target'].double())*std.view(1,-1)
        r=(batch['anchor'].double()-batch['target'].double())*std.view(1,-1)
        cols=torch.stack(((d*d*m).sum(1),(r*r*m).sum(1),m.sum(1),
                         (d.abs()*m).sum(1),(r.abs()*m).sum(1)),dim=1)
        idx=(batch['core']*(len(FAMILIES)*5)+batch['family']*5+batch['severity']).long()
        sums.index_add_(0,idx,cols)
    distributed_sum(sums)
    arr=sums.cpu().numpy()
    by_core=[]
    for ci in range(ncores):
        v=arr[ci*len(FAMILIES)*5:(ci+1)*len(FAMILIES)*5].sum(axis=0)
        by_core.append(dict(model_rmse=math.sqrt(v[0]/max(v[2],1)),
                            raw_rmse=math.sqrt(v[1]/max(v[2],1)),
                            count=int(v[2]),model_mae=float(v[3]/max(v[2],1)),
                            raw_mae=float(v[4]/max(v[2],1))))
    full=arr.sum(axis=0)
    overall=dict(model_rmse=math.sqrt(full[0]/max(full[2],1)),
        raw_rmse=math.sqrt(full[1]/max(full[2],1)),count=int(full[2]))
    return overall,by_core,arr


def make_stats(cubes,invalids,bands,seed=4123):
    rng=np.random.default_rng(seed)
    sums=np.zeros(bands,np.float64); sumsq=np.zeros(bands,np.float64)
    count=np.zeros(bands,np.float64)
    for cube,invalid in zip(cubes,invalids):
        h,w,b=cube.shape
        for _ in range(3500):
            y=int(rng.integers(h)); x=int(rng.integers(w))
            v=np.array(cube[y,x,:],dtype=np.float64)
            bad=np.asarray(invalid[y,x,:],dtype=bool)|~np.isfinite(v)
            v=np.where(bad,0.,v)
            sums+=v; sumsq+=v*v; count+=(~bad)
    if np.any(count < 100): raise RuntimeError('Some bands have too few finite training values')
    mean=sums/count; var=np.maximum(sumsq/count-mean*mean,1e-10)
    return mean.astype(np.float32),np.sqrt(var).astype(np.float32)


def load_core_data(spec,val_cores):
    import config
    from src.evaluation_utils import load_dataset,validate_wavelength_grid
    training=tuple(n for n in spec.train if n not in val_cores)
    if len(training)<2 or len(val_cores)<1: raise ValueError('Need >=2 training cores and held-out validation cores')
    cubes={}; invalids={}; wave=None; first=None
    for name in training+tuple(val_cores):
        cube, invalid, _split, candidate=load_dataset(name)
        if wave is None:
            wave=np.asarray(candidate,np.float32); first=name
        else: validate_wavelength_grid(wave,candidate,first,name)
        cubes[name]=cube; invalids[name]=invalid
    train_c=[cubes[n] for n in training]; train_i=[invalids[n] for n in training]
    val_c=[cubes[n] for n in val_cores]; val_i=[invalids[n] for n in val_cores]
    return training,train_c,train_i,val_c,val_i,wave


def run_worker(args):
    import config
    rank=int(os.environ.get('RANK',0)); local=int(os.environ.get('LOCAL_RANK',0))
    world=int(os.environ.get('WORLD_SIZE',1))
    if world != args.gpus: raise RuntimeError('torchrun rank count does not match --gpus')
    if not torch.cuda.is_available() or torch.cuda.device_count()<world:
        raise RuntimeError(f'Need {world} CUDA GPUs')
    torch.cuda.set_device(local)
    device=torch.device('cuda',local)
    if world>1: dist.init_process_group('nccl')
    try:
        random.seed(9011+rank);np.random.seed(9011+rank);torch.manual_seed(9011+rank)
        spec=resolve_experiment(args)
        val_cores=tuple(n.upper() for n in args.val_cores)
        if set(val_cores)&set(spec.tests): raise RuntimeError('Held-out TEST core selected as validation!')
        if not set(val_cores).issubset(set(spec.train)): raise RuntimeError('Validation cores must be drawn from training acquisitions')
        training,tr_c,tr_i,va_c,va_i,wave=load_core_data(spec,val_cores)
        bands=len(wave)
        output=Path(config.MODELS_DIR)/'texture_restoration_router_rf_v2'/'spectral_transformer_same_pixel_v1'
        if rank==0: output.mkdir(parents=True,exist_ok=True)
        if world>1: dist.barrier()
        stats_path=output/'TRAIN_ONLY_STATS.npz'
        if rank==0 and not stats_path.exists():
            mean,std=make_stats(tr_c,tr_i,bands)
            np.savez(stats_path,mean=mean,std=std,wavelengths=wave,train_cores=np.array(training))
        if world>1: dist.barrier()
        with np.load(stats_path,allow_pickle=False) as stats:
            mean=np.asarray(stats['mean'],np.float32); std=np.asarray(stats['std'],np.float32)
            train_saved=[str(x) for x in stats['train_cores']]
            if tuple(train_saved)!=tuple(training):
                raise RuntimeError('Existing stats file belongs to another train split; do not reuse')
            if not np.allclose(stats['wavelengths'],wave,rtol=0,atol=.5):
                raise RuntimeError('Existing stats wavelength grid mismatch')
        train_ds=SpectraDataset(training,tr_c,tr_i,bands,mean,std,wave,args.train_samples,4200,True,len(training))
        val_ds=SpectraDataset(val_cores,va_c,va_i,bands,mean,std,wave,args.val_samples,7770,False,len(val_cores))
        sampler=DistributedSampler(train_ds,num_replicas=world,rank=rank,shuffle=True,seed=877) if world>1 else None
        train_loader=DataLoader(train_ds,batch_size=args.batch_size,sampler=sampler,shuffle=(sampler is None),
                                num_workers=args.workers,pin_memory=True,drop_last=True)
        val_subset=Subset(val_ds,list(range(rank,len(val_ds),world)))
        val_loader=DataLoader(val_subset,batch_size=args.batch_size,num_workers=args.workers,pin_memory=True)
        model=SpectralTransformer(bands,args.width,args.layers,args.heads).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
        scaler=torch.amp.GradScaler('cuda',enabled=args.amp)
        last_path=output/'last_model.pt'; best_path=output/'best_model.pt'
        history_path=output/'training_history.csv'
        start=1; best=float('inf')
        history=[]
        if args.resume and last_path.is_file():
            saved=torch.load(last_path,map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model_state_dict'],strict=True)
            if 'optimizer_state_dict' in saved: optimizer.load_state_dict(saved['optimizer_state_dict'])
            start=int(saved['epoch'])+1
            if best_path.is_file():
                prev=torch.load(best_path,map_location='cpu',weights_only=False)
                best=float(prev['validation']['overall']['model_rmse'])
            if rank==0 and history_path.is_file():
                with open(history_path,newline='',encoding='utf8') as f: history=list(csv.DictReader(f))
        elif last_path.exists():
            raise RuntimeError('Existing training found. Use --resume; original training will not be overwritten.')
        if world>1: dist.barrier()
        if world>1: model=DDP(model,device_ids=[local],output_device=local)
        for epoch in range(start,args.epochs+1):
            train_ds.set_epoch(epoch)
            if sampler is not None:sampler.set_epoch(epoch)
            model.train(); epoch_loss=0.; nstep=0
            for batch in train_loader:
                batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',enabled=args.amp):
                    pred=model(batch['feat']); loss=train_loss(pred,batch)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer);scaler.update()
                epoch_loss+=float(loss.detach().item());nstep+=1
            overall,bycore,groups=validate(model,val_loader,device,len(val_cores),torch.as_tensor(std,device=device,dtype=torch.float64))
            score=overall['model_rmse']
            safe=(score < overall['raw_rmse'] and
                  all(p['model_rmse'] <= p['raw_rmse']*1.01 for p in bycore))
            accept=safe and score<best-1e-8
            if accept:best=score
            if rank==0:
                vals=dict(overall=overall,by_core={val_cores[i]:v for i,v in enumerate(bycore)},
                          accepted=bool(accept),safe=bool(safe),best=float(best))
                core=model.module if isinstance(model,DDP) else model
                record=dict(epoch=epoch,train_loss=epoch_loss/max(nstep,1),
                            val_rmse=score,raw_rmse=overall['raw_rmse'],
                            val_core_1_rmse=bycore[0]['model_rmse'],
                            raw_core_1_rmse=bycore[0]['raw_rmse'],
                            val_core_2_rmse=bycore[1]['model_rmse'] if len(bycore)>1 else float('nan'),
                            raw_core_2_rmse=bycore[1]['raw_rmse'] if len(bycore)>1 else float('nan'),
                            accepted=int(accept),safety=int(safe))
                history.append(record)
                torch.save(checkpoint(model,args,epoch,vals,mean,std,wave,training,val_cores,optimizer),last_path)
                if accept:torch.save(checkpoint(model,args,epoch,vals,mean,std,wave,training,val_cores),best_path)
                with open(history_path,'w',newline='',encoding='utf8') as f:
                    wr=csv.DictWriter(f,fieldnames=list(record));wr.writeheader();wr.writerows(history)
                with open(output/'scenario_severity_validation.csv','w',newline='',encoding='utf8') as f:
                    wr=csv.writer(f)
                    wr.writerow(['validation_core','scenario','severity','n_voxels','transformer_rmse','spectral_raw_rmse'])
                    for ci,name in enumerate(val_cores):
                        for fi,fam in enumerate(FAMILIES):
                            for s in range(5):
                                z=groups[ci*len(FAMILIES)*5+fi*5+s]
                                wr.writerow([name,fam,s+1,int(z[2]),math.sqrt(z[0]/max(z[2],1)),
                                             math.sqrt(z[1]/max(z[2],1))])
                summary=(f'SPECTRAL-ONLY TRANSFORMER: epoch {epoch}/{args.epochs}\n'
                         f'Train cores: {", ".join(training)}\n'
                         f'VALIDATION CORES (fully held out): {", ".join(val_cores)}\n'
                         f'No SWIR5/SWIR10 was used. No spatial input.\n'
                         f'Validation REFLECTANCE RMSE: transformer={score:.7f} raw_spectral={overall["raw_rmse"]:.7f}\n'
                         + '\n'.join(f'{val_cores[i]}: transformer={v["model_rmse"]:.7f} raw={v["raw_rmse"]:.7f}'
                                     for i,v in enumerate(bycore))
                         + f'\nAcceptance of current epoch: {accept}; best accepted RMSE: {best}\n'
                         + f'Best checkpoint exists: {best_path.exists()}\n'
                         + 'A missing best_model.pt means no model passed the baseline safety checks.\n'
                         + 'Whole-spectrum missing is INELIGIBLE; preserve the Region expert.\n'
                         + 'Do not tune using SWIR5/SWIR10 test results.\n')
                (output/'TRANSFORMER_SUMMARY.txt').write_text(summary,encoding='utf8')
                print(f'Epoch {epoch}: val={score:.6f}, raw={overall["raw_rmse"]:.6f}, accepted={accept}',flush=True)
            if world>1:dist.barrier()
        if rank==0:print(f'Outputs: {output}',flush=True)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


def smoke_test():
    rng=np.random.default_rng(123)
    bands=21; mean=np.ones(bands,np.float32)*.2;std=np.ones(bands,np.float32)*.03
    # Every model input depends ONLY on observed values, mask, wavelength and raw interpolation.
    v=np.asarray(.2+.04*np.sin(np.arange(bands)*.3),np.float32)
    m=band_mask(rng,bands,'spatio_spectral_block',2)
    m[4:9]=True
    m[0]=False; m[-1]=False
    anchor=interpolate_raw(v,m,np.ones(bands,bool))
    assert anchor is not None
    feats=np.stack((np.where(m,0.,(v-mean)/std),(anchor-mean)/std,
                    m.astype(np.float32),np.linspace(0,1,bands)),axis=-1).astype(np.float32)
    net=SpectralTransformer(bands,width=32,layers=1,heads=4,dropout=0.)
    net.train()
    f=torch.tensor(feats).unsqueeze(0)
    t=torch.tensor((v-mean)/std).unsqueeze(0)
    batch={'target':t,'missing':torch.tensor(m.astype(np.float32)).unsqueeze(0),
           'anchor':torch.tensor((anchor-mean)/std).unsqueeze(0)}
    p=net(f);loss=train_loss(p,batch)
    loss.backward()
    assert p.shape==(1,bands) and torch.isfinite(loss)
    assert net.head.weight.grad is not None
    v2=v.copy();v2[m]=1.e6
    feat2=np.stack((np.where(m,0.,(v2-mean)/std),(anchor-mean)/std,
                    m.astype(np.float32),np.linspace(0,1,bands)),axis=-1).astype(np.float32)
    net.eval()
    with torch.no_grad():
        p1=net(torch.tensor(feats).unsqueeze(0));p2=net(torch.tensor(feat2).unsqueeze(0))
    assert torch.allclose(p1,p2,atol=1e-6),'Hidden GT influenced prediction'
    assert np.allclose(p1[0][~m].numpy(),(v[~m]-mean[~m])/std[~m],atol=1e-5)
    print('SMOKE TEST PASSED: Transformer, loss, gradients, observed-band preservation, no masked GT leakage')


def main():
    p=argparse.ArgumentParser(description='Core-heldout spectral-only Transformer for TPSSR')
    add_experiment_args(p)
    p.add_argument('--gpus',type=int,choices=[1,2],default=2)
    p.add_argument('--batch-size',type=int,default=16,help='Per GPU')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--train-samples',type=int,default=4500)
    p.add_argument('--val-samples',type=int,default=900)
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--amp',action='store_true',help='Mixed precision, optional')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--val-cores',nargs='+',default=['SWIR4','SWIR9'],
                   help='Entire acquisitions held out from the eight train cores')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--smoke-test',action='store_true')
    args=p.parse_args()
    if args.smoke_test: smoke_test();return
    # torchrun child: train on its assigned rank, NEVER launch torchrun again.
    if args.worker:
        run_worker(args)
        return
    # Guard against accidental recursive distributed launching.
    if 'LOCAL_RANK' in os.environ or 'RANK' in os.environ:
        raise RuntimeError('Distributed worker process missing --worker; recursive launch prevented')
    spec=resolve_experiment(args)
    val_cores=tuple(n.upper() for n in args.val_cores)
    if len(val_cores)!=len(set(val_cores)) or len(val_cores)!=2:
        raise RuntimeError('Select exactly two distinct core-heldout validation acquisitions')
    if len(spec.train)!=8 or len(spec.tests)!=2:
        raise RuntimeError('This version requires exactly the 8-train 2-test preset')
    if not set(val_cores).issubset(set(spec.train)) or set(val_cores)&set(spec.tests):
        raise RuntimeError('Validation cores must be training-acquisition IDs, never test cores')
    if args.batch_size<1 or args.epochs<1 or args.train_samples<2*args.gpus or args.val_samples<10:
        raise RuntimeError('Invalid training settings')
    if torch.cuda.device_count()<args.gpus:raise RuntimeError('Requested CUDA GPUs unavailable')
    print('TRAIN SPECTRAL TRANSFORMER: 6 training cores / 2 independent validation cores',flush=True)
    print('No Spatial features. No SWIR5/SWIR10 accessed. Original TPSSR unchanged.',flush=True)
    env=env_for(spec,spec.primary_test)
    env.setdefault('OMP_NUM_THREADS','1')
    cli=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={args.gpus}',
         str(Path(__file__).resolve()),'--worker','--gpus',str(args.gpus),
         '--epochs',str(args.epochs),'--batch-size',str(args.batch_size),
         '--train-samples',str(args.train_samples),'--val-samples',str(args.val_samples),
         '--width',str(args.width),'--layers',str(args.layers),'--heads',str(args.heads),
         '--workers',str(args.workers),'--lr',str(args.lr),'--val-cores',*val_cores]
    if args.amp:cli.append('--amp')
    if args.resume:cli.append('--resume')
    subprocess.run(cli,cwd=CORE,env=env,check=True)


if __name__=='__main__': main()
