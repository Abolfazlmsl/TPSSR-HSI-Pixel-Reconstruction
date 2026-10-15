#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TPSSR: matched spectral-only CNN vs CNN-Transformer architecture study.

One independent process per GPU (CNN GPU 0, Hybrid GPU 1); no NCCL or DDP.
This file is standalone ADDITION to project root; it imports the EXISTING
train_spectral_transformer.py's trusted data/mask helpers.
No access to SWIR5/SWIR10; cross-acquisition validation SWIR4/SWIR9.
Run: python train_spectral_arch_compare.py --preset swir_8x2 --gpus 2 ...
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
CORE = ROOT / 'core'
for p in (str(ROOT), str(CORE)):
    if p not in sys.path: sys.path.insert(0,p)

from experiment_config import add_experiment_args, resolve_experiment
from _launcher_common import env_for
try:
    from train_spectral_transformer import (
        SpectraDataset, load_core_data, make_stats, FAMILIES,
        band_mask, interpolate_raw
    )
except ImportError as exc:
    raise RuntimeError('Keep the existing train_spectral_transformer.py next to this file.') from exc

MODEL_NAMES = ('cnn', 'hybrid')


class SpectralResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float = .05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width,width,5,padding=2*dilation,dilation=dilation),
            nn.GroupNorm(8,width), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width,width,1),
        )
        self.scale = nn.Parameter(torch.tensor(.1))
    def forward(self,x):
        return F.gelu(x + self.scale*self.net(x))


class SpectralCNN(nn.Module):
    def __init__(self,bands,width=64):
        super().__init__()
        if width%8:raise ValueError('width must be divisible by 8')
        self.bands=int(bands)
        self.stem=nn.Conv1d(4,width,5,padding=2)
        self.blocks=nn.Sequential(*(SpectralResidualBlock(width,d) for d in (1,2,4,8,1)))
        self.head=nn.Conv1d(width,2,1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    def forward(self,feat):
        if feat.ndim!=3 or feat.shape[1:]!=(self.bands,4):
            raise RuntimeError(f'Expected [N,{self.bands},4], received {tuple(feat.shape)}')
        v=self.blocks(self.stem(feat.transpose(1,2)))
        out=self.head(v).transpose(1,2)
        # Input channel 1 is interpolation, 2 is corruption mask (1=missing).
        correction=torch.tanh(out[...,0])*torch.sigmoid(out[...,1])
        return feat[...,1]+feat[...,2]*correction


class SpectralHybrid(nn.Module):
    def __init__(self,bands,width=64,heads=4,layers=2):
        super().__init__()
        if width%8 or width%heads:raise ValueError('width must be divisible by 8 and heads')
        self.bands=int(bands)
        self.stem=nn.Conv1d(4,width,5,padding=2)
        self.local=nn.Sequential(SpectralResidualBlock(width,1),SpectralResidualBlock(width,2))
        self.pos=nn.Parameter(torch.zeros(1,bands,width))
        block=nn.TransformerEncoderLayer(d_model=width,nhead=heads,
            dim_feedforward=width*3,dropout=.06,activation='gelu',
            batch_first=True,norm_first=True)
        self.context=nn.TransformerEncoder(block,num_layers=layers,enable_nested_tensor=False)
        self.tail=SpectralResidualBlock(width,1)
        self.head=nn.Conv1d(width,2,1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    def forward(self,feat):
        if feat.ndim!=3 or feat.shape[1:]!=(self.bands,4):
            raise RuntimeError(f'Expected [N,{self.bands},4], received {tuple(feat.shape)}')
        v=self.local(self.stem(feat.transpose(1,2))).transpose(1,2)
        v=v+self.context(v+self.pos)
        out=self.head(self.tail(v.transpose(1,2))).transpose(1,2)
        correction=torch.tanh(out[...,0])*torch.sigmoid(out[...,1])
        return feat[...,1]+feat[...,2]*correction


def construct(name,bands,args):
    if name=='cnn':return SpectralCNN(bands,args.width)
    if name=='hybrid':return SpectralHybrid(bands,args.width,args.heads,args.layers)
    raise ValueError(name)


def loss_function(pred,batch):
    missing=batch['missing']
    denom=missing.sum().clamp_min(1.)
    err=(pred-batch['target'])
    rawerr=(batch['anchor']-batch['target']).abs().detach()
    mse=(err.square()*missing).sum()/denom
    worse=(F.relu(err.abs()-rawerr).square()*missing).sum()/denom
    conserve=((pred-batch['anchor']).square()*missing).sum()/denom
    return mse+.50*worse+.020*conserve


def atomic_save(obj,path:Path):
    temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(obj,temp)
    os.replace(temp,path)


def read_validation(loader,model,device,std,mean,core_names):
    """Paired reconstruction comparisons. Each indexed sample is identical for both networks."""
    model.eval()
    rows=[]
    next_idx=0
    sc=torch.as_tensor(std,device=device,dtype=torch.float64).view(1,-1)
    with torch.no_grad():
        for batch in loader:
            batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
            pred=model(batch['feat']).to(torch.float64)
            tgt=batch['target'].to(torch.float64)
            anch=batch['anchor'].to(torch.float64)
            mask=batch['missing'].to(torch.float64)
            diff=(pred-tgt)*sc; rawdiff=(anch-tgt)*sc
            n=mask.sum(dim=1)
            sq_model=(diff.square()*mask).sum(dim=1)
            sq_raw=(rawdiff.square()*mask).sum(dim=1)
            abs_model=(diff.abs()*mask).sum(dim=1)
            abs_raw=(rawdiff.abs()*mask).sum(dim=1)
            ci=batch['core'].cpu().numpy()
            fi=batch['family'].cpu().numpy()
            sv=batch['severity'].cpu().numpy()
            mcount=n.cpu().numpy(); se_m=sq_model.cpu().numpy();se_r=sq_raw.cpu().numpy()
            ae_m=abs_model.cpu().numpy();ae_r=abs_raw.cpu().numpy()
            for j in range(len(ci)):
                c=int(ci[j]);f=int(fi[j]);s=int(sv[j])+1
                if mcount[j]<1:continue
                rows.append(dict(case_index=next_idx+j, validation_core=core_names[c],
                    scenario=FAMILIES[f],severity=s,n_voxels=int(mcount[j]),
                    model_squared_error=float(se_m[j]),raw_squared_error=float(se_r[j]),
                    model_absolute_error=float(ae_m[j]),raw_absolute_error=float(ae_r[j]),
                    model_rmse=math.sqrt(float(se_m[j]/mcount[j])),
                    raw_rmse=math.sqrt(float(se_r[j]/mcount[j]))))
            next_idx+=len(ci)
    return rows


def summarize(rows,core_names):
    if not rows:raise RuntimeError('No validation samples')
    def score(group):
        n=sum(x['n_voxels'] for x in group)
        return dict(n_voxels=n,model_rmse=math.sqrt(sum(x['model_squared_error'] for x in group)/n),
          raw_rmse=math.sqrt(sum(x['raw_squared_error'] for x in group)/n),
          model_mae=sum(x['model_absolute_error'] for x in group)/n,
          raw_mae=sum(x['raw_absolute_error'] for x in group)/n,
          mean_case_model_rmse=sum(x['model_rmse'] for x in group)/len(group),
          mean_case_raw_rmse=sum(x['raw_rmse'] for x in group)/len(group),
          wins=sum(x['model_rmse']<x['raw_rmse']-1.e-12 for x in group),
          losses=sum(x['model_rmse']>x['raw_rmse']+1.e-12 for x in group),
          n_cases=len(group))
    all_score=score(rows)
    by_core={name:score([x for x in rows if x['validation_core']==name]) for name in core_names}
    by_severity={str(s):score([x for x in rows if x['severity']==s]) for s in range(1,6)}
    # Predeclared safety. Model must improve pooled RMSE, protect both independent
    # cores and mean case RMSE; need >=50% sample wins vs raw interpolation.
    safe=(all_score['model_rmse']<=.998*all_score['raw_rmse']
        and all(x['model_rmse']<=x['raw_rmse']*1.005 for x in by_core.values())
        and all_score['mean_case_model_rmse']<=all_score['mean_case_raw_rmse']*1.005
        and all_score['wins']>=all_score['losses']
        and all(by_severity[str(s)]['model_rmse']<=by_severity[str(s)]['raw_rmse']*1.05
                for s in (1,2)))
    return dict(overall=all_score,by_core=by_core,by_severity=by_severity,safe=bool(safe))


def write_csv(path,rows):
    if not rows:return
    with open(path,'w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def group_rows(path,rows):
    output=[]
    for name in ('SWIR4','SWIR9'):
        for fam in FAMILIES:
            for sev in range(1,6):
                r=[x for x in rows if x['validation_core']==name and x['scenario']==fam
                      and x['severity']==sev]
                if not r: continue
                n=sum(z['n_voxels'] for z in r)
                output.append(dict(validation_core=name,scenario=fam,severity=sev,
                    n_cases=len(r),n_voxels=n,
                    model_rmse=math.sqrt(sum(z['model_squared_error'] for z in r)/n),
                    spectral_raw_rmse=math.sqrt(sum(z['raw_squared_error'] for z in r)/n),
                    wins=sum(z['model_rmse'] < z['raw_rmse'] for z in r)))
    write_csv(path,output)


def ensure_metadata(saved,args,model_name,train_names,val_names,wave):
    cfg=saved.get('run_config',{})
    wanted={'architecture':model_name,'train_cores':list(train_names),
            'val_cores':list(val_names),'width':args.width,'layers':args.layers,
            'heads':args.heads,'train_samples':args.train_samples,
            'val_samples':args.val_samples,'batch_size':args.batch_size,
            'seed':args.seed,'lr':args.lr,'amp':bool(args.amp)}
    for key,v in wanted.items():
        if cfg.get(key)!=v:raise RuntimeError(
            f'Resume configuration mismatch for {model_name}: {key} saved={cfg.get(key)!r} requested={v!r}')
    if len(saved['wavelengths_nm'])!=len(wave) or not np.allclose(saved['wavelengths_nm'],wave,atol=.5,rtol=0):
        raise RuntimeError('Wavelength grid mismatch on resume')


def model_payload(model,optimizer,args,epoch,metrics,mean,std,wave,train_names,val_names,name,param_count):
    return dict(architecture='spectral_only_'+name+'_residual_v1',
        model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        optimizer_state_dict=optimizer.state_dict(),epoch=epoch,
        band_mean=np.asarray(mean,np.float32),band_std=np.asarray(std,np.float32),
        wavelengths_nm=np.asarray(wave,np.float32),
        model_params=dict(bands=model.bands,width=args.width,layers=args.layers,heads=args.heads),
        run_config=dict(architecture=name,train_cores=list(train_names),val_cores=list(val_names),
          width=args.width,layers=args.layers,heads=args.heads,train_samples=args.train_samples,
          val_samples=args.val_samples,batch_size=args.batch_size,seed=args.seed,
          lr=args.lr,amp=bool(args.amp)),
        validation=metrics,n_parameters=param_count,
        policy='SAME PIXEL ONLY. <2 unmasked bands ineligible; Region fallback. NO test acquisitions.')


def run_worker(args):
    if args.worker_model not in MODEL_NAMES or args.gpu_index not in (0,1):
        raise RuntimeError('Invalid worker parameters')
    if not torch.cuda.is_available() or torch.cuda.device_count()<=args.gpu_index:
        raise RuntimeError(f'GPU {args.gpu_index} unavailable')
    torch.cuda.set_device(args.gpu_index)
    device=torch.device('cuda',args.gpu_index)
    torch.set_num_threads(1)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import config
    spec=resolve_experiment(args)
    val_cores=tuple(c.upper() for c in args.val_cores)
    train_names, train_c,tr_i,val_c,val_i,wave=load_core_data(spec,val_cores)
    model_name=args.worker_model
    output=(Path(config.MODELS_DIR)/'texture_restoration_router_rf_v2'
        /'spectral_architecture_compare_v1'/model_name)
    output.mkdir(parents=True,exist_ok=True)
    last_path=output/'last_model.pt';best_path=output/'best_model.pt'
    candidate_path=output/'diagnostic_best_val.pt'
    history_path=output/'training_history.csv'
    if last_path.exists() and not args.resume:
        raise RuntimeError(f'{last_path} exists: use --resume or choose new output directory')
    stats_path=output/'TRAIN_ONLY_STATS.npz'
    if stats_path.exists():
        with np.load(stats_path,allow_pickle=False) as f:
            mean=np.asarray(f['mean'],np.float32)
            std=np.asarray(f['std'],np.float32)
            if tuple(f['train_cores'].tolist())!=tuple(train_names):
                raise RuntimeError('Cached stats belong to a different train split')
            if not np.allclose(f['wavelengths_nm'],wave,atol=.5,rtol=0):
                raise RuntimeError('Wavelength grids mismatched in cached stats')
    else:
        mean,std=make_stats(train_c,tr_i,len(wave),seed=4123)
        np.savez(stats_path,mean=mean,std=std,wavelengths_nm=wave,train_cores=np.array(train_names))
    data_train=SpectraDataset(train_names,train_c,tr_i,len(wave),mean,std,wave,args.train_samples,
                 4200,True,len(train_names))
    data_val=SpectraDataset(val_cores,val_c,val_i,len(wave),mean,std,wave,args.val_samples,
                 7770,False,len(val_cores))
    gen=torch.Generator();gen.manual_seed(args.seed)
    train_loader=DataLoader(data_train,batch_size=args.batch_size,shuffle=True,
           drop_last=False,num_workers=args.workers,pin_memory=True,generator=gen)
    val_loader=DataLoader(data_val,batch_size=args.batch_size,shuffle=False,
           num_workers=args.workers,pin_memory=True)
    model=construct(model_name,len(wave),args).to(device)
    npar=sum(p.numel() for p in model.parameters())
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    history=[];start=1;best_accepted=float('inf');best_candidate=float('inf')
    if args.resume and last_path.exists():
        chk=torch.load(last_path,map_location='cpu',weights_only=False)
        ensure_metadata(chk,args,model_name,train_names,val_cores,wave)
        model.load_state_dict(chk['model_state_dict'])
        optim.load_state_dict(chk['optimizer_state_dict'])
        start=int(chk['epoch'])+1
        if best_path.exists():
            prev=torch.load(best_path,map_location='cpu',weights_only=False)
            best_accepted=float(prev['validation']['overall']['model_rmse'])
        if candidate_path.exists():
            prev=torch.load(candidate_path,map_location='cpu',weights_only=False)
            best_candidate=float(prev['validation']['overall']['model_rmse'])
        if history_path.exists():
            with open(history_path,encoding='utf8',newline='') as f:history=list(csv.DictReader(f))
        # Recreate training shuffle generator position from previous epoch count.
        # A deterministic fixed ordering per epoch avoids hidden RNG divergence.
    print(f'[{model_name.upper()} / GPU {args.gpu_index}] {len(train_names)} train cores; '
          f'{len(val_cores)} validation cores; {npar:,} params; epochs {start}-{args.epochs}',flush=True)
    for epoch in range(start,args.epochs+1):
        data_train.set_epoch(epoch)
        # Deterministic permutation for each epoch, including after resume.
        gen.manual_seed(args.seed+epoch*104729)
        model.train();loss_total=0.;nsteps=0
        for batch in train_loader:
            batch={key:val.to(device,non_blocking=True) for key,val in batch.items()}
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',enabled=args.amp):
                pred=model(batch['feat'])
                loss=loss_function(pred,batch)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optim.step()
            loss_total+=float(loss.detach());nsteps+=1
        rows=read_validation(val_loader,model,device,std,mean,val_cores)
        score=summarize(rows,val_cores)
        rmse=score['overall']['model_rmse'];raw=score['overall']['raw_rmse'];accepted=score['safe']
        saved=model_payload(model,optim,args,epoch,score,mean,std,wave,train_names,val_cores,model_name,npar)
        atomic_save(saved,last_path)
        if rmse<best_candidate-1e-12:
            best_candidate=rmse
            atomic_save(saved,candidate_path)
            write_csv(output/'diagnostic_best_val_per_case.csv',rows)
            group_rows(output/'diagnostic_scenario_severity.csv',rows)
        if accepted and rmse<best_accepted-1e-12:
            best_accepted=rmse
            atomic_save(saved,best_path)
            write_csv(output/'best_accepted_val_per_case.csv',rows)
            group_rows(output/'best_accepted_scenario_severity.csv',rows)
        history.append(dict(epoch=epoch,train_loss=loss_total/max(nsteps,1),
            val_rmse=rmse,raw_rmse=raw,
            val_mean_case_rmse=score['overall']['mean_case_model_rmse'],
            raw_mean_case_rmse=score['overall']['mean_case_raw_rmse'],
            wins=score['overall']['wins'],losses=score['overall']['losses'],
            SWIR4_rmse=score['by_core'][val_cores[0]]['model_rmse'],
            SWIR9_rmse=score['by_core'][val_cores[1]]['model_rmse'],
            accepted=int(accepted)))
        write_csv(history_path,history)
        print(f'[{model_name.upper()} / GPU {args.gpu_index}] epoch {epoch}/{args.epochs}: '
              f'pooled_rmse={rmse:.6f} vs raw={raw:.6f}, '
              f'wins={score["overall"]["wins"]}/{score["overall"]["n_cases"]}, '
              f'accepted={accepted}',flush=True)
    status=dict(model=model_name,gpu=args.gpu_index,n_parameters=npar,
        best_accepted_exists=best_path.exists(),best_accepted_rmse=(best_accepted if math.isfinite(best_accepted) else None),
        best_candidate_rmse=(best_candidate if math.isfinite(best_candidate) else None),
        latest_epoch=max(start-1,args.epochs if start<=args.epochs else start-1),
        train_cores=list(train_names),validation_cores=list(val_cores),
        checkpoint_dir=str(output))
    (output/'TRAINING_SUMMARY.json').write_text(json.dumps(status,indent=2),encoding='utf8')
    print(f'[{model_name.upper()} / GPU {args.gpu_index}] done; output={output}',flush=True)


def combine(args):
    import config
    results=[]
    parent=Path(config.MODELS_DIR)/'texture_restoration_router_rf_v2'/'spectral_architecture_compare_v1'
    for name in MODEL_NAMES:
        folder=parent/name
        path=folder/'best_model.pt'
        approved=path.exists()
        if not approved:path=folder/'diagnostic_best_val.pt'
        if not path.exists():raise RuntimeError(f'Missing results from {name}: {path}')
        chk=torch.load(path,map_location='cpu',weights_only=False)
        score=chk['validation']
        a=score['overall'];bc=score['by_core']
        results.append(dict(architecture=name,accepted=int(approved),selected_epoch=chk['epoch'],
           n_parameters=chk['n_parameters'],pooled_rmse=a['model_rmse'],raw_rmse=a['raw_rmse'],
           mean_case_rmse=a['mean_case_model_rmse'],raw_mean_case_rmse=a['mean_case_raw_rmse'],
           wins=a['wins'],losses=a['losses'],n_cases=a['n_cases'],
           swir4_rmse=bc['SWIR4']['model_rmse'],swir4_raw=bc['SWIR4']['raw_rmse'],
           swir9_rmse=bc['SWIR9']['model_rmse'],swir9_raw=bc['SWIR9']['raw_rmse'],
           checkpoint=str(path)))
    per_case={}
    for r in results:
        folder=parent/r['architecture']
        filename=('best_accepted_val_per_case.csv' if r['accepted'] else
                  'diagnostic_best_val_per_case.csv')
        with open(folder/filename,encoding='utf8',newline='') as f:
            per_case[r['architecture']]={int(z['case_index']):z for z in csv.DictReader(f)}
    cnn=per_case['cnn']; hybrid=per_case['hybrid']
    if set(cnn)!=set(hybrid):
        raise RuntimeError('Architecture validations used different sample indices')
    paired=[]
    for idx in sorted(cnn):
        a,b=cnn[idx],hybrid[idx]
        if (a['scenario'],a['severity'],a['validation_core'],a['n_voxels']) != (b['scenario'],b['severity'],b['validation_core'],b['n_voxels']):
            raise RuntimeError(f'Validation alignment mismatch for sample {idx}')
        if abs(float(a['raw_squared_error'])-float(b['raw_squared_error']))>1e-7:
            raise RuntimeError(f'Raw baseline mismatch for sample {idx}')
        paired.append(dict(case_index=idx,validation_core=a['validation_core'],
            scenario=a['scenario'],severity=int(a['severity']),n_voxels=int(a['n_voxels']),
            spectral_raw_rmse=float(a['raw_rmse']),cnn_rmse=float(a['model_rmse']),
            hybrid_rmse=float(b['model_rmse']),
            cnn_wins=int(float(a['model_rmse'])<float(b['model_rmse'])),
            hybrid_wins=int(float(b['model_rmse'])<float(a['model_rmse']))))
    write_csv(parent/'PAIRED_CNN_VS_HYBRID.csv',paired)
    scenarios=[]
    for core_name in ('SWIR4','SWIR9'):
        for fam in FAMILIES:
            for sev in range(1,6):
                sub=[k for k in paired if k['validation_core']==core_name and
                     k['scenario']==fam and k['severity']==sev]
                if not sub:continue
                n=sum(k['n_voxels'] for k in sub)
                # Use model squared errors from per-case files for pooled RMSE.
                ids=[k['case_index'] for k in sub]
                def weighted_rmse(which):
                    return math.sqrt(sum(float(per_case[which][i]['model_squared_error']) for i in ids)/n)
                raw=math.sqrt(sum(float(cnn[i]['raw_squared_error']) for i in ids)/n)
                scenarios.append(dict(validation_core=core_name,scenario=fam,severity=sev,
                    n_cases=len(sub),n_voxels=n,raw_rmse=raw,
                    cnn_rmse=weighted_rmse('cnn'),hybrid_rmse=weighted_rmse('hybrid'),
                    cnn_sample_wins=sum(k['cnn_wins'] for k in sub),
                    hybrid_sample_wins=sum(k['hybrid_wins'] for k in sub)))
    write_csv(parent/'SCENARIO_SEVERITY_COMPARISON.csv',scenarios)
    write_csv(parent/'ARCHITECTURE_COMPARISON.csv',results)
    lines=['TPSSR SPECTRAL-ONLY ARCHITECTURE COMPARISON',
      'Training: SWIR1, SWIR2, SWIR3, SWIR6, SWIR7, SWIR8',
      'Core-level validation: SWIR4, SWIR9 (never SWIR5 or SWIR10)',
      'CNN GPU0, Hybrid CNN-Transformer GPU1; independent runs; identical data and seeds.',
      'All reported validation targets are synthetic held-out bands (same coordinates/masks).',
      'SSIM cannot be meaningfully evaluated from isolated spectra; final image test required.',
      'A model with accepted=0 is DIAGNOSTIC ONLY, NOT APPROVED to replace Spectral.',
      'Both architectures compared on identical 1800 validation samples.',
      'Fully missing spectra ineligible; Region remains necessary.','']
    for r in results:
        lines += [f'{r["architecture"]}: accepted={bool(r["accepted"])}; '
                f'epoch={r["selected_epoch"]}; params={r["n_parameters"]:,}',
                  f'  pooled RMSE={r["pooled_rmse"]:.7f}; raw={r["raw_rmse"]:.7f}',
                  f'  mean-case RMSE={r["mean_case_rmse"]:.7f}; raw={r["raw_mean_case_rmse"]:.7f}',
                  f'  wins={r["wins"]}/{r["n_cases"]}; SWIR4={r["swir4_rmse"]:.7f} raw={r["swir4_raw"]:.7f}; '
                  f'SWIR9={r["swir9_rmse"]:.7f} raw={r["swir9_raw"]:.7f}','']
    lines += [f'CNN beats Hybrid on {sum(z["cnn_wins"] for z in paired)} / {len(paired)} validation samples;',
              f'Hybrid beats CNN on {sum(z["hybrid_wins"] for z in paired)} / {len(paired)}.', '']
    (parent/'ARCHITECTURE_SUMMARY.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines),flush=True)


def smoke_test():
    bands=41
    arr=np.asarray(.35+.09*np.sin(np.arange(bands)*.22),np.float32)
    m=band_mask(np.random.default_rng(3),bands,'spatio_spectral_block',4)
    m[5:11]=True;m[0]=False;m[-1]=False
    mean=np.zeros(bands,np.float32);std=np.ones(bands,np.float32)
    anchor=interpolate_raw(arr,m,np.ones(bands,bool))
    assert anchor is not None
    f=np.stack((np.where(m,0.,arr),anchor,m.astype(np.float32),
                np.linspace(0,1,bands)),axis=-1).astype(np.float32)
    f2=f.copy();changed=arr.copy();changed[m]=3000
    f2[:,0]=np.where(m,0.,changed)
    batch={'target':torch.from_numpy(arr).unsqueeze(0),
         'anchor':torch.from_numpy(anchor).unsqueeze(0),
         'missing':torch.from_numpy(m.astype(np.float32)).unsqueeze(0)}
    class Args:width=32;heads=4;layers=1
    for name in MODEL_NAMES:
        net=construct(name,bands,Args())
        net.train()
        p=net(torch.from_numpy(f).unsqueeze(0))
        assert torch.allclose(p.squeeze(0),torch.from_numpy(anchor),atol=1e-6)
        loss=loss_function(p,batch)
        loss.backward()
        assert torch.isfinite(loss)
        assert net.head.weight.grad is not None
        net.eval()
        with torch.no_grad():
            p1=net(torch.from_numpy(f).unsqueeze(0))
            p2=net(torch.from_numpy(f2).unsqueeze(0))
        assert torch.allclose(p1,p2,atol=1e-6),'Hidden values influence predictions'
        assert np.allclose(p1[0][~m].numpy(),arr[~m],atol=1e-6)
    print('SMOKE TEST PASSED: CNN + Hybrid, loss, gradients, no masked GT leak, observed bands unchanged.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_experiment_args(parser)
    parser.add_argument('--gpus',type=int,choices=(2,),default=2,
          help='GPU0 trains CNN and GPU1 trains hybrid simultaneously; no NCCL')
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--train-samples',type=int,default=4500)
    parser.add_argument('--val-samples',type=int,default=1800)
    parser.add_argument('--batch-size',type=int,default=16,help='Per GPU')
    parser.add_argument('--width',type=int,default=64)
    parser.add_argument('--layers',type=int,default=2)
    parser.add_argument('--heads',type=int,default=4)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--seed',type=int,default=20261012)
    parser.add_argument('--workers',type=int,default=0,
          help='DataLoader workers per process; 0 minimizes RAM and avoids subprocess hangs')
    parser.add_argument('--val-cores',nargs='+',default=['SWIR4','SWIR9'])
    parser.add_argument('--amp',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke-test',action='store_true')
    parser.add_argument('--worker-model',choices=MODEL_NAMES,help=argparse.SUPPRESS)
    parser.add_argument('--gpu-index',type=int,default=-1,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.smoke_test:smoke_test();return
    if args.worker_model is not None:
        run_worker(args)
        return
    if any(k in os.environ for k in ('RANK','LOCAL_RANK','WORLD_SIZE')):
        raise RuntimeError('Unexpected distributed environment; run directly, not via torchrun')
    spec=resolve_experiment(args)
    val_names=tuple(x.upper() for x in args.val_cores)
    if tuple(val_names)!=('SWIR4','SWIR9'):
        raise RuntimeError('This controlled comparison fixes validation acquisitions to SWIR4, SWIR9')
    if len(spec.train)!=8 or tuple(spec.tests)!=('SWIR5','SWIR10') or set(val_names)&set(spec.tests):
        raise RuntimeError('Use --preset swir_8x2; never use test cores in training/validation')
    if args.batch_size<1 or args.train_samples<10 or args.val_samples<90 or args.epochs<1:
        raise ValueError('Insufficient training/validation settings')
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:
        raise RuntimeError('Two CUDA devices required')
    env=env_for(spec,spec.primary_test)
    env['OMP_NUM_THREADS']='1'
    env['MKL_NUM_THREADS']='1'
    # Do not forward any --worker-model. Workers enter run_worker directly; no recursive spawn.
    workers=[]
    try:
        for gpu,name in enumerate(MODEL_NAMES):
            cmd=[sys.executable,str(Path(__file__).resolve()),
                  '--preset','swir_8x2','--worker-model',name,'--gpu-index',str(gpu),
                  '--gpus','2','--epochs',str(args.epochs),
                  '--train-samples',str(args.train_samples),
                  '--val-samples',str(args.val_samples),'--batch-size',str(args.batch_size),
                  '--width',str(args.width),'--layers',str(args.layers),'--heads',str(args.heads),
                  '--workers',str(args.workers),'--lr',str(args.lr),'--seed',str(args.seed),
                  '--val-cores',*val_names]
            if args.resume:cmd.append('--resume')
            if args.amp:cmd.append('--amp')
            print(f'Launching {name} on GPU {gpu}',flush=True)
            workers.append((name,subprocess.Popen(cmd,cwd=str(CORE),env=env)))
        while workers:
            pending=[]
            for name,p in workers:
                code=p.poll()
                if code is None:pending.append((name,p))
                elif code!=0:raise RuntimeError(f'{name} worker failed with code {code}')
            workers=pending
            if workers:time.sleep(.5)
    except BaseException:
        for _,p in workers:
            if p.poll() is None:p.terminate()
        for _,p in workers:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill()
        raise
    # config.MODELS_DIR needs experiment env. This process already has spec but
    # has not imported config with it. Read output paths from child as project-relative.
    os.environ.update({k:v for k,v in env.items() if k.startswith(('HSI_','TPSSR_'))})
    if 'config' in sys.modules: del sys.modules['config']
    combine(args)

if __name__=='__main__': main()
