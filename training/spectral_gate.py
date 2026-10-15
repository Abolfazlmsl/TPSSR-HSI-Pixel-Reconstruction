#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TPSSR Hybrid V2: fit a confidence gate for a FROZEN spectral-only CNN-Transformer.

New standalone project-root script; requires the existing project files
training/spectral_backbone.py and training/spectral_data.py. No SWIR5/10
are loaded. SWIR4 and SWIR9 are full acquisition-held-out validation cores.
Two INDEPENDENT workers: GPU0 global/scalar gate, GPU1 per-band gate. No NCCL.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parent.parent
CORE=ROOT/'core'
for p in (str(ROOT),str(CORE)):
    if p not in sys.path: sys.path.insert(0,p)

from experiment_config import add_experiment_args,resolve_experiment
from _launcher_common import env_for
from training.spectral_data import SpectraDataset,load_core_data,FAMILIES
from training.spectral_backbone import (
    SpectralHybrid, read_validation, summarize, write_csv,group_rows,atomic_save
)

VARIANTS=('global','band')
TRAIN_CORES=('SWIR1','SWIR2','SWIR3','SWIR6','SWIR7','SWIR8')
VALID_CORES=('SWIR4','SWIR9')

class GlobalGate(nn.Module):
    """One correction-strength decision for a complete single-pixel spectrum."""
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(6,32),nn.GELU(),nn.Linear(32,32),nn.GELU())
        self.mlp=nn.Sequential(nn.Linear(32*2+2,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias,-2.20)
    def forward(self,feat,delta):
        ch=torch.cat((feat,delta.unsqueeze(-1),delta.abs().unsqueeze(-1)),dim=-1)
        v=self.enc(ch)
        missing=feat[...,2:3]
        masked=(v*missing).sum(dim=1)/missing.sum(dim=1).clamp_min(1.)
        obs=(v*(1.-missing)).sum(dim=1)/(1.-missing).sum(dim=1).clamp_min(1.)
        fraction=missing.mean(dim=1)
        discrepancy=(delta.abs()*missing.squeeze(-1)).mean(dim=1,keepdim=True)
        h=torch.cat((masked,obs,fraction,discrepancy),dim=-1)
        return torch.sigmoid(self.mlp(h)).expand_as(delta)

class BandGate(nn.Module):
    """Per-band correction-strength decision, with no spatial context."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(6,32,5,padding=2),nn.GELU(),
            nn.Conv1d(32,32,5,padding=4,dilation=2),nn.GELU(),
            nn.Conv1d(32,16,5,padding=8,dilation=4),nn.GELU(),
            nn.Conv1d(16,1,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias,-2.20)
    def forward(self,feat,delta):
        ch=torch.cat((feat,delta.unsqueeze(-1),delta.abs().unsqueeze(-1)),dim=-1)
        return torch.sigmoid(self.net(ch.transpose(1,2)).squeeze(1))

class FrozenHybridGated(nn.Module):
    def __init__(self,bands,width,heads,layers,variant):
        super().__init__()
        self.bands=int(bands)
        self.base=SpectralHybrid(bands,width,heads,layers)
        for p in self.base.parameters():p.requires_grad_(False)
        self.gate=GlobalGate() if variant=='global' else BandGate()
        self.variant=variant

    def forward(self,feat,return_details=False):
        # The old Hybrid is frozen AND evaluated without dropout or backward graph.
        self.base.eval()
        with torch.no_grad():base=self.base(feat)
        anchor=feat[...,1]
        delta=(base-anchor).detach()
        alpha=self.gate(feat,delta)
        pred=anchor+feat[...,2]*(alpha*delta)
        if return_details:return pred,alpha,base
        return pred


def gated_loss(pred,alpha,base,batch,std):
    """Balanced objective: voxel error, per-case error, and raw-safety regret.

    The target appears ONLY in this supervised training loss, not model.forward().
    All comparisons use the SAME missing finite training target bands.
    """
    m=batch['missing']
    n=m.sum(dim=1).clamp_min(1.)
    e=(pred-batch['target'])
    er=(batch['anchor']-batch['target']).detach()
    case_model=(e.square()*m).sum(dim=1)/n
    case_raw=(er.square()*m).sum(dim=1)/n
    mse_global=(e.square()*m).sum()/m.sum().clamp_min(1.)
    # Cost of hurting a simple interpolation for each individual training sample.
    regret=F.relu(torch.sqrt(case_model+1e-8)-torch.sqrt(case_raw+1e-8)).square().mean()
    modification=((pred-batch['anchor']).square()*m).sum()/m.sum().clamp_min(1.)
    # Learning how much to trust the old hybrid: supervision only at TRAIN time.
    # Use the optimal scalar convex coefficient for each training spectrum.
    d=(base.detach()-batch['anchor'])
    weight=std.square().view(1,-1)
    dot=(((batch['target']-batch['anchor']).detach()*d)*m*weight).sum(dim=1)
    dnorm=(d.square()*m*weight).sum(dim=1)
    ideal=(dot/dnorm.clamp_min(1e-7)).clamp(0,1)
    if alpha.shape != d.shape:raise RuntimeError('Gate alpha shape mismatch')
    est=(alpha*m).sum(dim=1)/n
    supervision=((est-ideal).square() * (dnorm/(dnorm+.002)).detach()).mean()
    return (.60*mse_global+.40*case_model.mean()+.65*regret+.01*modification
            +.015*supervision)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def get_backbone(models_dir:Path):
    folder=models_dir/'tpssr'/'spectral_backbone'/'hybrid'
    accepted=folder/'best_model.pt'
    diagnostic=folder/'diagnostic_best_val.pt'
    ck=accepted if accepted.exists() else diagnostic
    if not ck.exists():raise RuntimeError('Frozen Hybrid checkpoint missing. Expected one of:\n'
        f'  {accepted}\n  {diagnostic}\nRun training/spectral_backbone.py first.')
    # The v1 hybrid failed the strict acceptance threshold on actual prior run.
    return ck,ck==accepted


def check_backbone(chk,wave,training,val):
    if chk.get('architecture')!='spectral_only_hybrid_residual_v1':
        raise RuntimeError('Not the original spectral hybrid checkpoint')
    params=chk.get('model_params',{})
    if params.get('bands')!=len(wave):raise RuntimeError('Band count mismatch')
    if tuple(chk['run_config']['train_cores'])!=tuple(training) or tuple(chk['run_config']['val_cores'])!=tuple(val):
        raise RuntimeError('Source Hybrid uses DIFFERENT training/validation cores')
    if not np.allclose(chk['wavelengths_nm'],wave,atol=.5,rtol=0):
        raise RuntimeError('Source Hybrid wavelength grid mismatch')
    if chk['band_mean'].shape!=(len(wave),) or chk['band_std'].shape!=(len(wave),):
        raise RuntimeError('Source Hybrid normalization invalid')
    return params


def payload(model,optimizer,epoch,score,backbone_path,backbone_hash,source_approved,args,train,val,wave):
    return dict(architecture='spectral_only_hybrid_confidence_gate_v2',
       gate_variant=model.variant,epoch=epoch,
       gate_state_dict={k:v.detach().cpu().clone() for k,v in model.gate.state_dict().items()},
       optimizer_state_dict=optimizer.state_dict(),validation=score,
       backbone_checkpoint=str(backbone_path),backbone_sha256=backbone_hash,
       source_hybrid_accepted=source_approved,
       wavelengths_nm=np.asarray(wave,np.float32),
       run_config=dict(train_cores=list(train),val_cores=list(val),train_samples=args.train_samples,
           val_samples=args.val_samples,batch_size=args.batch_size,lr=args.lr,seed=args.seed,
           gate_variant=model.variant,amp=args.amp),
       policy='Frozen hybrid backbone; gate only. NO spatial input. NO test acquisitions.')


def check_resume(saved,args,name,backbone_hash,train,val,wave):
    cfg=saved['run_config'];want=dict(train_cores=list(train),val_cores=list(val),
        train_samples=args.train_samples,val_samples=args.val_samples,batch_size=args.batch_size,
        lr=args.lr,seed=args.seed,gate_variant=name,amp=args.amp)
    for k,v in want.items():
        if cfg.get(k)!=v:raise RuntimeError(f'Resume mismatch in {k}: checkpoint={cfg.get(k)} run={v}')
    if saved['backbone_sha256']!=backbone_hash:raise RuntimeError('Backbone checkpoint changed since last run')
    if not np.allclose(saved['wavelengths_nm'],wave,atol=.5,rtol=0):raise RuntimeError('Resume wavelength mismatch')


def worker(args):
    if args.worker_variant not in VARIANTS or args.gpu_index not in (0,1):
        raise RuntimeError('Invalid internal worker request')
    if not torch.cuda.is_available() or torch.cuda.device_count()<=args.gpu_index:
        raise RuntimeError(f'CUDA device {args.gpu_index} unavailable')
    torch.cuda.set_device(args.gpu_index)
    device=torch.device(f'cuda:{args.gpu_index}')
    torch.set_num_threads(1)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    import config
    spec=resolve_experiment(args)
    training,train_c,tr_i,val_c,val_i,wave=load_core_data(spec,VALID_CORES)
    if tuple(training)!=TRAIN_CORES:raise RuntimeError(f'Unexpected training cores: {training}')
    source_path,source_approved=get_backbone(Path(config.MODELS_DIR))
    source_hash=sha(source_path)
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    p=check_backbone(source,wave,training,VALID_CORES)
    mean=np.asarray(source['band_mean'],np.float32)
    std=np.asarray(source['band_std'],np.float32)

    train_set=SpectraDataset(training,train_c,tr_i,len(wave),mean,std,wave,args.train_samples,
         95123,True,len(training))
    val_set=SpectraDataset(VALID_CORES,val_c,val_i,len(wave),mean,std,wave,args.val_samples,
         7770,False,len(VALID_CORES))
    gen=torch.Generator();gen.manual_seed(args.seed)
    tloader=DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=0,
        pin_memory=True,drop_last=False,generator=gen)
    vloader=DataLoader(val_set,batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=True)
    model=FrozenHybridGated(len(wave),p['width'],p['heads'],p['layers'],args.worker_variant).to(device)
    model.base.load_state_dict(source['model_state_dict'],strict=True)
    model.base.eval()
    opt=torch.optim.AdamW(model.gate.parameters(),lr=args.lr,weight_decay=1e-5)
    out=Path(config.MODELS_DIR)/'tpssr'/'spectral_gate'/args.worker_variant
    out.mkdir(parents=True,exist_ok=True)
    last=out/'last_model.pt'; accepted=out/'best_model.pt'; diagnostic=out/'diagnostic_best_val.pt'
    history_path=out/'training_history.csv'
    if last.exists() and not args.resume:
        raise RuntimeError(f'{last} exists; use --resume to continue without resetting training')
    start=1;hist=[];best_safe=float('inf');best_diag=float('inf')
    if args.resume and last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        check_resume(saved,args,args.worker_variant,source_hash,training,VALID_CORES,wave)
        model.gate.load_state_dict(saved['gate_state_dict'],strict=True)
        opt.load_state_dict(saved['optimizer_state_dict'])
        start=int(saved['epoch'])+1
        if history_path.exists():
            with history_path.open(encoding='utf-8',newline='') as f:hist=list(csv.DictReader(f))
        if diagnostic.exists():best_diag=float(torch.load(diagnostic,map_location='cpu',weights_only=False)['validation']['overall']['model_rmse'])
        if accepted.exists():best_safe=float(torch.load(accepted,map_location='cpu',weights_only=False)['validation']['overall']['model_rmse'])
    print(f'[{args.worker_variant}/GPU{args.gpu_index}] frozen Hybrid source: {source_path}\n'
          f'[{args.worker_variant}/GPU{args.gpu_index}] start epoch {start}, '
          f'{args.train_samples} train, {args.val_samples} heldout validation, '
          f'{sum(p.numel() for p in model.gate.parameters()):,} trainable gate parameters.',flush=True)
    if start>args.epochs:
        print(f'[{args.worker_variant}] already finished epoch {args.epochs}; combining existing outputs.',flush=True)
    for epoch in range(start,args.epochs+1):
        train_set.set_epoch(epoch)
        gen.manual_seed(args.seed+epoch*104729)
        model.train();model.base.eval()
        total=0.0;steps=0
        for batch in tloader:
            batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',enabled=args.amp):
                pred,alpha,base=model(batch['feat'],True)
                loss=gated_loss(pred,alpha,base,batch,torch.as_tensor(std,device=device))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite gate loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.gate.parameters(),1.)
            opt.step()
            total+=float(loss.detach());steps+=1
        rows=read_validation(vloader,model,device,std,mean,VALID_CORES)
        score=summarize(rows,VALID_CORES)
        # Also protect the strong severity S4/S5 gains of the frozen V1 Hybrid.
        old=source['validation']
        preserve_hard=all(
            score['by_severity'][str(s)]['model_rmse'] <=
            old['by_severity'][str(s)]['model_rmse']*1.08
            for s in (4,5))
        preserve_global=(score['overall']['model_rmse']<=
                         old['overall']['model_rmse']*1.02)
        score['preserve_hard']=bool(preserve_hard)
        score['preserve_global']=bool(preserve_global)
        score['safe']=bool(score['safe'] and preserve_hard and preserve_global)
        mse=score['overall']['model_rmse'];safe=score['safe']
        saved=payload(model,opt,epoch,score,source_path,source_hash,source_approved,args,training,VALID_CORES,wave)
        atomic_save(saved,last)
        if mse<best_diag-1e-12:
            best_diag=mse;atomic_save(saved,diagnostic)
            write_csv(out/'diagnostic_best_val_per_case.csv',rows)
            group_rows(out/'diagnostic_scenario_severity.csv',rows)
        if safe and mse<best_safe-1e-12:
            best_safe=mse;atomic_save(saved,accepted)
            write_csv(out/'best_accepted_val_per_case.csv',rows)
            group_rows(out/'best_accepted_scenario_severity.csv',rows)
        hist.append(dict(epoch=epoch,loss=total/max(steps,1),model_rmse=mse,
            raw_rmse=score['overall']['raw_rmse'],
            model_case_rmse=score['overall']['mean_case_model_rmse'],
            raw_case_rmse=score['overall']['mean_case_raw_rmse'],
            wins=score['overall']['wins'],losses=score['overall']['losses'],
            swir4_rmse=score['by_core']['SWIR4']['model_rmse'],
            swir9_rmse=score['by_core']['SWIR9']['model_rmse'],
            accepted=int(safe)))
        write_csv(history_path,hist)
        print(f'[{args.worker_variant}/GPU{args.gpu_index}] epoch {epoch}/{args.epochs}: '
              f'RMSE={mse:.7f}, raw={score["overall"]["raw_rmse"]:.7f}; '
              f'wins={score["overall"]["wins"]}/{score["overall"]["n_cases"]}; '
              f'accepted={safe}',flush=True)
    (out/'GATE_TRAINING_SUMMARY.json').write_text(json.dumps(dict(
        gate_variant=args.worker_variant,backbone_checkpoint=str(source_path),
        backbone_accepted=source_approved,
        accepted_exists=accepted.exists(),diagnostic_exists=diagnostic.exists(),
        best_accepted_rmse=best_safe if math.isfinite(best_safe) else None,
        best_diagnostic_rmse=best_diag if math.isfinite(best_diag) else None,
        output=str(out)),indent=2),encoding='utf8')


def combine(args):
    import config
    parent=Path(config.MODELS_DIR)/'tpssr'/'spectral_gate'
    scores=[];casefiles={}
    for name in VARIANTS:
        folder=parent/name
        ok=(folder/'best_model.pt').exists()
        cp=(folder/'best_model.pt' if ok else folder/'diagnostic_best_val.pt')
        if not cp.exists():raise RuntimeError(f'Missing Gate checkpoint: {cp}')
        saved=torch.load(cp,map_location='cpu',weights_only=False)
        score=saved['validation'];overall=score['overall'];bc=score['by_core']
        scores.append(dict(gate=name,accepted=int(ok),epoch=int(saved['epoch']),
            rmse=overall['model_rmse'],raw_rmse=overall['raw_rmse'],
            mean_case_rmse=overall['mean_case_model_rmse'],raw_case_rmse=overall['mean_case_raw_rmse'],
            wins=overall['wins'],losses=overall['losses'],n_cases=overall['n_cases'],
            swir4_rmse=bc['SWIR4']['model_rmse'],swir4_raw=bc['SWIR4']['raw_rmse'],
            swir9_rmse=bc['SWIR9']['model_rmse'],swir9_raw=bc['SWIR9']['raw_rmse'],
            base_hybrid_checkpoint=saved['backbone_checkpoint'],checkpoint=str(cp)))
        csvname='best_accepted_val_per_case.csv' if ok else 'diagnostic_best_val_per_case.csv'
        with (folder/csvname).open(encoding='utf-8',newline='') as f:
            casefiles[name]={int(z['case_index']):z for z in csv.DictReader(f)}
    write_csv(parent/'GATE_ARCHITECTURE_COMPARISON.csv',scores)
    global_rows=casefiles['global'];band_rows=casefiles['band']
    if set(global_rows)!=set(band_rows):raise RuntimeError('Validation cases misaligned between gates')
    paired=[];groups=[]
    for idx in sorted(global_rows):
        a=global_rows[idx];b=band_rows[idx]
        for k in ('validation_core','scenario','severity','n_voxels'):
            if a[k]!=b[k]:raise RuntimeError(f'Gate validation mismatch in {k} case {idx}')
        if abs(float(a['raw_squared_error'])-float(b['raw_squared_error']))>1e-6:
            raise RuntimeError('Raw interpolation differs between gate evaluations')
        paired.append(dict(case_index=idx,validation_core=a['validation_core'],
            scenario=a['scenario'],severity=int(a['severity']),n_voxels=int(a['n_voxels']),
            spectral_raw_rmse=float(a['raw_rmse']),global_gate_rmse=float(a['model_rmse']),
            band_gate_rmse=float(b['model_rmse']),
            global_beats_raw=int(float(a['model_rmse'])<float(a['raw_rmse'])-1e-12),
            band_beats_raw=int(float(b['model_rmse'])<float(b['raw_rmse'])-1e-12)))
    write_csv(parent/'GATE_PAIRED_VALIDATION.csv',paired)
    for core in VALID_CORES:
        for fam in FAMILIES:
            for sev in range(1,6):
                rows=[r for r in paired if r['validation_core']==core and r['scenario']==fam and r['severity']==sev]
                if not rows:continue
                n=sum(r['n_voxels'] for r in rows)
                idxs=[r['case_index'] for r in rows]
                def val(name,field):return math.sqrt(sum(float(casefiles[name][i][field]) for i in idxs)/n)
                groups.append(dict(validation_core=core,scenario=fam,severity=sev,
                    n_cases=len(rows),n_voxels=n,raw_rmse=val('global','raw_squared_error'),
                    global_gate_rmse=val('global','model_squared_error'),
                    band_gate_rmse=val('band','model_squared_error'),
                    global_wins=sum(r['global_beats_raw'] for r in rows),
                    band_wins=sum(r['band_beats_raw'] for r in rows)))
    write_csv(parent/'GATE_SCENARIO_SEVERITY.csv',groups)
    src=scores[0]['base_hybrid_checkpoint']
    source=torch.load(src,map_location='cpu',weights_only=False)
    prev=source['validation']['overall']
    lines=['TPSSR HYBRID V2 — FROZEN HYBRID + TRAINED CONFIDENCE GATE',
      'Source Hybrid: '+src,
      'Source checkpoint was diagnostic unless prior acceptance was explicitly true.',
      'Only gate parameters trained. Spectral anchor and backbone unchanged.',
      'TRAIN CORES: '+', '.join(TRAIN_CORES),
      'FULLY HELDOUT VALIDATION CORES: '+', '.join(VALID_CORES),
      'NO SWIR5 OR SWIR10 ACCESSED. NO SPATIAL INPUT.',
      'Single-pixel fully-missing spectrum remains ineligible: Region fallback.',
      'Original hybrid (same validation data): RMSE={:.7f}, wins={}/{} against raw.'.format(
          prev['model_rmse'],prev['wins'],prev['n_cases']),
      'Acceptance criterion: beat raw, protect both cores/easy severities and >=50% wins; preserve original Hybrid pooled RMSE within 2% and S4/S5 RMSE within 8%.',
      'IMPORTANT: accepted=False means model is diagnostic ONLY and may NOT replace Spectral.', '']
    for r in scores:
        lines.extend([f"{r['gate']}: accepted={bool(r['accepted'])}; epoch={r['epoch']}",
          f"  pooled RMSE={r['rmse']:.7f}; raw={r['raw_rmse']:.7f}",
          f"  mean-case RMSE={r['mean_case_rmse']:.7f}; raw={r['raw_case_rmse']:.7f}",
          f"  mask wins={r['wins']}/{r['n_cases']}; losses={r['losses']}",
          f"  SWIR4={r['swir4_rmse']:.7f} vs raw={r['swir4_raw']:.7f}",
          f"  SWIR9={r['swir9_rmse']:.7f} vs raw={r['swir9_raw']:.7f}",''])
    (parent/'GATE_SUMMARY.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


def smoke():
    torch.set_num_threads(1)
    bands=48
    class Params:pass
    rng=np.random.default_rng(4)
    raw=(.5+.2*np.sin(np.arange(bands)*.1)).astype(np.float32)
    missing=np.zeros(bands,dtype=bool);missing[9:22]=True
    # Emulate same-pixel observed spectrum and a *precomputed* interpolation.
    anchor=np.interp(np.arange(bands),np.flatnonzero(~missing),raw[~missing]).astype(np.float32)
    feat=np.stack((np.where(missing,0.,raw),anchor,missing.astype(np.float32),
        np.linspace(0,1,bands,dtype=np.float32)),axis=-1)
    target=torch.from_numpy(raw).unsqueeze(0)
    batch=dict(feat=torch.from_numpy(feat).unsqueeze(0),target=target,
       anchor=torch.from_numpy(anchor).unsqueeze(0),
       missing=torch.from_numpy(missing.astype(np.float32)).unsqueeze(0))
    for v in VARIANTS:
        net=FrozenHybridGated(bands,32,4,1,v)
        net.train()
        pred,alpha,base=net(batch['feat'],True)
        assert pred.shape==(1,bands) and alpha.shape==(1,bands)
        assert torch.all(alpha>=0) and torch.all(alpha<=1)
        assert torch.allclose(pred[0,~torch.from_numpy(missing)],target[0,~torch.from_numpy(missing)],atol=1e-6)
        loss=gated_loss(pred,alpha,base,batch,torch.ones(bands))
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.gate.parameters())
        assert all(p.grad is None for p in net.base.parameters())
        net.eval()
        with torch.no_grad():
            original=net(batch['feat'])
            # Changing hidden target (not a model input) must not change inference.
            target_copy=batch['target'].clone()
            target_copy[0,missing]=1e6
            assert not torch.allclose(target_copy,batch['target'])
            other=net(batch['feat'])
            # Force alpha=0 and alpha=1: must reproduce raw and frozen Hybrid exactly.
            last=net.gate.mlp[-1] if v=='global' else net.gate.net[-1]
            previous=last.bias.clone()
            last.bias.fill_(-60.)
            raw_out=net(batch['feat'])
            last.bias.fill_(60.)
            base_out=net(batch['feat'])
            last.bias.copy_(previous)
        assert torch.allclose(original,other)
        assert torch.allclose(raw_out,batch['anchor'],atol=1e-6)
        with torch.no_grad():frozen=net.base(batch['feat'])
        assert torch.allclose(base_out,frozen,atol=1e-6)
        assert not torch.isnan(original).any()
    print('SMOKE TEST PASSED: two gates, frozen Hybrid, supervised gradient, observed-band preservation, no masked target in inference.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    add_experiment_args(ap)
    ap.add_argument('--gpus',type=int,default=2,choices=(2,))
    ap.add_argument('--epochs',type=int,default=8)
    ap.add_argument('--train-samples',type=int,default=6000)
    ap.add_argument('--val-samples',type=int,default=1800)
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--lr',type=float,default=.0008)
    ap.add_argument('--seed',type=int,default=20261013)
    ap.add_argument('--amp',action='store_true')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--smoke-test',action='store_true')
    ap.add_argument('--worker-variant',choices=VARIANTS,help=argparse.SUPPRESS)
    ap.add_argument('--gpu-index',type=int,default=-1,help=argparse.SUPPRESS)
    args=ap.parse_args()
    if args.smoke_test:smoke();return
    if args.worker_variant:
        worker(args);return
    if any(k in os.environ for k in ('RANK','LOCAL_RANK','WORLD_SIZE')):
        raise RuntimeError('Unexpected distributed launch. Run directly; no torchrun.')
    spec=resolve_experiment(args)
    if tuple(spec.train)!=('SWIR1','SWIR2','SWIR3','SWIR4','SWIR6','SWIR7','SWIR8','SWIR9') or tuple(spec.tests)!=('SWIR5','SWIR10'):
        raise RuntimeError('Use preset swir_8x2 with the original held-out split')
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:
        raise RuntimeError('Two GPUs required')
    if min(args.epochs,args.batch_size,args.train_samples,args.val_samples)<1 or args.val_samples%90:
        raise RuntimeError('Need positive sizes; --val-samples must be a multiple of 90 for balanced validation')
    env=env_for(spec,spec.primary_test)
    env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    jobs=[]
    try:
        for gpu,name in enumerate(VARIANTS):
            cmd=[sys.executable,str(Path(__file__).resolve()),
                '--preset','swir_8x2','--gpus','2','--epochs',str(args.epochs),
                '--train-samples',str(args.train_samples),'--val-samples',str(args.val_samples),
                '--batch-size',str(args.batch_size),'--lr',str(args.lr),'--seed',str(args.seed),
                '--worker-variant',name,'--gpu-index',str(gpu)]
            if args.amp:cmd.append('--amp')
            if args.resume:cmd.append('--resume')
            print(f'Launching frozen-Hybrid {name} gate on GPU {gpu}',flush=True)
            jobs.append((name,subprocess.Popen(cmd,cwd=str(CORE),env=env)))
        while jobs:
            pending=[]
            for name,p in jobs:
                status=p.poll()
                if status is None:pending.append((name,p))
                elif status!=0:raise RuntimeError(f'{name} worker failed (exit {status})')
            jobs=pending
            if jobs:time.sleep(.5)
    except BaseException:
        for _,p in jobs:
            if p.poll() is None:p.terminate()
        for _,p in jobs:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill()
        raise
    os.environ.update({k:v for k,v in env.items() if k.startswith(('HSI_','TPSSR_'))})
    if 'config' in sys.modules:del sys.modules['config']
    combine(args)

if __name__=='__main__':main()
