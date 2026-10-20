"""Learned local spectral fusion with a frozen hybrid backbone.

The model learns donor-specific spectral corrections and blends them with
interpolation and the accepted CNN-Transformer output. Training and validation
use only source acquisitions; the destination acquisition is never sampled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / 'core'
for p in (ROOT, CORE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiment_config import add_experiment_args, resolve_experiment, spectral_validation_cores
from src.training_corruptions import generate_training_mask
from training.spectral_data import load_core_data
from training.spectral_gate import sha

ARCH = 'local_neighbor_residual_fusion_v1'
RADIUS = 14
MAX_DONORS = 8
TRAIN_SCENARIOS = (
    'isolated_voxels', 'single_detector_line', 'multiple_detector_lines',
    'partial_vertical_lines', 'spatial_patch', 'consecutive_spectral_bands',
    'spatio_spectral_block', 'band_wide_saturation', 'mixed_saturation',
)
# The last two mixed-saturation severities remove the entire center spectrum.
# Those samples have no spectral anchors and belong exclusively to Region.
TRAIN_CASES = tuple((name,sev) for name in TRAIN_SCENARIOS for sev in range(1,6)
                    if name!='mixed_saturation' or sev<4)


def crop_at(cube, mask, y, x, radius=RADIUS):
    """Copy just the local window. Hidden measurements are erased immediately."""
    h, w, _ = cube.shape
    y0, y1 = max(0, y-radius), min(h, y+radius+1)
    x0, x1 = max(0, x-radius), min(w, x+radius+1)
    values = np.array(cube[y0:y1, x0:x1, :], dtype=np.float32, copy=True)
    bad = np.array(mask[y0:y1, x0:x1, :], dtype=bool, copy=True)
    bad |= ~np.isfinite(values)
    values[bad] = np.nan
    return values, bad, y-y0, x-x0


def neighbor_features(cube, mask, cy, cx, missing, observed, anchor, std):
    """Top-K observed-only curvature donors, using the V2 similarity criteria.

    A donor must cover the entire synthetic target gap. A candidate is scored
    exclusively on bands observed at the target. Missing reflectances never
    participate, even if they still exist in the original calibrated array.
    """
    h, w, bands = cube.shape
    residual = np.zeros((MAX_DONORS, bands), np.float32)
    metadata = np.zeros((MAX_DONORS, 2), np.float32)
    valid = np.zeros(MAX_DONORS, np.float32)
    reference = np.asarray(cube[cy, cx, observed], np.float64)
    if reference.size < 30 or not missing.any():
        return residual, metadata, valid
    scale = max(float(np.std(reference)), 0.01)
    minimum = max(30, int(0.7 * observed.sum()))
    axis = np.arange(bands, dtype=np.float64)
    candidates = []
    previous_radius = 0
    for radius in (4, 8, 14):
        for yy in range(max(0, cy-radius), min(h, cy+radius+1)):
            for xx in range(max(0, cx-radius), min(w, cx+radius+1)):
                dist = max(abs(yy-cy), abs(xx-cx))
                if (yy == cy and xx == cx) or dist <= previous_radius:
                    continue
                donor = cube[yy, xx]
                donor_valid = (~mask[yy, xx]) & np.isfinite(donor)
                if not np.all(donor_valid[missing]):
                    continue
                joint = donor_valid & observed
                if int(joint.sum()) < minimum:
                    continue
                nx = np.asarray(donor[joint], np.float64)
                ty = np.asarray(cube[cy, cx, joint], np.float64)
                nxmean, tymean = nx.mean(), ty.mean()
                var = float(np.mean((nx-nxmean)**2))
                if var < 1.e-9:
                    continue
                slope = float(np.mean((nx-nxmean)*(ty-tymean))/var)
                if not 0.5 <= slope <= 1.5:
                    continue
                intercept = tymean-slope*nxmean
                discrepancy = float(np.sqrt(np.mean((slope*nx+intercept-ty)**2)))/scale
                if not np.isfinite(discrepancy) or discrepancy > 0.75:
                    continue
                support = np.flatnonzero(donor_valid & ~missing)
                if support.size < 2:
                    continue
                estimated = np.interp(axis, support, donor[support])
                correction = np.zeros(bands, dtype=np.float32)
                correction[missing] = (
                    slope*(donor[missing].astype(np.float64)-estimated[missing])/std[missing]
                ).astype(np.float32)
                score = discrepancy + 0.012*float(np.hypot(yy-cy, xx-cx))
                candidates.append((score, discrepancy, float(np.hypot(yy-cy, xx-cx)), correction))
        if len(candidates) >= 5:
            break
        previous_radius = radius
    candidates.sort(key=lambda item:item[0])
    for k, (_rank, discrepancy, distance, correction) in enumerate(candidates[:MAX_DONORS]):
        residual[k] = correction
        metadata[k] = (discrepancy, distance/14.0)
        valid[k] = 1.
    return residual, metadata, valid


def make_features(cube, mask, cy, cx, mean, std, wavelength):
    """Identical input assembly during training and evaluation.

    Inputs contain only observed reflectances and the missingness mask. The
    caller may retain ground truth separately, exclusively for the loss.
    """
    bands = cube.shape[2]
    m = np.asarray(mask[cy, cx], bool)
    t = cube[cy, cx]
    observed = (~m) & np.isfinite(t)
    indices = np.flatnonzero(observed)
    if indices.size < 2 or not m.any():
        return None
    axis = np.arange(bands, dtype=np.float64)
    anchor = np.interp(axis, indices, t[indices]).astype(np.float32)
    missing = ~observed
    residual, donor_meta, donor_valid = neighbor_features(
        cube, mask, cy, cx, missing, observed, anchor, std)
    observed_normal = np.zeros(bands, np.float32)
    observed_normal[observed] = (t[observed]-mean[observed])/std[observed]
    norm_anchor = (anchor-mean)/std
    wave = (wavelength-wavelength[0])/max(float(wavelength[-1]-wavelength[0]), 1e-6)
    nearest = np.minimum.reduce(np.abs(axis[:,None]-indices[None,:]),axis=1) / max(bands-1, 1)
    edges = np.diff(np.pad(missing.astype(np.int8), (1,1)))
    length = np.max(np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)) if np.any(edges) else 0
    global_loss = mask.mean(axis=(0,1)).astype(np.float32)
    # Spatially distant pixels inside a block receive less local support than
    # those at its boundary. Compute support only from observable donor bands.
    available = np.all(((~mask)&np.isfinite(cube))[:,:,missing],axis=2)
    available[cy,cx] = False
    sy,sx = np.where(available)
    if sy.size:
        distance = float(np.minimum(14,np.min(np.maximum(np.abs(sy-cy),np.abs(sx-cx)))))/14.
    else:
        distance = 1.
    support_fraction = float(available.mean())
    features = np.stack((
        observed_normal, norm_anchor, missing.astype(np.float32), wave,
        np.full(bands, observed.mean(), np.float32),
        np.full(bands, length/max(bands,1), np.float32),
        global_loss, nearest.astype(np.float32),
        np.full(bands, np.sum(donor_valid)/MAX_DONORS, np.float32),
        np.full(bands, distance, np.float32),
        np.full(bands, support_fraction, np.float32),
    ), axis=-1).astype(np.float32)
    hybrid_input = np.stack((observed_normal, norm_anchor, missing.astype(np.float32),wave),axis=-1)
    return (features, hybrid_input, residual, donor_meta, donor_valid)


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width,width,5,padding=2*dilation,dilation=dilation),
            nn.GELU(),nn.Conv1d(width,width,1),nn.GELU())
        self.scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, x):
        return x+self.scale*self.net(x)


class NeighborFusion(nn.Module):
    """Donor attention and per-band learned mixture of three spectral experts."""
    def __init__(self, width=48):
        super().__init__()
        self.width = int(width)
        self.donor_encoder = nn.Sequential(nn.Conv1d(3,16,5,padding=2),nn.GELU(),
                                           nn.Conv1d(16,16,5,padding=4,dilation=2),nn.GELU())
        self.donor_attention = nn.Conv1d(16,1,1)
        self.context = nn.Sequential(nn.Conv1d(14,self.width,5,padding=2),nn.GELU(),
                         ResidualBlock(self.width,1),ResidualBlock(self.width,2),
                         ResidualBlock(self.width,4),ResidualBlock(self.width,8))
        self.head = nn.Conv1d(self.width,4,1)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([1.8,-0.8,-0.8,0.0]))

    def forward(self, feat, hybrid, donors, donor_meta, donor_valid):
        n, bands, channels = feat.shape
        if channels != 11 or donors.shape != (n,MAX_DONORS,bands):
            raise ValueError('Invalid fusion features')
        d = torch.stack((
            donors,
            donor_meta[...,0,None].expand(-1,-1,bands),
            donor_meta[...,1,None].expand(-1,-1,bands)),dim=2)
        encoded = self.donor_encoder(d.reshape(n*MAX_DONORS,3,bands))
        logits = self.donor_attention(encoded).reshape(n,MAX_DONORS,bands)
        eligible = donor_valid[...,None]>0.5
        attn = torch.softmax(logits.masked_fill(~eligible,-1e4),dim=1)
        attn = attn*eligible.to(attn.dtype)
        attn = attn / attn.sum(1,keepdim=True).clamp_min(1e-8)
        correction = (attn*donors).sum(dim=1)
        certainty = attn.square().sum(dim=1)
        have_donors = eligible.any(dim=1).to(feat.dtype)
        anchor = feat[...,1]
        d_hybrid = hybrid-anchor
        encoded_input = torch.cat((feat,d_hybrid[...,None],
                                   correction[...,None],certainty[...,None]),dim=-1)
        raw = self.head(self.context(encoded_input.transpose(1,2)))
        weights = raw[:,:3].clone()
        weights[:,2] = weights[:,2].masked_fill(have_donors<0.5,-1e4)
        weights = torch.softmax(weights,dim=1)
        gain = torch.sigmoid(raw[:,3])
        neighbor = anchor + gain*correction
        pred = weights[:,0]*anchor+weights[:,1]*hybrid+weights[:,2]*neighbor
        pred = torch.where(feat[...,2]>0.5,pred,feat[...,0])
        return pred


class FusionDataset(Dataset):
    def __init__(self, names,cubes,invalids,mean,std,wavelength,count,seed,training):
        self.names,self.cubes,self.invalids=tuple(names),cubes,invalids
        self.mean,self.std,self.wavelength=mean,std,wavelength
        self.count,self.seed,self.training=int(count),int(seed),bool(training)
        self.epoch=0

    def set_epoch(self, epoch):
        self.epoch=int(epoch)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        index=int(index)
        rng=np.random.default_rng(self.seed+10007*index+(self.epoch*10000019 if self.training else 0))
        idx=index%len(self.names)
        cube,invalid=self.cubes[idx],self.invalids[idx]
        h,w,b=cube.shape
        if self.training:
            scenario,severity=TRAIN_CASES[int(rng.integers(len(TRAIN_CASES)))]
        else:
            scenario,severity=TRAIN_CASES[(index//len(self.names))%len(TRAIN_CASES)]
        for _ in range(80):
            y=int(rng.integers(RADIUS,max(RADIUS+1,h-RADIUS))) if h>2*RADIUS else int(rng.integers(h))
            x=int(rng.integers(RADIUS,max(RADIUS+1,w-RADIUS))) if w>2*RADIUS else int(rng.integers(w))
            raw,bad,cy,cx=crop_at(cube,invalid,y,x)
            truth=raw[cy,cx].copy()
            original_valid=~bad[cy,cx]
            if original_valid.sum()<max(35,int(0.90*b)):
                continue
            artificial,_,_=generate_training_mask(raw.shape,rng,severity=severity,scenario=scenario,
                                                  center=(cy,cx),band_center=int(rng.integers(b)))
            target_mask=artificial[cy,cx] & original_valid
            if not target_mask.any():
                continue
            full_mask=bad|artificial
            raw[full_mask]=np.nan
            features=make_features(raw,full_mask,cy,cx,self.mean,self.std,self.wavelength)
            if features is None:
                continue
            features=tuple(torch.from_numpy(np.ascontiguousarray(z)) for z in features)
            norm_truth=np.where(original_valid,(np.nan_to_num(truth,nan=0.)-self.mean)/self.std,0.).astype(np.float32)
            return (*features,torch.from_numpy(norm_truth),
                    torch.from_numpy(target_mask.astype(np.float32)),
                    torch.tensor(idx,dtype=torch.long),
                    torch.tensor(TRAIN_SCENARIOS.index(scenario),dtype=torch.long),
                    torch.tensor(severity-1,dtype=torch.long))
        raise RuntimeError(f'Unable to sample valid {scenario} S{severity} from {self.names[idx]}')


def frozen_hybrid(model, hybrid_feats):
    with torch.no_grad():
        return model(hybrid_feats).detach()


def batch_losses(pred,batch,std):
    truth=batch[5]
    mask=batch[6]
    diff=(pred-truth)*std
    anchor_diff=(batch[0][...,1]-truth)*std
    per_sample=(diff.square()*mask).sum(1)/mask.sum(1).clamp_min(1.)
    correction=(pred-batch[0][...,1])*std
    loss=(per_sample+0.005*(correction.square()*mask).sum(1)/mask.sum(1).clamp_min(1.)).mean()
    baseline=(anchor_diff.square()*mask).sum(1)
    return loss, (diff.square()*mask).sum(),baseline,mask.sum()


@torch.no_grad()
def neighbor_v2_reference(batch, hybrid):
    """Source-validation approximation of V2's spectral decision logic.

    Inference computes global-band prevalence over the complete mask. The
    synthetic validation window provides an approximation of that prevalence.
    Final V2 comparison must use the unchanged 200-mask test runner.
    """
    feat,donors,valid=batch[0],batch[2],batch[4]
    anchor=feat[...,1]
    counts=valid.sum(1)
    masked=valid[...,None] < 0.5
    median=donors.masked_fill(masked,float('nan')).nanmedian(dim=1).values
    median=torch.nan_to_num(median,nan=0.)
    neighbor=anchor+0.5*median
    longest=feat[:,0,5]*feat.shape[1]
    use_neighbor=(counts>=5)[:,None]
    use_hybrid=(longest>=10)[:,None]
    global_band=(feat[...,6]>=0.995)
    guarded=torch.where(global_band, anchor+0.5*(hybrid-anchor),hybrid)
    return torch.where(use_neighbor,neighbor,torch.where(use_hybrid,guarded,anchor))


def csv_write(path,rows):
    if not rows:return
    path=Path(path)
    with path.with_suffix('.tmp').open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    os.replace(path.with_suffix('.tmp'),path)


def checkpoint_path(root):
    return Path(root)/'tpssr'/'neighbor_fusion'/'best_model.pt'


def load_fusion_checkpoint(path, device, gate_path, wave, mean, std):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('architecture')!=ARCH:
        raise RuntimeError('Unexpected fusion architecture')
    if saved.get('gate_sha256')!=sha(gate_path):
        raise RuntimeError('Fusion training gate and inference gate differ')
    if (not np.allclose(saved['wavelengths_nm'],wave,atol=.5,rtol=0) or
        not np.allclose(saved['band_mean'],mean,atol=1e-6,rtol=1e-5) or
        not np.allclose(saved['band_std'],std,atol=1e-6,rtol=1e-5)):
        raise RuntimeError('Fusion wavelength or normalization mismatch')
    model=NeighborFusion(saved['width']).to(device)
    model.load_state_dict(saved['state_dict'],strict=True)
    return model.eval(),saved


def run_training(args):
    from src.pipeline import _load_accepted_spectral_hybrid_v2
    import config
    spec=resolve_experiment(args)
    validation=tuple(spectral_validation_cores(spec))
    training=tuple(n for n in spec.train if n not in validation)
    rank=int(os.environ.get('RANK',0));local_rank=int(os.environ.get('LOCAL_RANK',0))
    world=int(os.environ.get('WORLD_SIZE',1))
    if world!=args.gpus or world!=2 or not torch.cuda.is_available() or torch.cuda.device_count()<2:
        raise RuntimeError('Fusion training requires torchrun with exactly two CUDA workers')
    torch.cuda.set_device(local_rank)
    device=torch.device('cuda',local_rank)
    torch.set_num_threads(1)
    dist.init_process_group('nccl')
    try:
        random.seed(args.seed+rank);np.random.seed(args.seed+rank);torch.manual_seed(args.seed+rank)
        actual_train,train_c,train_i,val_c,val_i,wave=load_core_data(spec,validation)
        if tuple(actual_train)!=training:
            raise RuntimeError('Unexpected source training split')
        hybrid,mean,std,_,gate_path,backbone_path=_load_accepted_spectral_hybrid_v2(device)
        for param in hybrid.parameters():param.requires_grad_(False)
        reference=torch.as_tensor(std,device=device).view(1,-1)
        train_set=FusionDataset(training,train_c,train_i,mean,std,wave,args.train_samples,args.seed,True)
        valid_set=FusionDataset(validation,val_c,val_i,mean,std,wave,args.val_samples,args.seed+4000,False)
        train_sampler=DistributedSampler(train_set,num_replicas=world,rank=rank,shuffle=True,seed=args.seed)
        val_sampler=DistributedSampler(valid_set,num_replicas=world,rank=rank,shuffle=False)
        train_loader=DataLoader(train_set,batch_size=args.batch_size,sampler=train_sampler,
                                num_workers=args.workers,pin_memory=True,drop_last=False)
        val_loader=DataLoader(valid_set,batch_size=args.batch_size,sampler=val_sampler,
                              num_workers=args.workers,pin_memory=True,drop_last=False)
        model=NeighborFusion(args.width).to(device)
        ddp=DistributedDataParallel(model,device_ids=[local_rank])
        optimizer=torch.optim.AdamW(ddp.parameters(),lr=args.lr,weight_decay=1e-4)
        output=checkpoint_path(config.MODELS_DIR).parent
        if rank==0:
            output.mkdir(parents=True,exist_ok=True)
        dist.barrier()
        if (output/'best_model.pt').exists():
            raise FileExistsError('Fusion checkpoint already exists; refusing to overwrite it.')
        best=float('inf')
        history=[]
        for epoch in range(1,args.epochs+1):
            train_set.set_epoch(epoch)
            train_sampler.set_epoch(epoch)
            ddp.train()
            running=0.
            for batch in train_loader:
                batch=[z.to(device,non_blocking=True) for z in batch]
                with torch.no_grad(): h=frozen_hybrid(hybrid,batch[1])
                optimizer.zero_grad(set_to_none=True)
                pred=ddp(batch[0],h,batch[2],batch[3],batch[4])
                loss,_,_,_=batch_losses(pred,batch,reference)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite fusion training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ddp.parameters(),1.)
                optimizer.step();running+=float(loss.item())
            ddp.eval()
            sums=torch.zeros((len(validation),6),dtype=torch.float64,device=device)
            with torch.inference_mode():
                for batch in val_loader:
                    batch=[z.to(device,non_blocking=True) for z in batch]
                    h=frozen_hybrid(hybrid,batch[1])
                    pred=ddp(batch[0],h,batch[2],batch[3],batch[4])
                    _,sq,base,nobs=batch_losses(pred,batch,reference)
                    v2=neighbor_v2_reference(batch,h)
                    for ci in range(len(validation)):
                        eligible=batch[7]==ci
                        if not eligible.any():continue
                        mask=batch[6][eligible]
                        sums[ci,0]+=(((pred[eligible]-batch[5][eligible])*reference).square()*mask).sum().double()
                        sums[ci,1]+=(((batch[0][eligible,:,1]-batch[5][eligible])*reference).square()*mask).sum().double()
                        sums[ci,2]+=mask.sum().double()
                        sums[ci,3]+=eligible.sum().double()
                        sums[ci,4]+=(((v2[eligible]-batch[5][eligible])*reference).square()*mask).sum().double()
                        sums[ci,5]+=(((h[eligible]-batch[5][eligible])*reference).square()*mask).sum().double()
            dist.all_reduce(sums,op=dist.ReduceOp.SUM)
            summary=sums.detach().cpu().numpy()
            pooled=summary.sum(axis=0)
            rmse=math.sqrt(pooled[0]/max(pooled[2],1.))
            baseline=math.sqrt(pooled[1]/max(pooled[2],1.))
            ref_v2=math.sqrt(pooled[4]/max(pooled[2],1.))
            ref_hybrid=math.sqrt(pooled[5]/max(pooled[2],1.))
            record={'epoch':epoch,'fusion_rmse':rmse,'interpolation_rmse':baseline,
                    'neighbor_v2_approx_rmse':ref_v2,'hybrid_rmse':ref_hybrid,
                    'train_loss_rank0':running/max(len(train_loader),1)}
            for i,name in enumerate(validation):
                record[f'{name}_fusion_rmse']=math.sqrt(summary[i,0]/max(summary[i,2],1.))
                record[f'{name}_interpolation_rmse']=math.sqrt(summary[i,1]/max(summary[i,2],1.))
                record[f'{name}_neighbor_v2_approx_rmse']=math.sqrt(summary[i,4]/max(summary[i,2],1.))
                record[f'{name}_hybrid_rmse']=math.sqrt(summary[i,5]/max(summary[i,2],1.))
            if rank==0:
                history.append(record);csv_write(output/'training_history.csv',history)
                print(f'Fusion epoch {epoch}/{args.epochs}: validation RMSE {rmse:.6f} / interpolation {baseline:.6f} / hybrid {ref_hybrid:.6f} / NeighborV2~ {ref_v2:.6f} '
                      + ', '.join(f'{name}: {record[f"{name}_fusion_rmse"]:.6f}' for name in validation),flush=True)
                if rmse < best-1e-12:
                    best=rmse
                    payload={'architecture':ARCH,'width':args.width,'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                             'wavelengths_nm':np.asarray(wave,np.float32),
                             'band_mean':mean,'band_std':std,
                             'gate_sha256':sha(gate_path),'backbone_sha256':sha(backbone_path),
                             'training_cores':list(training),'validation_cores':list(validation),
                             'experiment_name':spec.experiment_name,'epoch':epoch,'validation_rmse':rmse,
                             'val_spectral_rmse':baseline,'val_neighbor_v2_approx_rmse':ref_v2,'val_hybrid_rmse':ref_hybrid,
                             'settings':{'train_samples':args.train_samples,'val_samples':args.val_samples,
                                         'batch_size':args.batch_size,'seed':args.seed,'radius':RADIUS}}
                    tmp=output/'best_model.pt.tmp'
                    torch.save(payload,tmp);os.replace(tmp,output/'best_model.pt')
        if rank==0:
            (output/'TRAINING_SUMMARY.json').write_text(json.dumps({
                'trained_on':training,'validated_on':validation,'best_rmse':best,
                'note':'Internal source validation only; run the independent test protocol for comparison.'},indent=2),encoding='utf8')
    finally:
        dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_experiment_args(parser)
    parser.add_argument('--gpus',type=int,choices=[2],default=2)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--train-samples',type=int,default=6000)
    parser.add_argument('--val-samples',type=int,default=1000)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--width',type=int,default=48)
    parser.add_argument('--lr',type=float,default=2e-4)
    parser.add_argument('--seed',type=int,default=7410)
    args=parser.parse_args()
    if min(args.epochs,args.train_samples,args.val_samples,args.batch_size,args.width)>0 and args.workers>=0:
        run_training(args)
    else:
        parser.error('Training parameters must be positive')


if __name__=='__main__':
    main()
