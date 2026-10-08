from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from src.sparse_router_model import ResBlock2d, _group_count


class ComponentThreeExpertRouter(nn.Module):
    """V31 learned component router.

    It predicts expected error per spectral band for three experts, but the
    *decision* is made after aggregation over a complete spatial corruption
    component.  Therefore one component receives exactly one expert.
    """

    ARCHITECTURE_VERSION = "tpssr_v31_component_three_expert_router_v1"
    CANDIDATE_NAMES = ("spectral", "deep_v31", "original_v27_v275_plus_partial_adapter")

    def __init__(self, n_bands, wavelengths_nm, d_model=192, n_heads=8,
                 n_layers=6, ff_mult=4, dropout=0.05, component_dim=12):
        super().__init__()
        self.n_bands = int(n_bands)
        self.d_model = int(d_model)
        self.n_candidates = 3
        self.component_dim = int(component_dim)

        wavelengths = np.asarray(wavelengths_nm, dtype=np.float32)
        if len(wavelengths) != self.n_bands:
            raise ValueError("Wavelength count does not match n_bands.")
        wmin, wmax = float(wavelengths.min()), float(wavelengths.max())
        self.register_buffer(
            "wavelength_norm",
            torch.from_numpy(((wavelengths-wmin)/max(wmax-wmin,1e-6)).astype(np.float32)),
            persistent=True,
        )

        self.spatial_stem = nn.Sequential(
            nn.Conv2d(2*self.n_bands, 96, 1, bias=False),
            nn.GroupNorm(_group_count(96),96), nn.SiLU(),
            ResBlock2d(96,dilation=1,dropout=0.03),
            ResBlock2d(96,dilation=2,dropout=0.03),
        )
        self.spatial_down1 = nn.Sequential(
            nn.Conv2d(96,160,3,stride=2,padding=1,bias=False),
            nn.GroupNorm(_group_count(160),160), nn.SiLU(),
            ResBlock2d(160,dilation=2,dropout=0.04),
            ResBlock2d(160,dilation=4,dropout=0.04),
        )
        self.spatial_down2 = nn.Sequential(
            nn.Conv2d(160,224,3,stride=2,padding=1,bias=False),
            nn.GroupNorm(_group_count(224),224), nn.SiLU(),
            ResBlock2d(224,dilation=2,dropout=0.04),
            ResBlock2d(224,dilation=4,dropout=0.04),
        )
        self.spatial_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(224,d_model), nn.SiLU(), nn.Dropout(dropout)
        )
        self.component_projection = nn.Sequential(
            nn.Linear(self.component_dim,d_model), nn.SiLU(), nn.Linear(d_model,d_model)
        )

        # center value/mask + spectral summary value/avail/conf + deep summary +
        # component band missing fraction/count + 6 legacy mask-geometry values.
        self.scalar_projection = nn.Sequential(
            nn.Linear(14,d_model), nn.SiLU(), nn.Linear(d_model,d_model)
        )
        self.wavelength_projection = nn.Sequential(
            nn.Linear(9,d_model), nn.SiLU(), nn.Linear(d_model,d_model)
        )
        self.learned_band_embedding = nn.Parameter(torch.randn(1,self.n_bands,d_model)*0.02)
        self.pre_norm = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,nhead=n_heads,dim_feedforward=d_model*ff_mult,
            dropout=dropout,activation="gelu",batch_first=True,norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer,num_layers=n_layers,norm=nn.LayerNorm(d_model))
        self.local_spectral = nn.Sequential(
            nn.Conv1d(d_model,d_model,7,padding=3,bias=False),
            nn.GroupNorm(_group_count(d_model),d_model), nn.SiLU(),
            nn.Conv1d(d_model,d_model,5,padding=2,bias=False),
            nn.GroupNorm(_group_count(d_model),d_model), nn.SiLU(),
        )

        self.expert_embedding = nn.Parameter(torch.randn(1,1,3,d_model)*0.02)
        # preview value/known/available/confidence + geometry6 + band missing frac/count
        self.candidate_metadata_projection = nn.Sequential(
            nn.Linear(12,d_model), nn.SiLU(), nn.Linear(d_model,d_model)
        )
        self.error_estimator = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model,160), nn.SiLU(), nn.Dropout(dropout), nn.Linear(160,1)
        )
        self.adjustment_head = nn.Sequential(
            nn.Linear(d_model+6+self.component_dim,160), nn.SiLU(), nn.Dropout(dropout), nn.Linear(160,3)
        )
        self.expert_bias = nn.Parameter(torch.zeros(3))

    def _wavelength_features(self,batch,device,dtype):
        w=self.wavelength_norm.to(device=device,dtype=dtype)
        f=[w]
        for q in (1.,2.,4.,8.):
            f.extend([torch.sin(math.pi*q*w),torch.cos(math.pi*q*w)])
        return torch.stack(f,dim=-1).unsqueeze(0).expand(batch,-1,-1)

    def forward(self, x, spectral_candidate, spectral_availability,
                spectral_confidence, geometry, deep_candidate,
                component_band_fraction, component_band_count,
                component_features):
        batch=x.shape[0]; cy=x.shape[-2]//2; cx=x.shape[-1]//2
        spatial=self.spatial_context(self.spatial_down2(self.spatial_down1(self.spatial_stem(x))))
        comp=self.component_projection(component_features)
        context=(spatial+comp).unsqueeze(1)

        center_values=x[:,:self.n_bands,cy,cx]
        center_mask=x[:,self.n_bands:,cy,cx]
        scalar=torch.cat([
            center_values.unsqueeze(-1), center_mask.unsqueeze(-1),
            spectral_candidate.unsqueeze(-1), spectral_availability.unsqueeze(-1),
            spectral_confidence.unsqueeze(-1), deep_candidate.unsqueeze(-1),
            component_band_fraction.unsqueeze(-1), component_band_count.unsqueeze(-1),
            geometry,
        ],dim=-1)
        tokens=self.scalar_projection(scalar)
        tokens=tokens+self.wavelength_projection(self._wavelength_features(batch,x.device,x.dtype))
        tokens=tokens+self.learned_band_embedding+context
        tokens=self.transformer(self.pre_norm(tokens))
        tokens=tokens+self.local_spectral(tokens.transpose(1,2)).transpose(1,2)

        ones=torch.ones_like(spectral_availability); zeros=torch.zeros_like(spectral_candidate)
        preview_value=torch.stack([spectral_candidate,deep_candidate,zeros],dim=-1)
        preview_known=torch.stack([spectral_availability,ones,zeros],dim=-1)
        expert_available=torch.stack([spectral_availability,ones,ones],dim=-1)
        expert_conf=torch.stack([spectral_confidence,ones,ones],dim=-1)
        geom=geometry.unsqueeze(2).expand(-1,-1,3,-1)
        bfrac=component_band_fraction.unsqueeze(-1).unsqueeze(2).expand(-1,-1,3,-1)
        bcount=component_band_count.unsqueeze(-1).unsqueeze(2).expand(-1,-1,3,-1)
        meta=torch.cat([
            preview_value.unsqueeze(-1),preview_known.unsqueeze(-1),
            expert_available.unsqueeze(-1),expert_conf.unsqueeze(-1),geom,bfrac,bcount
        ],dim=-1)
        state=tokens.unsqueeze(2)+self.expert_embedding+self.candidate_metadata_projection(meta)
        raw=self.error_estimator(state).squeeze(-1)
        floor=float(getattr(config,"SPARSE_ROUTER_ERROR_FLOOR",1e-4))
        predicted_error=F.softplus(raw)+floor

        comp_expand=component_features.unsqueeze(1).expand(-1,self.n_bands,-1)
        adj=torch.tanh(self.adjustment_head(torch.cat([tokens,geometry,comp_expand],dim=-1))) * float(getattr(config,"THREE_ROUTER_BIAS_SCALE",0.35))
        score=-float(getattr(config,"THREE_ROUTER_UNCERTAINTY_SCORE_WEIGHT",1.40))*torch.log(predicted_error.clamp_min(floor))
        score=score+adj+self.expert_bias.view(1,1,-1)
        score[:,:,0]=score[:,:,0]+float(getattr(config,"THREE_ROUTER_SPECTRAL_CONFIDENCE_WEIGHT",0.35))*torch.log(spectral_confidence.clamp_min(1e-4))
        score=score.masked_fill(expert_available<0.5,-1e4)
        temp=float(getattr(config,"THREE_ROUTER_TEMPERATURE",0.10))
        probs=torch.softmax(score/temp,dim=-1)
        return {
            "predicted_candidate_error":predicted_error,
            "routing_logits":score,
            "routing_probabilities":probs,
            "hard_route_index":score.argmax(dim=-1),
            "candidate_availability":expert_available,
            "candidate_confidence":expert_conf,
        }
