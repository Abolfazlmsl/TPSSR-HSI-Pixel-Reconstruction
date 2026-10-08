import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def _group_count(channels):
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SE2d(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden = max(
            16,
            channels // reduction,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                kernel_size=1,
            ),
            nn.SiLU(),
            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(
            self.pool(x)
        )


class ResBlock2d(nn.Module):
    def __init__(
        self,
        channels,
        dilation=1,
        dropout=0.0,
    ):
        super().__init__()

        padding = dilation

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
        )

        self.norm1 = nn.GroupNorm(
            _group_count(channels),
            channels,
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
        )

        self.norm2 = nn.GroupNorm(
            _group_count(channels),
            channels,
        )

        self.se = SE2d(channels)

        self.dropout = (
            nn.Dropout2d(dropout)
            if dropout > 0
            else nn.Identity()
        )

    def forward(self, x):
        residual = x

        x = F.silu(
            self.norm1(
                self.conv1(x)
            )
        )

        x = self.dropout(x)

        x = self.norm2(
            self.conv2(x)
        )

        x = self.se(x)

        return F.silu(
            x + residual
        )


class ConfidenceAwareSparseRouter(nn.Module):
    """
    Confidence-aware sparse mixture-of-experts model for hyperspectral reconstruction.

    Candidates:
        0: spectral analytical reconstruction
        1: spatial analytical reconstruction
        2: learned spectral-spatial Transformer reconstruction

    The model predicts the expected reconstruction error of every candidate
    for every spectral band. Forward inference uses one hard route per band.
    During training, a straight-through hard route preserves differentiability.

    Synthetic scenario names and severity labels are never provided to the
    network.
    """

    def __init__(
        self,
        n_bands,
        wavelengths_nm,
        d_model=192,
        n_heads=8,
        n_layers=6,
        ff_mult=4,
        dropout=0.05,
    ):
        super().__init__()

        self.n_bands = int(n_bands)
        self.d_model = int(d_model)

        wavelengths = np.asarray(wavelengths_nm, dtype=np.float32)
        if len(wavelengths) != self.n_bands:
            raise ValueError("Wavelength count does not match n_bands.")

        wave_min = float(np.min(wavelengths))
        wave_max = float(np.max(wavelengths))
        wave_norm = (wavelengths - wave_min) / max(
            wave_max - wave_min,
            1e-6,
        )

        self.register_buffer(
            "wavelength_norm",
            torch.from_numpy(wave_norm.astype(np.float32)),
            persistent=True,
        )

        # Spatial context backbone.
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(2 * n_bands, 96, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(96), 96),
            nn.SiLU(),
            ResBlock2d(96, dilation=1, dropout=0.03),
            ResBlock2d(96, dilation=2, dropout=0.03),
        )

        self.spatial_down1 = nn.Sequential(
            nn.Conv2d(
                96,
                160,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(160), 160),
            nn.SiLU(),
            ResBlock2d(160, dilation=2, dropout=0.04),
            ResBlock2d(160, dilation=4, dropout=0.04),
        )

        self.spatial_down2 = nn.Sequential(
            nn.Conv2d(
                160,
                224,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(224), 224),
            nn.SiLU(),
            ResBlock2d(224, dilation=2, dropout=0.04),
            ResBlock2d(224, dilation=4, dropout=0.04),
        )

        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_context = nn.Sequential(
            nn.Linear(224, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Same 14 per-band scalar inputs used by the earlier hybrid model.
        self.scalar_projection = nn.Sequential(
            nn.Linear(14, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.wavelength_projection = nn.Sequential(
            nn.Linear(9, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.learned_band_embedding = nn.Parameter(
            torch.randn(1, n_bands, d_model) * 0.02
        )

        self.pre_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.local_spectral = nn.Sequential(
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(_group_count(d_model), d_model),
            nn.SiLU(),
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(_group_count(d_model), d_model),
            nn.SiLU(),
        )

        self.deep_residual_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

        # A small bounded adjustment supplements uncertainty-based routing.
        # The error estimator remains the main routing signal.
        self.router_adjustment_head = nn.Sequential(
            nn.Linear(d_model + 6, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

        # Shared candidate error estimator with learned expert identities.
        self.expert_embedding = nn.Parameter(
            torch.randn(1, 1, 3, d_model) * 0.02
        )

        # Candidate metadata:
        # candidate value, availability, confidence, six geometry values.
        self.candidate_metadata_projection = nn.Sequential(
            nn.Linear(9, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.error_estimator = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def _wavelength_features(self, batch_size, device, dtype):
        w = self.wavelength_norm.to(device=device, dtype=dtype)
        features = [w]

        for frequency in (1.0, 2.0, 4.0, 8.0):
            features.append(
                torch.sin(math.pi * frequency * w)
            )
            features.append(
                torch.cos(math.pi * frequency * w)
            )

        features = torch.stack(features, dim=-1)

        return features.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

    def _encode_tokens(
        self,
        x,
        experts,
        availability,
        confidence,
        geometry,
    ):
        batch = x.shape[0]
        n_bands = self.n_bands
        cy = x.shape[-2] // 2
        cx = x.shape[-1] // 2

        spatial = self.spatial_stem(x)
        spatial = self.spatial_down1(spatial)
        spatial = self.spatial_down2(spatial)
        spatial = self.spatial_pool(spatial).flatten(1)
        spatial_context = self.spatial_context(
            spatial
        ).unsqueeze(1)

        center_values = x[:, :n_bands, cy, cx]
        center_mask = x[:, n_bands:, cy, cx]

        scalar = torch.cat(
            [
                center_values.unsqueeze(-1),
                center_mask.unsqueeze(-1),
                experts,
                availability,
                confidence,
                geometry,
            ],
            dim=-1,
        )

        tokens = self.scalar_projection(scalar)
        tokens = tokens + self.wavelength_projection(
            self._wavelength_features(
                batch,
                x.device,
                x.dtype,
            )
        )
        tokens = tokens + self.learned_band_embedding
        tokens = tokens + spatial_context
        tokens = self.pre_norm(tokens)
        tokens = self.transformer(tokens)

        local = self.local_spectral(
            tokens.transpose(1, 2)
        ).transpose(1, 2)

        return tokens + local

    def _deep_candidate(
        self,
        tokens,
        experts,
        availability,
        confidence,
    ):
        analytical_weight = availability * confidence
        analytical_den = analytical_weight.sum(dim=-1)

        analytical_base = (
            (experts * analytical_weight).sum(dim=-1)
            / analytical_den.clamp_min(1e-6)
        )

        analytical_base = torch.where(
            analytical_den > 0,
            analytical_base,
            torch.zeros_like(analytical_base),
        )

        residual = self.deep_residual_head(
            tokens
        ).squeeze(-1)

        return analytical_base + residual

    def _estimate_candidate_error(
        self,
        tokens,
        candidates,
        candidate_availability,
        candidate_confidence,
        geometry,
    ):
        batch, bands, _ = candidates.shape

        geometry_expanded = geometry.unsqueeze(2).expand(
            -1,
            -1,
            3,
            -1,
        )

        metadata = torch.cat(
            [
                candidates.unsqueeze(-1),
                candidate_availability.unsqueeze(-1),
                candidate_confidence.unsqueeze(-1),
                geometry_expanded,
            ],
            dim=-1,
        )

        candidate_state = (
            tokens.unsqueeze(2)
            + self.expert_embedding
            + self.candidate_metadata_projection(metadata)
        )

        raw = self.error_estimator(
            candidate_state
        ).squeeze(-1)

        predicted_error = (
            F.softplus(raw)
            + config.SPARSE_ROUTER_ERROR_FLOOR
        )

        return predicted_error

    def forward(
        self,
        x,
        experts,
        availability,
        confidence,
        geometry,
    ):
        tokens = self._encode_tokens(
            x,
            experts,
            availability,
            confidence,
            geometry,
        )

        deep_prediction = self._deep_candidate(
            tokens,
            experts,
            availability,
            confidence,
        )

        candidates = torch.cat(
            [
                experts,
                deep_prediction.unsqueeze(-1),
            ],
            dim=-1,
        )

        deep_available = torch.ones_like(
            availability[:, :, :1]
        )
        candidate_availability = torch.cat(
            [availability, deep_available],
            dim=-1,
        )

        # Deep confidence is neutral. Analytical confidence is explicitly
        # supplied by the geometry-derived experts.
        deep_confidence = torch.ones_like(
            confidence[:, :, :1]
        )
        candidate_confidence = torch.cat(
            [confidence, deep_confidence],
            dim=-1,
        )

        predicted_error = self._estimate_candidate_error(
            tokens,
            candidates,
            candidate_availability,
            candidate_confidence,
            geometry,
        )

        router_input = torch.cat(
            [tokens, geometry],
            dim=-1,
        )

        bounded_adjustment = (
            torch.tanh(
                self.router_adjustment_head(
                    router_input
                )
            )
            * config.SPARSE_ROUTER_BIAS_SCALE
        )

        uncertainty_score = (
            -config.SPARSE_ROUTER_UNCERTAINTY_SCORE_WEIGHT
            * torch.log(
                predicted_error.clamp_min(
                    config.SPARSE_ROUTER_ERROR_FLOOR
                )
            )
        )

        analytical_prior = torch.zeros_like(
            uncertainty_score
        )
        analytical_prior[:, :, :2] = (
            config.SPARSE_ROUTER_ANALYTICAL_CONFIDENCE_WEIGHT
            * torch.log(
                confidence.clamp_min(1e-4)
            )
        )

        routing_logits = (
            uncertainty_score
            + bounded_adjustment
            + analytical_prior
        )

        routing_logits = routing_logits.masked_fill(
            candidate_availability < 0.5,
            -1e4,
        )

        routing_probabilities = torch.softmax(
            routing_logits
            / config.SPARSE_ROUTER_TEMPERATURE,
            dim=-1,
        )

        hard_index = routing_logits.argmax(dim=-1)
        hard_weights = F.one_hot(
            hard_index,
            num_classes=3,
        ).to(routing_probabilities.dtype)

        if self.training:
            routing_weights = (
                hard_weights
                + routing_probabilities
                - routing_probabilities.detach()
            )
        else:
            routing_weights = hard_weights

        prediction = (
            routing_weights * candidates
        ).sum(dim=-1)

        return prediction, {
            "deep_prediction": deep_prediction,
            "candidates": candidates,
            "candidate_availability": candidate_availability,
            "candidate_confidence": candidate_confidence,
            "predicted_candidate_error": predicted_error,
            "routing_logits": routing_logits,
            "routing_probabilities": routing_probabilities,
            "routing_weights": routing_weights,
            "hard_route_index": hard_index,
        }
