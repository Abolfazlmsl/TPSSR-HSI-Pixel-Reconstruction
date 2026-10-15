"""TPSSR Topology-E2E: end-to-end structure-aware full reconstruction.

This is a clean Topology redesign, not a post-processing head.

Contract
--------
* Frozen Original-Geometry Stage-A is used only as structural guidance
  (feature, edge probability, orientation, signed contrast).
* Geometry final reconstruction is never used as an input or baseline.
* No Refinement, bridge, selector, trace, proposal bank, or post-fill correction is used.
* Topology predicts internal structure/polarity/radiometry and synthesizes the final
  missing pixels directly in one forward pass.
* Thin dark/bright structures are represented explicitly inside Topology and are fed
  into the synthesis trunk before the final pixel prediction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for g in (16, 8, 4, 2):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False)
        self.n1 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.skip(x)
        x = F.silu(self.n1(self.c1(x)))
        x = self.drop(x)
        x = self.n2(self.c2(x))
        return F.silu(x + s)


class DirectionalBank(nn.Module):
    """Native-resolution directional context without explicit path heuristics."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        branch = max(out_ch // 6, 8)
        self.h1 = nn.Conv2d(in_ch, branch, (1, 11), padding=(0, 5), bias=False)
        self.v1 = nn.Conv2d(in_ch, branch, (11, 1), padding=(5, 0), bias=False)
        self.d1 = nn.Conv2d(in_ch, branch, (3, 9), padding=(1, 4), bias=False)
        self.d2 = nn.Conv2d(in_ch, branch, (9, 3), padding=(4, 1), bias=False)
        self.iso1 = nn.Conv2d(in_ch, branch, 3, padding=2, dilation=2, bias=False)
        self.iso2 = nn.Conv2d(in_ch, branch, 3, padding=4, dilation=4, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(branch * 6, out_ch, 1, bias=False),
            nn.GroupNorm(_groups(out_ch), out_ch),
            nn.SiLU(),
            ResBlock(out_ch, out_ch, dilation=1, dropout=0.01),
            ResBlock(out_ch, out_ch, dilation=2, dropout=0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.h1(x), self.v1(x), self.d1(x), self.d2(x), self.iso1(x), self.iso2(x)], 1)
        return self.fuse(z)


class GeometryConditionedBlock(nn.Module):
    """Synthesis block modulated by Topology's own learned geometry representation."""
    def __init__(self, channels: int, geometry_ch: int, dilation: int):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.n1 = nn.GroupNorm(_groups(channels), channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(_groups(channels), channels)
        self.film = nn.Conv2d(geometry_ch, channels * 2, 3, padding=1)

    def forward(self, x: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        g, b = self.film(geometry).chunk(2, dim=1)
        g = 0.40 * torch.tanh(g)
        b = 0.40 * torch.tanh(b)
        z = self.n1(self.c1(x))
        z = F.silu(z * (1.0 + g) + b)
        z = self.n2(self.c2(z))
        return F.silu(x + z)


def _masked_box_mean(values: torch.Tensor, valid: torch.Tensor, kernel: int) -> torch.Tensor:
    pad = kernel // 2
    num = F.avg_pool2d(values * valid, kernel, stride=1, padding=pad)
    den = F.avg_pool2d(valid, kernel, stride=1, padding=pad)
    local = num / den.clamp_min(1e-4)
    gn = (values * valid).flatten(2).sum(2, keepdim=True)
    gd = valid.flatten(2).sum(2, keepdim=True).clamp_min(1.0)
    global_mean = (gn / gd).view(values.shape[0], values.shape[1], 1, 1)
    return torch.where(den > 1e-4, local, global_mean)


class EndToEndTopologyFiller(nn.Module):
    """Topology that learns structure + radiometry + texture jointly and fills directly."""

    ARCHITECTURE_VERSION = "TPSSR_Topology_E2E_STRUCTURE_AWARE_DIRECT_FILL_V1"

    def __init__(
        self,
        n_bands: int,
        structure_feature_channels: int,
        spectral_width: int = 48,
        base_width: int = 96,
        mid_width: int = 128,
        bottleneck_width: int = 160,
        radiometry_delta_scale: float = 3.0,
        detail_scale: float = 2.0,
    ):
        super().__init__()
        self.n_bands = int(n_bands)
        self.structure_feature_channels = int(structure_feature_channels)
        self.spectral_width = int(spectral_width)
        self.base_width = int(base_width)
        self.mid_width = int(mid_width)
        self.bottleneck_width = int(bottleneck_width)
        self.radiometry_delta_scale = float(radiometry_delta_scale)
        self.detail_scale = float(detail_scale)

        self.spectral = nn.Sequential(
            nn.Conv2d(self.n_bands, spectral_width, 1, bias=False),
            nn.GroupNorm(_groups(spectral_width), spectral_width),
            nn.SiLU(),
            ResBlock(spectral_width, spectral_width, 1, 0.01),
        )

        self.geometry_adapter = nn.Sequential(
            nn.Conv2d(structure_feature_channels + 4, base_width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(base_width), base_width),
            nn.SiLU(),
            ResBlock(base_width, base_width, 1, 0.01),
            ResBlock(base_width, base_width, 2, 0.01),
        )

        directional_in = spectral_width + 1 + 4 + 1 + 1
        self.directional = DirectionalBank(directional_in, base_width)

        # Multi-scale context sees the corrupted HSI, healthy pixels, mask and Geometry guidance.
        context_in = spectral_width + 1 + 1 + 1 + 1 + 1 + 1 + 4 + base_width + base_width
        self.context0 = nn.Sequential(
            nn.Conv2d(context_in, base_width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(base_width), base_width),
            nn.SiLU(),
            ResBlock(base_width, base_width, 1, 0.02),
            ResBlock(base_width, base_width, 2, 0.02),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(base_width, mid_width, 4, 2, 1, bias=False),
            nn.GroupNorm(_groups(mid_width), mid_width), nn.SiLU(),
            ResBlock(mid_width, mid_width, 1, 0.02),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(mid_width, bottleneck_width, 4, 2, 1, bias=False),
            nn.GroupNorm(_groups(bottleneck_width), bottleneck_width), nn.SiLU(),
        )
        self.bottleneck = nn.Sequential(
            ResBlock(bottleneck_width, bottleneck_width, 1, 0.03),
            ResBlock(bottleneck_width, bottleneck_width, 2, 0.03),
            ResBlock(bottleneck_width, bottleneck_width, 4, 0.03),
            ResBlock(bottleneck_width, bottleneck_width, 8, 0.03),
            ResBlock(bottleneck_width, bottleneck_width, 4, 0.03),
            ResBlock(bottleneck_width, bottleneck_width, 2, 0.03),
        )
        self.up1 = ResBlock(bottleneck_width + mid_width, mid_width, 1, 0.02)
        self.up0 = ResBlock(mid_width + base_width * 3, base_width, 1, 0.02)

        # Internal geometry is part of Topology itself. These are auxiliary predictions,
        # not a post-processing head; the synthesis trunk consumes them directly.
        geometry_in = base_width * 4 + 4 + 1 + 1 + 1 + 1
        self.geometry_trunk = nn.Sequential(
            nn.Conv2d(geometry_in, base_width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(base_width), base_width), nn.SiLU(),
            ResBlock(base_width, base_width, 1, 0.01),
            ResBlock(base_width, base_width, 2, 0.01),
            ResBlock(base_width, base_width, 4, 0.01),
            ResBlock(base_width, base_width, 2, 0.01),
        )
        self.structure_head = nn.Conv2d(base_width, 1, 1)
        self.dark_head = nn.Conv2d(base_width, 1, 1)
        self.bright_head = nn.Conv2d(base_width, 1, 1)

        # Radiometry is predicted before texture, from healthy-context anchors plus learned context.
        rad_in = base_width * 3 + 4 + 3 + 3
        self.anchor_weights = nn.Sequential(
            nn.Conv2d(rad_in, 48, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(48), 48), nn.SiLU(), nn.Conv2d(48, 3, 1),
        )
        self.radiometry = nn.Sequential(
            ResBlock(rad_in, base_width, 1, 0.01),
            ResBlock(base_width, base_width, 2, 0.01),
            ResBlock(base_width, base_width, 4, 0.01),
            nn.Conv2d(base_width, 1, 1),
        )

        synthesis_in = (
            spectral_width + base_width * 5 + 4 + 3 + 1 + 1 + 1 + 1
        )
        self.synthesis_stem = nn.Sequential(
            nn.Conv2d(synthesis_in, base_width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(base_width), base_width), nn.SiLU(),
            ResBlock(base_width, base_width, 1, 0.01),
        )
        self.synthesis_blocks = nn.ModuleList([
            GeometryConditionedBlock(base_width, base_width, d)
            for d in (1, 2, 4, 8, 4, 2, 1, 2, 1, 1)
        ])
        self.detail_head = nn.Sequential(
            ResBlock(base_width, base_width, 1, 0.01),
            nn.Conv2d(base_width, 1, 1),
        )

    def forward(
        self,
        corrupted_cube: torch.Tensor,
        observed_target: torch.Tensor,
        mask: torch.Tensor,
        spatial_valid: torch.Tensor,
        valid_fraction: torch.Tensor,
        distance: torch.Tensor,
        band_position: torch.Tensor,
        structure: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        spec = self.spectral(corrupted_cube)
        n, _, h, w = spec.shape
        band_map = band_position.view(n, 1, 1, 1).expand(-1, -1, h, w)
        smap = torch.cat([structure["edge_prob"], structure["orientation"], structure["contrast_hint"]], 1)
        geometry = self.geometry_adapter(torch.cat([structure["feature"], smap], 1))

        valid = ((1.0 - mask) * spatial_valid).clamp(0.0, 1.0)
        observed_low = _masked_box_mean(observed_target, valid, 5)
        observed_high = (observed_target - observed_low) * valid
        anchors = torch.cat([
            _masked_box_mean(observed_target, valid, 9),
            _masked_box_mean(observed_target, valid, 17),
            _masked_box_mean(observed_target, valid, 33),
        ], 1)

        directional = self.directional(torch.cat([spec, observed_target, smap, mask, distance], 1))
        c0 = self.context0(torch.cat([
            spec, observed_target, spatial_valid, valid_fraction, mask, distance, band_map,
            smap, geometry, directional,
        ], 1))
        c1 = self.down1(c0)
        cz = self.bottleneck(self.down2(c1))
        u1 = F.interpolate(cz, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, c1], 1))
        u0 = F.interpolate(u1, size=c0.shape[-2:], mode="bilinear", align_corners=False)
        u0 = self.up0(torch.cat([u0, c0, geometry, directional], 1))

        geom = self.geometry_trunk(torch.cat([
            u0, c0, geometry, directional, smap, mask, distance, observed_target, observed_high,
        ], 1))
        structure_logit = self.structure_head(geom)
        dark_logit = self.dark_head(geom)
        bright_logit = self.bright_head(geom)
        structure_prob = torch.sigmoid(structure_logit)
        dark_prob = torch.sigmoid(dark_logit)
        bright_prob = torch.sigmoid(bright_logit)
        geometry_probs = torch.cat([structure_prob, dark_prob, bright_prob], 1)

        rad_features = torch.cat([u0, geometry, directional, smap, anchors, geometry_probs], 1)
        aw = torch.softmax(self.anchor_weights(rad_features), 1)
        anchor = (aw * anchors).sum(1, keepdim=True)
        radiometry_delta = self.radiometry_delta_scale * torch.tanh(self.radiometry(rad_features))
        radiometry = anchor + radiometry_delta

        synth = self.synthesis_stem(torch.cat([
            spec, u0, c0, geometry, directional, geom, smap, geometry_probs,
            radiometry, observed_high, mask, distance,
        ], 1))
        for block in self.synthesis_blocks:
            synth = block(synth, geom)
        detail = self.detail_scale * torch.tanh(self.detail_head(synth))
        direct_prediction = radiometry + detail
        prediction = observed_target * (1.0 - mask) + direct_prediction * mask

        return {
            "prediction": prediction,
            "direct_prediction": direct_prediction,
            "coarse_prediction": radiometry,
            "radiometry": radiometry,
            "radiometry_delta": radiometry_delta,
            "detail": detail,
            "structure_logit": structure_logit,
            "dark_logit": dark_logit,
            "bright_logit": bright_logit,
            "structure_prob": structure_prob,
            "dark_prob": dark_prob,
            "bright_prob": bright_prob,
            "geometry_feature": geom,
            "anchor": anchor,
            "anchor_weights": aw,
        }
