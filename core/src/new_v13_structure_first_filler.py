"""TPSSR V13 Structure-First: explicit native-resolution geometry before radiometry.

This is a clean redesign of the V13 hard-region filler.

Design contract
---------------
* Frozen Original-V27 Stage-A is structural guidance only.
* V27 final reconstruction, V16, bridge/trace/selector and post-fill overlays are not used.
* V13 first predicts an explicit native-resolution structure representation:
  support, ridge/core, dark/bright polarity, tangent orientation, thickness and
  signed contrast.
* Radiometry is predicted only after this structure representation exists.
* Correspondence is conditioned by the predicted structure and remains an
  internal V13 mechanism; healthy observed target-band pixels are the only
  transported values.
* The structure decoder itself never downsamples/upsamples. All thin-structure
  predictions are made at native patch resolution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.new_v13_e2e_structure_filler import ResBlock, DirectionalBank, GeometryConditionedBlock, _groups, _masked_box_mean
from src.new_v13_e2e_correspondence_filler import NativeCorrespondenceTransport


class NativeStructureDecoder(nn.Module):
    """All-native-resolution structure decoder with a large receptive field."""
    def __init__(self, in_ch: int, width: int = 112):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            ResBlock(width, width, d, 0.015) for d in (1, 2, 4, 8, 12, 8, 4, 2, 1)
        ])
        self.refine = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width), nn.SiLU(),
            ResBlock(width, width, 1, 0.01),
        )

        self.support_head = nn.Conv2d(width, 1, 1)
        self.core_head = nn.Conv2d(width, 1, 1)
        self.dark_head = nn.Conv2d(width, 1, 1)
        self.bright_head = nn.Conv2d(width, 1, 1)
        self.tangent_head = nn.Conv2d(width, 2, 1)
        self.thickness_head = nn.Conv2d(width, 1, 1)
        self.contrast_mag_head = nn.Conv2d(width, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.stem(x)
        for block in self.blocks:
            z = block(z)
        z = self.refine(z)

        support_logit = self.support_head(z)
        core_logit = self.core_head(z)
        dark_logit = self.dark_head(z)
        bright_logit = self.bright_head(z)
        tangent_raw = self.tangent_head(z)
        tangent = F.normalize(tangent_raw, dim=1, eps=1e-6)
        thickness = torch.sigmoid(self.thickness_head(z))
        contrast_mag = F.softplus(self.contrast_mag_head(z)).clamp_max(4.0)

        support = torch.sigmoid(support_logit)
        core = torch.sigmoid(core_logit)
        dark = torch.sigmoid(dark_logit)
        bright = torch.sigmoid(bright_logit)
        polarity = torch.tanh(bright_logit - dark_logit)
        signed_contrast = polarity * contrast_mag

        return {
            "feature": z,
            "support_logit": support_logit,
            "core_logit": core_logit,
            "dark_logit": dark_logit,
            "bright_logit": bright_logit,
            "support_prob": support,
            "core_prob": core,
            "dark_prob": dark,
            "bright_prob": bright,
            "tangent": tangent,
            "thickness": thickness,
            "contrast_mag": contrast_mag,
            "signed_contrast": signed_contrast,
        }


class StructureFirstV13Filler(nn.Module):
    ARCHITECTURE_VERSION = "TPSSR_V13_STRUCTURE_FIRST_NATIVE_GEOMETRY_V1"

    def __init__(
        self,
        n_bands: int,
        structure_feature_channels: int,
        spectral_width: int = 48,
        base_width: int = 112,
        transport_embed_ch: int = 64,
        transport_heads: int = 4,
        transport_topk: int = 8,
        transport_radius: int = 22,
        transport_temperature: float = 0.18,
        radiometry_delta_scale: float = 1.50,
        detail_scale: float = 1.35,
    ):
        super().__init__()
        self.n_bands = int(n_bands)
        self.structure_feature_channels = int(structure_feature_channels)
        self.spectral_width = int(spectral_width)
        self.base_width = int(base_width)
        self.transport_embed_ch = int(transport_embed_ch)
        self.transport_heads = int(transport_heads)
        self.transport_topk = int(transport_topk)
        self.transport_radius = int(transport_radius)
        self.transport_temperature = float(transport_temperature)
        self.radiometry_delta_scale = float(radiometry_delta_scale)
        self.detail_scale = float(detail_scale)

        self.spectral = nn.Sequential(
            nn.Conv2d(self.n_bands, spectral_width, 1, bias=False),
            nn.GroupNorm(_groups(spectral_width), spectral_width), nn.SiLU(),
            ResBlock(spectral_width, spectral_width, 1, 0.01),
        )
        self.v27_adapter = nn.Sequential(
            nn.Conv2d(structure_feature_channels + 4, base_width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(base_width), base_width), nn.SiLU(),
            ResBlock(base_width, base_width, 1, 0.01),
            ResBlock(base_width, base_width, 2, 0.01),
        )
        directional_in = spectral_width + 1 + 4 + 1 + 1
        self.directional = DirectionalBank(directional_in, base_width)

        native_in = spectral_width + 1 + 1 + 1 + 1 + 1 + 1 + 4 + base_width + base_width
        self.structure_decoder = NativeStructureDecoder(native_in, base_width)

        # Structure-conditioned correspondence. No hard top-1 injection.
        transport_in = spectral_width + base_width * 3 + 4 + 4 + 2 + 1 + 1 + 2
        self.transport = NativeCorrespondenceTransport(
            in_ch=transport_in,
            embed_ch=transport_embed_ch,
            heads=transport_heads,
            topk=transport_topk,
            candidate_radius=transport_radius,
            temperature=transport_temperature,
            sparse_mode=False,
            oracle_topk=3,
            oracle_temperature=0.10,
            mix_temperature=1.0,
            hard_fraction=0.0,
        )

        # Radiometry comes *after* explicit structure. It chooses among observed
        # transport + smooth anchors and then adds a structure-constrained signed
        # contrast term and a bounded residual.
        rad_in = base_width * 3 + 4 + 4 + 2 + 1 + 1 + 1 + 3 + 1 + 1 + 1 + 1 + 1
        self.radiometry_trunk = nn.Sequential(
            nn.Conv2d(rad_in, base_width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(base_width), base_width), nn.SiLU(),
            ResBlock(base_width, base_width, 1, 0.01),
            ResBlock(base_width, base_width, 2, 0.01),
            ResBlock(base_width, base_width, 4, 0.01),
            ResBlock(base_width, base_width, 8, 0.01),
            ResBlock(base_width, base_width, 4, 0.01),
            ResBlock(base_width, base_width, 2, 0.01),
        )
        self.source_mix = nn.Conv2d(base_width, 4, 1)
        self.radiometry_delta = nn.Conv2d(base_width, 1, 1)

        geom_cond_in = base_width + 4 + 2 + 1 + 1
        self.geometry_adapter = nn.Sequential(
            nn.Conv2d(geom_cond_in, base_width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(base_width), base_width), nn.SiLU(),
            ResBlock(base_width, base_width, 1, 0.01),
        )

        synth_in = spectral_width + base_width * 4 + 4 + 4 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 1
        self.synthesis_stem = nn.Sequential(
            nn.Conv2d(synth_in, base_width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(base_width), base_width), nn.SiLU(),
            ResBlock(base_width, base_width, 1, 0.01),
        )
        self.synthesis_blocks = nn.ModuleList([
            GeometryConditionedBlock(base_width, base_width, d)
            for d in (1, 2, 4, 8, 12, 8, 4, 2, 1)
        ])
        self.detail_head = nn.Sequential(
            ResBlock(base_width, base_width, 1, 0.01), nn.Conv2d(base_width, 1, 1)
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
        supervision_target: torch.Tensor | None = None,
        supervision_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        spec = self.spectral(corrupted_cube)
        n, _, h, w = spec.shape
        band_map = band_position.view(n, 1, 1, 1).expand(-1, -1, h, w)
        smap = torch.cat([structure["edge_prob"], structure["orientation"], structure["contrast_hint"]], 1)
        v27 = self.v27_adapter(torch.cat([structure["feature"], smap], 1))
        directional = self.directional(torch.cat([spec, observed_target, smap, mask, distance], 1))

        native = torch.cat([
            spec, observed_target, spatial_valid, valid_fraction, mask, distance, band_map,
            smap, v27, directional,
        ], 1)
        geom = self.structure_decoder(native)

        support = geom["support_prob"]
        core = geom["core_prob"]
        dark = geom["dark_prob"]
        bright = geom["bright_prob"]
        tangent = geom["tangent"]
        thickness = geom["thickness"]
        signed_contrast = geom["signed_contrast"]

        valid = ((1.0 - mask) * spatial_valid).clamp(0.0, 1.0)
        observed_low = _masked_box_mean(observed_target, valid, 5)
        observed_high = (observed_target - observed_low) * valid
        anchors = torch.cat([
            _masked_box_mean(observed_target, valid, 7),
            _masked_box_mean(observed_target, valid, 15),
            _masked_box_mean(observed_target, valid, 31),
        ], 1)

        transport_features = torch.cat([
            spec, geom["feature"], v27, directional, smap,
            support, core, dark, bright, tangent, thickness, signed_contrast,
            mask, distance,
        ], 1)
        tr = self.transport(
            transport_features, observed_target, observed_low, mask, spatial_valid,
            -tangent, supervision_target=supervision_target,
            supervision_weight=supervision_weight,
        )
        transported = tr["transport"]
        tconf = tr["confidence"]
        tent = tr["entropy"]

        pred_maps = torch.cat([support, core, dark, bright], 1)
        rad_features = torch.cat([
            geom["feature"], v27, directional, smap, pred_maps, tangent,
            thickness, signed_contrast, transported, anchors, tconf, tent,
            observed_low, mask, distance,
        ], 1)
        rf = self.radiometry_trunk(rad_features)
        source_logits = self.source_mix(rf)
        source_w = torch.softmax(source_logits, dim=1)
        candidates = torch.cat([transported, anchors], 1)
        base = (source_w * candidates).sum(1, keepdim=True)

        # Geometry-first radiometry: the signed contrast exists before texture.
        structure_weight = (support * (0.30 + 0.70 * core)).clamp(0.0, 1.0)
        bounded_delta = self.radiometry_delta_scale * torch.tanh(self.radiometry_delta(rf))
        radiometry = base + structure_weight * signed_contrast + bounded_delta

        geometry_cond = self.geometry_adapter(torch.cat([
            geom["feature"], pred_maps, tangent, thickness, signed_contrast,
        ], 1))
        synth = self.synthesis_stem(torch.cat([
            spec, geom["feature"], rf, v27, directional, smap, pred_maps, tangent,
            thickness, signed_contrast, radiometry, transported, observed_high, mask, distance,
        ], 1))
        for block in self.synthesis_blocks:
            synth = block(synth, geometry_cond)
        raw_detail = self.detail_scale * torch.tanh(self.detail_head(synth))
        # Preserve the explicit geometry/radiometry on the strongest predicted cores.
        detail = raw_detail * (1.0 - 0.60 * core * support)
        direct_prediction = radiometry + detail
        prediction = observed_target * (1.0 - mask) + direct_prediction * mask

        return {
            "prediction": prediction,
            "direct_prediction": direct_prediction,
            "coarse_prediction": radiometry,
            "radiometry": radiometry,
            "base_radiometry": base,
            "source_weights": source_w,
            "transported": transported,
            "transport_confidence": tconf,
            "transport_entropy": tent,
            "donor_rank_loss": tr["donor_rank_loss"],
            "donor_oracle_hit": tr["donor_oracle_hit"],
            "detail": detail,
            "raw_detail": raw_detail,
            "support_logit": geom["support_logit"],
            "core_logit": geom["core_logit"],
            "dark_logit": geom["dark_logit"],
            "bright_logit": geom["bright_logit"],
            "support_prob": support,
            "core_prob": core,
            "dark_prob": dark,
            "bright_prob": bright,
            "tangent": tangent,
            "thickness": thickness,
            "contrast_mag": geom["contrast_mag"],
            "signed_contrast": signed_contrast,
            "geometry_feature": geom["feature"],
        }
