"""TPSSR V13 Topology-First: explicit skeleton topology before radiometry.

Design contract
---------------
* Frozen Original-V27 Stage-A is structural guidance only.
* V27 final reconstruction, V16, bridge/trace/selector/post-fill are not used.
* V13 predicts a native-resolution topology representation before radiometry:
  skeleton/centerline, topology-derived support, tangent orientation, width,
  a single signed polarity (- dark, + bright), and signed contrast.
* Support is not an independent soft blob head. It is differentiably generated
  from the predicted skeleton and width distribution, so geometry stays tied to
  the centerline.
* Correspondence and radiometry are conditioned by this topology. Healthy
  observed target-band pixels are the only transported values.
* No spatial downsample/upsample exists in the topology decoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.new_v13_e2e_structure_filler import (
    ResBlock, DirectionalBank, GeometryConditionedBlock, _groups, _masked_box_mean,
)
from src.new_v13_e2e_correspondence_filler import NativeCorrespondenceTransport


class NativeTopologyDecoder(nn.Module):
    """Native-resolution topology decoder.

    The key difference from Structure-First V1 is representational: a thin
    skeleton is primary. Region support is *derived* from skeleton + predicted
    width instead of being independently painted as a probability blob.
    """
    WIDTH_KERNELS = (3, 5, 9, 13)
    WIDTH_VALUES = (0.15, 0.35, 0.65, 1.00)

    def __init__(self, in_ch: int, width: int = 112):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width), nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            ResBlock(width, width, d, 0.015) for d in (1, 2, 4, 8, 12, 16, 12, 8, 4, 2, 1)
        ])
        self.refine = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width), nn.SiLU(),
            ResBlock(width, width, 1, 0.01),
        )

        self.skeleton_head = nn.Conv2d(width, 1, 1)
        self.polarity_head = nn.Conv2d(width, 1, 1)  # signed: - dark, + bright
        self.tangent_head = nn.Conv2d(width, 2, 1)
        self.width_head = nn.Conv2d(width, len(self.WIDTH_KERNELS), 1)
        self.contrast_mag_head = nn.Conv2d(width, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.stem(x)
        for block in self.blocks:
            z = block(z)
        z = self.refine(z)

        skeleton_logit = self.skeleton_head(z)
        skeleton = torch.sigmoid(skeleton_logit)
        polarity_logit = self.polarity_head(z)
        polarity = torch.tanh(polarity_logit)
        tangent = F.normalize(self.tangent_head(z), dim=1, eps=1e-6)
        width_logits = self.width_head(z)
        width_prob = torch.softmax(width_logits, dim=1)
        width_values = z.new_tensor(self.WIDTH_VALUES).view(1, -1, 1, 1)
        width_expected = (width_prob * width_values).sum(1, keepdim=True)

        # Topology-derived support. Every support pixel must be explainable by
        # a centerline plus a width scale; no independent support head exists.
        dilated = [F.max_pool2d(skeleton, k, 1, k // 2) for k in self.WIDTH_KERNELS]
        support = torch.zeros_like(skeleton)
        for i, d in enumerate(dilated):
            support = support + width_prob[:, i:i+1] * d
        support = support.clamp(0.0, 1.0)

        contrast_mag = F.softplus(self.contrast_mag_head(z)).clamp_max(4.0)
        signed_contrast = polarity * contrast_mag

        # Diagnostic probabilities are mutually exclusive through one signed
        # polarity variable, unlike the old independent dark/bright heads.
        dark_prob = support * torch.sigmoid(-2.5 * polarity_logit)
        bright_prob = support * torch.sigmoid(2.5 * polarity_logit)

        return {
            "feature": z,
            "skeleton_logit": skeleton_logit,
            "skeleton_prob": skeleton,
            "support_prob": support,
            "polarity_logit": polarity_logit,
            "polarity": polarity,
            "dark_prob": dark_prob,
            "bright_prob": bright_prob,
            "tangent": tangent,
            "width_logits": width_logits,
            "width_prob": width_prob,
            "width_expected": width_expected,
            "contrast_mag": contrast_mag,
            "signed_contrast": signed_contrast,
        }


class TopologyFirstV13Filler(nn.Module):
    ARCHITECTURE_VERSION = "TPSSR_V13_TOPOLOGY_FIRST_SKELETON_POLARITY_V1"

    def __init__(
        self,
        n_bands: int,
        structure_feature_channels: int,
        spectral_width: int = 48,
        base_width: int = 112,
        transport_embed_ch: int = 64,
        transport_heads: int = 4,
        transport_topk: int = 8,
        transport_radius: int = 24,
        transport_temperature: float = 0.16,
        radiometry_delta_scale: float = 1.40,
        detail_scale: float = 1.25,
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
        self.topology_decoder = NativeTopologyDecoder(native_in, base_width)

        # spec + three learned contexts + V27 map + topology representation.
        topology_extra = 4 + 1 + 1 + 1 + 2 + 1 + 1 + 4 + 1 + 1
        transport_in = spectral_width + base_width * 3 + topology_extra
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

        # Radiometry is explicitly conditioned on skeleton topology.
        topology_maps_ch = 1 + 1 + 1 + 2 + 1 + 1 + 4  # support, skeleton, polarity, tangent, width, contrast, width bins
        rad_extra = 4 + topology_maps_ch + 1 + 3 + 1 + 1 + 1 + 1 + 1
        rad_in = base_width * 3 + rad_extra
        self.radiometry_trunk = nn.Sequential(
            nn.Conv2d(rad_in, base_width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(base_width), base_width), nn.SiLU(),
            ResBlock(base_width, base_width, 1, 0.01),
            ResBlock(base_width, base_width, 2, 0.01),
            ResBlock(base_width, base_width, 4, 0.01),
            ResBlock(base_width, base_width, 8, 0.01),
            ResBlock(base_width, base_width, 12, 0.01),
            ResBlock(base_width, base_width, 4, 0.01),
            ResBlock(base_width, base_width, 2, 0.01),
        )
        self.source_mix = nn.Conv2d(base_width, 4, 1)
        self.radiometry_delta = nn.Conv2d(base_width, 1, 1)

        geom_cond_in = base_width + topology_maps_ch
        self.geometry_adapter = nn.Sequential(
            nn.Conv2d(geom_cond_in, base_width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(base_width), base_width), nn.SiLU(),
            ResBlock(base_width, base_width, 1, 0.01),
        )

        synth_extra = 4 + topology_maps_ch + 1 + 1 + 1 + 1 + 1
        synth_in = spectral_width + base_width * 4 + synth_extra
        self.synthesis_stem = nn.Sequential(
            nn.Conv2d(synth_in, base_width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(base_width), base_width), nn.SiLU(),
            ResBlock(base_width, base_width, 1, 0.01),
        )
        self.synthesis_blocks = nn.ModuleList([
            GeometryConditionedBlock(base_width, base_width, d)
            for d in (1, 2, 4, 8, 12, 16, 12, 8, 4, 2, 1)
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
        topo = self.topology_decoder(native)

        skeleton = topo["skeleton_prob"]
        support = topo["support_prob"]
        polarity = topo["polarity"]
        tangent = topo["tangent"]
        width_expected = topo["width_expected"]
        width_prob = topo["width_prob"]
        signed_contrast = topo["signed_contrast"]

        valid = ((1.0 - mask) * spatial_valid).clamp(0.0, 1.0)
        observed_low = _masked_box_mean(observed_target, valid, 5)
        observed_high = (observed_target - observed_low) * valid
        anchors = torch.cat([
            _masked_box_mean(observed_target, valid, 7),
            _masked_box_mean(observed_target, valid, 15),
            _masked_box_mean(observed_target, valid, 31),
        ], 1)

        topology_maps = torch.cat([
            support, skeleton, polarity, tangent, width_expected, signed_contrast, width_prob,
        ], 1)
        transport_features = torch.cat([
            spec, topo["feature"], v27, directional, smap, topology_maps, mask, distance,
        ], 1)
        tr = self.transport(
            transport_features, observed_target, observed_low, mask, spatial_valid,
            -tangent, supervision_target=supervision_target,
            supervision_weight=supervision_weight,
        )
        transported = tr["transport"]
        tconf = tr["confidence"]
        tent = tr["entropy"]

        rad_features = torch.cat([
            topo["feature"], v27, directional, smap, topology_maps,
            transported, anchors, tconf, tent, observed_low, mask, distance,
        ], 1)
        rf = self.radiometry_trunk(rad_features)
        source_w = torch.softmax(self.source_mix(rf), dim=1)
        candidates = torch.cat([transported, anchors], 1)
        base = (source_w * candidates).sum(1, keepdim=True)

        topology_weight = (support * (0.20 + 0.80 * skeleton)).clamp(0.0, 1.0)
        bounded_delta = self.radiometry_delta_scale * torch.tanh(self.radiometry_delta(rf))
        radiometry = base + topology_weight * signed_contrast + bounded_delta

        geometry_cond = self.geometry_adapter(torch.cat([topo["feature"], topology_maps], 1))
        synth = self.synthesis_stem(torch.cat([
            spec, topo["feature"], rf, v27, directional, smap, topology_maps,
            radiometry, transported, observed_high, mask, distance,
        ], 1))
        for block in self.synthesis_blocks:
            synth = block(synth, geometry_cond)
        raw_detail = self.detail_scale * torch.tanh(self.detail_head(synth))
        # Keep one/few-pixel topology from being washed away by texture residuals.
        detail = raw_detail * (1.0 - 0.72 * skeleton * support)
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
            "skeleton_logit": topo["skeleton_logit"],
            "skeleton_prob": skeleton,
            "support_prob": support,
            "polarity_logit": topo["polarity_logit"],
            "polarity": polarity,
            "dark_prob": topo["dark_prob"],
            "bright_prob": topo["bright_prob"],
            "tangent": tangent,
            "width_logits": topo["width_logits"],
            "width_prob": width_prob,
            "width_expected": width_expected,
            "contrast_mag": topo["contrast_mag"],
            "signed_contrast": signed_contrast,
            "topology_feature": topo["feature"],
        }
