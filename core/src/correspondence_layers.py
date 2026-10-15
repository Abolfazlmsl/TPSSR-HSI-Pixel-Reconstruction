"""TPSSR Topology-E2E-Corr: direct Topology reconstruction with native-resolution correspondence transport.

This is an internal Topology redesign, not a post-processing correction.

Key change from Topology-E2E v1
--------------------------
The old Topology-E2E radiometric base was dominated by 9/17/33-pixel masked box
means. That is a robust smooth fallback, but it can average away thin dark or
bright laminae before the synthesis trunk sees them.

Topology-E2E-Corr keeps that branch only as a fallback. For missing pixels it also
learns native-resolution correspondence to intact pixels around the hole and
transports their *observed target-band radiometry* directly into the hole.
A learned gate inside Topology decides where transport should dominate. The final
reconstruction is still synthesized by Topology in one forward pass.

Contract
--------
* Frozen Original-Geometry Stage-A supplies structural guidance only.
* Geometry final reconstruction is never used.
* No Refinement, bridge, selector, proposal bank, explicit trace, or post-fill overlay.
* Correspondence operates at native patch resolution; no down/up sampling in
  the transport branch.
* Transport values come only from healthy observed target-band pixels.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.structure_layers import (
    ResBlock,
    DirectionalBank,
    GeometryConditionedBlock,
    _groups,
    _masked_box_mean,
)


class NativeCorrespondenceTransport(nn.Module):
    """Query missing pixels against intact native-resolution context.

    The candidate bank is a wide healthy ring around the synthetic hole. Each
    attention head retrieves observed target-band values from that bank. A
    weak Geometry-orientation/distance bias only regularizes the learned similarity;
    it does not create an explicit path.
    """

    def __init__(
        self,
        in_ch: int,
        embed_ch: int = 64,
        heads: int = 4,
        topk: int = 12,
        candidate_radius: int = 20,
        temperature: float = 0.30,
        sparse_mode: bool = False,
        oracle_topk: int = 3,
        oracle_temperature: float = 0.12,
        mix_temperature: float = 0.50,
        hard_fraction: float = 0.35,
    ):
        super().__init__()
        if embed_ch % heads != 0:
            raise ValueError("embed_ch must be divisible by heads")
        self.embed_ch = int(embed_ch)
        self.heads = int(heads)
        self.head_dim = self.embed_ch // self.heads
        self.topk = int(topk)
        self.candidate_radius = int(candidate_radius)
        self.temperature = float(temperature)
        self.sparse_mode = bool(sparse_mode)
        self.oracle_topk = int(oracle_topk)
        self.oracle_temperature = float(oracle_temperature)
        self.mix_temperature = float(mix_temperature)
        self.hard_fraction = float(hard_fraction)

        self.pre = nn.Sequential(
            nn.Conv2d(in_ch, embed_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(embed_ch), embed_ch),
            nn.SiLU(),
            ResBlock(embed_ch, embed_ch, 1, 0.01),
            ResBlock(embed_ch, embed_ch, 2, 0.01),
        )
        self.query = nn.Conv2d(embed_ch, embed_ch, 1, bias=False)
        self.key = nn.Conv2d(embed_ch, embed_ch, 1, bias=False)
        self.head_mix = nn.Conv2d(embed_ch, self.heads, 1)

        # Positive after softplus. Kept modest initially so Geometry only nudges the
        # learned correspondence rather than dictating a hand-written path.
        self.orientation_strength_raw = nn.Parameter(torch.tensor(-0.65))
        self.distance_strength_raw = nn.Parameter(torch.tensor(-1.10))

    def forward(
        self,
        features: torch.Tensor,
        observed_target: torch.Tensor,
        observed_low: torch.Tensor,
        mask: torch.Tensor,
        spatial_valid: torch.Tensor,
        orientation: torch.Tensor,
        supervision_target: torch.Tensor | None = None,
        supervision_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z = self.pre(features)
        qmap = self.query(z)
        kmap = self.key(z)
        mixmap = self.head_mix(z)

        n, _, h, w = z.shape
        transport = observed_target.clone()
        transport_low = observed_low.clone()
        confidence = torch.zeros_like(observed_target)
        entropy = torch.zeros_like(observed_target)
        donor_rank_num = z.sum() * 0.0
        donor_rank_den = z.new_tensor(0.0)
        donor_hit_num = z.new_tensor(0.0)
        donor_hit_den = z.new_tensor(0.0)
        target_low_full = None
        target_high_full = None
        if supervision_target is not None:
            target_low_full = F.avg_pool2d(supervision_target, 5, stride=1, padding=2)
            target_high_full = supervision_target - target_low_full

        radius = min(self.candidate_radius, max(h, w) - 1)
        kernel = 2 * radius + 1
        if kernel <= 1:
            dilated = mask
        else:
            dilated = F.max_pool2d(mask, kernel, stride=1, padding=radius)

        healthy = (spatial_valid > 0.5) & (mask < 0.5)
        ring = (dilated > 0.01) & healthy

        orient_strength = F.softplus(self.orientation_strength_raw)
        dist_strength = F.softplus(self.distance_strength_raw)
        scale = 1.0 / math.sqrt(float(self.head_dim))

        for b in range(n):
            q_idx = torch.nonzero(mask[b, 0] > 0.5, as_tuple=False)
            if q_idx.numel() == 0:
                continue

            s_idx = torch.nonzero(ring[b, 0], as_tuple=False)
            if s_idx.numel() == 0:
                s_idx = torch.nonzero(healthy[b, 0], as_tuple=False)
            if s_idx.numel() == 0:
                # Pathological patch: retain the smooth observed-low fallback.
                qy, qx = q_idx[:, 0], q_idx[:, 1]
                transport[b, 0, qy, qx] = observed_low[b, 0, qy, qx]
                continue

            qy, qx = q_idx[:, 0], q_idx[:, 1]
            sy, sx = s_idx[:, 0], s_idx[:, 1]
            q_count, s_count = int(q_idx.shape[0]), int(s_idx.shape[0])

            q = qmap[b, :, qy, qx].transpose(0, 1).reshape(q_count, self.heads, self.head_dim)
            k = kmap[b, :, sy, sx].transpose(0, 1).reshape(s_count, self.heads, self.head_dim)
            q = F.normalize(q, dim=-1, eps=1e-6)
            k = F.normalize(k, dim=-1, eps=1e-6)

            # [heads, queries, sources]
            scores = torch.einsum("qhd,shd->hqs", q, k) * scale

            # Weak geometry prior: favour nearby intact pixels lying along the
            # Geometry edge tangent. Geometry orientation encodes gradient orientation as
            # [cos(2theta), sin(2theta)], so the tangent has the negated pair.
            dy = (sy.to(q.dtype)[None, :] - qy.to(q.dtype)[:, None]) / max(float(h - 1), 1.0)
            dx = (sx.to(q.dtype)[None, :] - qx.to(q.dtype)[:, None]) / max(float(w - 1), 1.0)
            dist = torch.sqrt(dx.square() + dy.square() + 1e-8)
            r2 = dx.square() + dy.square() + 1e-8
            cos2phi = (dx.square() - dy.square()) / r2
            sin2phi = (2.0 * dx * dy) / r2
            oq = orientation[b, :, qy, qx].transpose(0, 1).to(q.dtype)
            tangent_c2 = -oq[:, 0:1]
            tangent_s2 = -oq[:, 1:2]
            align = cos2phi * tangent_c2 + sin2phi * tangent_s2
            scores = scores + orient_strength * align.unsqueeze(0) - dist_strength * dist.unsqueeze(0)

            # Training-only donor supervision. GT is used only to build a ranking
            # target for the attention scores; it never enters transported values
            # or the reconstruction path. This makes correspondence learn *which*
            # healthy pixels are valid donors instead of averaging many plausible
            # candidates.
            if supervision_target is not None and target_high_full is not None:
                q_target = supervision_target[b, 0, qy, qx].to(q.dtype)
                q_high = target_high_full[b, 0, qy, qx].to(q.dtype)
                src_value = observed_target[b, 0, sy, sx].to(q.dtype)
                src_high = (observed_target[b, 0, sy, sx] - observed_low[b, 0, sy, sx]).to(q.dtype)
                value_cost = (src_value[None, :] - q_target[:, None]).abs()
                high_cost = (src_high[None, :] - q_high[:, None]).abs()
                active_pol = (q_high.abs() > 0.08).to(q.dtype)[:, None]
                polarity_mismatch = ((q_high[:, None] * src_high[None, :]) < 0).to(q.dtype) * active_pol
                geom_cost = 0.5 * (1.0 - align).clamp(0.0, 2.0)
                oracle_cost = value_cost + 0.45 * high_cost + 0.30 * polarity_mismatch + 0.15 * geom_cost + 0.05 * dist
                ok = min(max(self.oracle_topk, 1), s_count)
                oracle_cost_top, oracle_idx = torch.topk(oracle_cost, k=ok, dim=-1, largest=False, sorted=False)
                oracle_w = torch.softmax(-oracle_cost_top / max(self.oracle_temperature, 1e-4), dim=-1).detach()
                logp_all = torch.log_softmax(scores / max(self.oracle_temperature, 1e-4), dim=-1)
                oracle_idx_h = oracle_idx.unsqueeze(0).expand(self.heads, -1, -1)
                chosen_logp = torch.gather(logp_all, 2, oracle_idx_h)
                rank_q = -(chosen_logp * oracle_w.unsqueeze(0)).sum(-1).mean(0)
                rank_q = rank_q / max(math.log(float(max(s_count, 2))), 1.0)
                if supervision_weight is not None:
                    qw = supervision_weight[b, 0, qy, qx].to(q.dtype).detach()
                else:
                    qw = torch.ones_like(rank_q)
                donor_rank_num = donor_rank_num + (rank_q * qw).sum()
                donor_rank_den = donor_rank_den + qw.sum()
                pred_top1 = scores.argmax(dim=-1)
                hit_q = (pred_top1.unsqueeze(-1) == oracle_idx.unsqueeze(0)).any(-1).to(q.dtype).mean(0)
                donor_hit_num = donor_hit_num + (hit_q * qw).sum()
                donor_hit_den = donor_hit_den + qw.sum()

            kk = min(self.topk, s_count)
            top_scores, top_idx = torch.topk(scores, k=kk, dim=-1, largest=True, sorted=False)
            attn = torch.softmax(top_scores / max(self.temperature, 1e-4), dim=-1)

            source_value = observed_target[b, 0, sy, sx]
            source_low = observed_low[b, 0, sy, sx]
            gather_value = source_value[top_idx]
            gather_low = source_low[top_idx]
            soft_value = (attn * gather_value).sum(-1)
            soft_low = (attn * gather_low).sum(-1)
            if self.sparse_mode and self.hard_fraction > 0.0:
                best_local = attn.argmax(dim=-1, keepdim=True)
                hard_value = torch.gather(gather_value, -1, best_local).squeeze(-1)
                hard_low = torch.gather(gather_low, -1, best_local).squeeze(-1)
                frac = min(max(self.hard_fraction, 0.0), 1.0)
                soft_value = (1.0 - frac) * soft_value + frac * hard_value
                soft_low = (1.0 - frac) * soft_low + frac * hard_low
            per_head_value = soft_value.transpose(0, 1)  # [Q,H]
            per_head_low = soft_low.transpose(0, 1)

            mix_temp = self.mix_temperature if self.sparse_mode else 1.0
            mix = torch.softmax(mixmap[b, :, qy, qx].transpose(0, 1) / max(mix_temp, 1e-4), dim=-1)
            q_transport = (mix * per_head_value).sum(-1)
            q_low = (mix * per_head_low).sum(-1)
            q_conf = (mix * attn.max(dim=-1).values.transpose(0, 1)).sum(-1)

            if kk > 1:
                q_entropy_h = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(-1) / math.log(float(kk))
                q_entropy = (mix * q_entropy_h.transpose(0, 1)).sum(-1)
            else:
                q_entropy = torch.zeros_like(q_conf)

            transport[b, 0, qy, qx] = q_transport
            transport_low[b, 0, qy, qx] = q_low
            confidence[b, 0, qy, qx] = q_conf
            entropy[b, 0, qy, qx] = q_entropy

        return {
            "transport": transport,
            "transport_low": transport_low,
            "transport_high": transport - transport_low,
            "confidence": confidence,
            "entropy": entropy,
            "feature": z,
            "donor_rank_loss": donor_rank_num / donor_rank_den.clamp_min(1.0),
            "donor_oracle_hit": donor_hit_num / donor_hit_den.clamp_min(1.0),
        }


class EndToEndCorrespondenceTopologyFiller(nn.Module):
    """Direct Topology filler with learned native-resolution radiometry transport."""

    ARCHITECTURE_VERSION = "TPSSR_Topology_E2E_NATIVE_CORRESPONDENCE_TRANSPORT_V2"

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
        transport_embed_ch: int = 64,
        transport_heads: int = 4,
        transport_topk: int = 12,
        transport_radius: int = 20,
        transport_temperature: float = 0.30,
        sparse_mode: bool = False,
        oracle_topk: int = 3,
        oracle_temperature: float = 0.12,
        mix_temperature: float = 0.50,
        hard_fraction: float = 0.35,
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
        self.transport_embed_ch = int(transport_embed_ch)
        self.transport_heads = int(transport_heads)
        self.transport_topk = int(transport_topk)
        self.transport_radius = int(transport_radius)
        self.transport_temperature = float(transport_temperature)
        self.sparse_mode = bool(sparse_mode)
        self.oracle_topk = int(oracle_topk)
        self.oracle_temperature = float(oracle_temperature)
        self.mix_temperature = float(mix_temperature)
        self.hard_fraction = float(hard_fraction)

        # Keep the proven Topology-E2E backbone names/shapes so its checkpoint can be
        # used only as an initialization. Inference remains a single new model.
        self.spectral = nn.Sequential(
            nn.Conv2d(self.n_bands, spectral_width, 1, bias=False),
            nn.GroupNorm(_groups(spectral_width), spectral_width), nn.SiLU(),
            ResBlock(spectral_width, spectral_width, 1, 0.01),
        )
        self.geometry_adapter = nn.Sequential(
            nn.Conv2d(structure_feature_channels + 4, base_width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(base_width), base_width), nn.SiLU(),
            ResBlock(base_width, base_width, 1, 0.01), ResBlock(base_width, base_width, 2, 0.01),
        )
        directional_in = spectral_width + 1 + 4 + 1 + 1
        self.directional = DirectionalBank(directional_in, base_width)

        context_in = spectral_width + 1 + 1 + 1 + 1 + 1 + 1 + 4 + base_width + base_width
        self.context0 = nn.Sequential(
            nn.Conv2d(context_in, base_width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(base_width), base_width), nn.SiLU(),
            ResBlock(base_width, base_width, 1, 0.02), ResBlock(base_width, base_width, 2, 0.02),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(base_width, mid_width, 4, 2, 1, bias=False),
            nn.GroupNorm(_groups(mid_width), mid_width), nn.SiLU(), ResBlock(mid_width, mid_width, 1, 0.02),
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

        geometry_in = base_width * 4 + 4 + 1 + 1 + 1 + 1
        self.geometry_trunk = nn.Sequential(
            nn.Conv2d(geometry_in, base_width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(base_width), base_width), nn.SiLU(),
            ResBlock(base_width, base_width, 1, 0.01), ResBlock(base_width, base_width, 2, 0.01),
            ResBlock(base_width, base_width, 4, 0.01), ResBlock(base_width, base_width, 2, 0.01),
        )
        self.structure_head = nn.Conv2d(base_width, 1, 1)
        self.dark_head = nn.Conv2d(base_width, 1, 1)
        self.bright_head = nn.Conv2d(base_width, 1, 1)

        # Smooth-context fallback retained only for non-structural/uncertain pixels.
        rad_in = base_width * 3 + 4 + 3 + 3
        self.anchor_weights = nn.Sequential(
            nn.Conv2d(rad_in, 48, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(48), 48), nn.SiLU(), nn.Conv2d(48, 3, 1),
        )
        self.radiometry = nn.Sequential(
            ResBlock(rad_in, base_width, 1, 0.01), ResBlock(base_width, base_width, 2, 0.01),
            ResBlock(base_width, base_width, 4, 0.01), nn.Conv2d(base_width, 1, 1),
        )

        transport_in = spectral_width + base_width * 3 + 4 + 3 + 1 + 1
        self.transport = NativeCorrespondenceTransport(
            transport_in, embed_ch=transport_embed_ch, heads=transport_heads,
            topk=transport_topk, candidate_radius=transport_radius, temperature=transport_temperature,
            sparse_mode=sparse_mode, oracle_topk=oracle_topk, oracle_temperature=oracle_temperature,
            mix_temperature=mix_temperature, hard_fraction=hard_fraction,
        )
        gate_in = base_width * 3 + 4 + 3 + 1 + 1 + 1
        self.transport_gate = nn.Sequential(
            nn.Conv2d(gate_in, 64, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(64), 64), nn.SiLU(),
            ResBlock(64, 64, 1, 0.01), nn.Conv2d(64, 1, 1),
        )

        synthesis_in = spectral_width + base_width * 5 + 4 + 3 + 1 + 1 + 1 + 1
        self.synthesis_stem = nn.Sequential(
            nn.Conv2d(synthesis_in, base_width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(base_width), base_width), nn.SiLU(), ResBlock(base_width, base_width, 1, 0.01),
        )
        self.synthesis_blocks = nn.ModuleList([
            GeometryConditionedBlock(base_width, base_width, d)
            for d in (1, 2, 4, 8, 4, 2, 1, 2, 1, 1)
        ])
        self.detail_head = nn.Sequential(ResBlock(base_width, base_width, 1, 0.01), nn.Conv2d(base_width, 1, 1))

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
        oracle_target: torch.Tensor | None = None,
        oracle_weight: torch.Tensor | None = None,
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

        # Old averaging branch is now explicitly the fallback rather than the
        # primary radiometric estimate.
        rad_features = torch.cat([u0, geometry, directional, smap, anchors, geometry_probs], 1)
        aw = torch.softmax(self.anchor_weights(rad_features), 1)
        anchor = (aw * anchors).sum(1, keepdim=True)
        fallback_delta = self.radiometry_delta_scale * torch.tanh(self.radiometry(rad_features))
        fallback_radiometry = anchor + fallback_delta

        transport_features = torch.cat([
            spec, geometry, directional, geom, smap, geometry_probs, mask, distance,
        ], 1)
        tr = self.transport(
            transport_features, observed_target, observed_low, mask, spatial_valid,
            structure["orientation"], supervision_target=oracle_target, supervision_weight=oracle_weight,
        )
        transported = tr["transport"]
        transport_conf = tr["confidence"]
        transport_entropy = tr["entropy"]

        gate_logit = self.transport_gate(torch.cat([
            geom, geometry, directional, smap, geometry_probs, transport_conf, mask, distance,
        ], 1))
        learned_gate = torch.sigmoid(gate_logit)
        # Structure probability and correspondence confidence are part of the
        # gate *inside Topology*. This prevents a flat-region transport from
        # overwhelming the stable fallback while allowing strong structures to
        # commit to observed donor radiometry.
        # The gate itself is supervised to be high on real structures and low
        # on flat regions. Confidence is therefore only a mild reliability
        # modulation; multiplying by structure probability again would
        # suppress exactly the uncertain thin structures we are trying to fix.
        if self.sparse_mode:
            transport_gate = learned_gate * (0.80 + 0.20 * transport_conf)
        else:
            transport_gate = learned_gate * (0.60 + 0.40 * transport_conf)
        transport_gate = transport_gate.clamp(0.0, 1.0)

        radiometry = (1.0 - transport_gate) * fallback_radiometry + transport_gate * transported

        synth = self.synthesis_stem(torch.cat([
            spec, u0, c0, geometry, directional, geom, smap, geometry_probs,
            radiometry, observed_high, mask, distance,
        ], 1))
        for block in self.synthesis_blocks:
            synth = block(synth, geom)
        raw_detail = self.detail_scale * torch.tanh(self.detail_head(synth))
        # Once correspondence is confident on a structure, do not let an
        # unconstrained residual wash the transported dark/bright amplitude out.
        detail_protect = 1.0 - 0.65 * transport_gate * structure_prob
        detail = raw_detail * detail_protect
        direct_prediction = radiometry + detail
        prediction = observed_target * (1.0 - mask) + direct_prediction * mask

        return {
            "prediction": prediction,
            "direct_prediction": direct_prediction,
            "coarse_prediction": radiometry,
            "radiometry": radiometry,
            "fallback_radiometry": fallback_radiometry,
            "transported": transported,
            "transported_low": tr["transport_low"],
            "transported_high": tr["transport_high"],
            "transport_confidence": transport_conf,
            "transport_entropy": transport_entropy,
            "transport_gate_logit": gate_logit,
            "transport_gate": transport_gate,
            "detail": detail,
            "raw_detail": raw_detail,
            "structure_logit": structure_logit,
            "dark_logit": dark_logit,
            "bright_logit": bright_logit,
            "structure_prob": structure_prob,
            "dark_prob": dark_prob,
            "bright_prob": bright_prob,
            "geometry_feature": geom,
            "anchor": anchor,
            "anchor_weights": aw,
            "donor_rank_loss": tr["donor_rank_loss"],
            "donor_oracle_hit": tr["donor_oracle_hit"],
        }
