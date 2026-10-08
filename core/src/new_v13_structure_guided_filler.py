"""TPSSR New V13: V27-guided texture-preserving region filler.

This module intentionally does NOT modify the frozen V27 structure predictor.
It decomposes reconstruction into:
  1) low-frequency/radiometric reconstruction,
  2) native-resolution high-frequency texture reconstruction,
then fuses the two only inside the missing region.

The texture path never downsamples, so fine laminae do not need to pass through
an interpolation bottleneck.
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


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)


def _masked_box_mean(values: torch.Tensor, valid: torch.Tensor, kernel: int) -> torch.Tensor:
    pad = kernel // 2
    num = F.avg_pool2d(values * valid, kernel, stride=1, padding=pad)
    den = F.avg_pool2d(valid, kernel, stride=1, padding=pad)
    local = num / den.clamp_min(1e-4)

    # Pixels deep inside a large hole may have no observed samples in a small
    # window. Fall back to the observed patch mean without using hidden values.
    global_num = (values * valid).flatten(2).sum(dim=2, keepdim=True)
    global_den = valid.flatten(2).sum(dim=2, keepdim=True).clamp_min(1.0)
    global_mean = (global_num / global_den).view(values.shape[0], values.shape[1], 1, 1)
    return torch.where(den > 1e-4, local, global_mean)


class NewV13StructureGuidedFiller(nn.Module):
    """One-band filler conditioned on frozen V27 Stage-A predictions.

    Inputs follow the original V27 convention. ``structure`` must contain:
      edge_prob:      [N,1,H,W]
      orientation:    [N,2,H,W]
      contrast_hint:  [N,1,H,W]
      feature:        [N,Cs,H,W]

    The model predicts one normalized target band at a time, matching V27's
    whole-spectrum-hole training/evaluation setup.
    """

    ARCHITECTURE_VERSION = "TPSSR_NEW_V13_V27_GUIDED_LF_HF_NATIVE_TEXTURE_V1"

    def __init__(
        self,
        n_bands: int,
        structure_feature_channels: int,
        spectral_width: int = 32,
        base_width: int = 64,
        mid_width: int = 96,
        bottleneck_width: int = 128,
        hf_residual_scale: float = 1.50,
        low_delta_scale: float = 2.50,
    ):
        super().__init__()
        self.n_bands = int(n_bands)
        self.structure_feature_channels = int(structure_feature_channels)
        self.spectral_width = int(spectral_width)
        self.base_width = int(base_width)
        self.mid_width = int(mid_width)
        self.bottleneck_width = int(bottleneck_width)
        self.hf_residual_scale = float(hf_residual_scale)
        self.low_delta_scale = float(low_delta_scale)

        self.spectral_compressor = nn.Sequential(
            nn.Conv2d(self.n_bands, spectral_width, 1, bias=False),
            nn.GroupNorm(_groups(spectral_width), spectral_width),
            nn.SiLU(),
            nn.Conv2d(spectral_width, spectral_width, 1, bias=False),
            nn.GroupNorm(_groups(spectral_width), spectral_width),
            nn.SiLU(),
        )

        # Keep V27's full-resolution correspondence feature available to both
        # branches. This is the structure path; it is not downsampled first.
        self.structure_adapter = nn.Sequential(
            nn.Conv2d(structure_feature_channels + 4, base_width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(base_width), base_width),
            nn.SiLU(),
            ResidualBlock(base_width, base_width, dilation=1, dropout=0.01),
        )

        # Spectral context + observed target + geometry + V27 maps + three
        # leak-free radiometric anchors + observed high-pass + band position.
        global_in = spectral_width + 1 + 1 + 1 + 1 + 4 + 3 + 1 + 1
        self.global_stem = nn.Sequential(
            nn.Conv2d(global_in, base_width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(base_width), base_width),
            nn.SiLU(),
            ResidualBlock(base_width, base_width, dropout=0.02),
        )
        self.global_down1 = nn.Sequential(
            nn.Conv2d(base_width, mid_width, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(mid_width), mid_width),
            nn.SiLU(),
        )
        self.global_enc1 = ResidualBlock(mid_width, mid_width, dilation=1, dropout=0.02)
        self.global_down2 = nn.Sequential(
            nn.Conv2d(mid_width, bottleneck_width, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(bottleneck_width), bottleneck_width),
            nn.SiLU(),
        )
        self.global_context = nn.Sequential(
            ResidualBlock(bottleneck_width, bottleneck_width, dilation=1, dropout=0.03),
            ResidualBlock(bottleneck_width, bottleneck_width, dilation=2, dropout=0.03),
            ResidualBlock(bottleneck_width, bottleneck_width, dilation=4, dropout=0.03),
            ResidualBlock(bottleneck_width, bottleneck_width, dilation=8, dropout=0.03),
        )
        self.global_up1 = ResidualBlock(bottleneck_width + mid_width, mid_width, dropout=0.02)
        self.global_up0 = ResidualBlock(mid_width + base_width + base_width, base_width, dropout=0.02)

        # Radiometry branch predicts the smooth/low-frequency component. It can
        # make a large correction when the local layer is genuinely dark/bright.
        self.radiometry_head = nn.Sequential(
            ResidualBlock(base_width + 3 + 4, base_width, dilation=1, dropout=0.01),
            ResidualBlock(base_width, base_width, dilation=2, dropout=0.01),
            nn.Conv2d(base_width, 1, 1),
        )
        self.anchor_mix = nn.Sequential(
            nn.Conv2d(base_width + 3 + 4, 32, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(32), 32),
            nn.SiLU(),
            nn.Conv2d(32, 3, 1),
        )

        # Native-resolution texture branch. There is deliberately no stride,
        # pooling, or interpolation anywhere in this path.
        texture_in = spectral_width + base_width + base_width + 4 + 1 + 1 + 1
        self.texture_stem = nn.Sequential(
            nn.Conv2d(texture_in, base_width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(base_width), base_width),
            nn.SiLU(),
        )
        self.texture_body = nn.Sequential(
            ResidualBlock(base_width, base_width, dilation=1, dropout=0.01),
            ResidualBlock(base_width, base_width, dilation=2, dropout=0.01),
            ResidualBlock(base_width, base_width, dilation=4, dropout=0.01),
            ResidualBlock(base_width, base_width, dilation=8, dropout=0.01),
            ResidualBlock(base_width, base_width, dilation=4, dropout=0.01),
            ResidualBlock(base_width, base_width, dilation=2, dropout=0.01),
            ResidualBlock(base_width, base_width, dilation=1, dropout=0.01),
        )
        self.texture_head = nn.Conv2d(base_width, 1, 1)

        # A learned internal texture amplitude is not a post-hoc gate. It only
        # balances the LF/HF decomposition during reconstruction.
        self.texture_amplitude = nn.Sequential(
            nn.Conv2d(base_width + 4 + 1, 32, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(32), 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, 1),
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
        spectral = self.spectral_compressor(corrupted_cube)
        n, _, h, w = spectral.shape

        band_map = band_position.view(n, 1, 1, 1).expand(-1, -1, h, w)
        structure_map = torch.cat(
            [structure["edge_prob"], structure["orientation"], structure["contrast_hint"]],
            dim=1,
        )
        structure_full = self.structure_adapter(torch.cat([structure["feature"], structure_map], dim=1))

        observed_valid = ((1.0 - mask) * spatial_valid).clamp(0.0, 1.0)
        anchor9 = _masked_box_mean(observed_target, observed_valid, 9)
        anchor17 = _masked_box_mean(observed_target, observed_valid, 17)
        anchor33 = _masked_box_mean(observed_target, observed_valid, 33)
        anchors = torch.cat([anchor9, anchor17, anchor33], dim=1)
        observed_low = _masked_box_mean(observed_target, observed_valid, 5)
        observed_high = (observed_target - observed_low) * observed_valid

        global_input = torch.cat(
            [
                spectral,
                observed_target,
                valid_fraction,
                mask,
                distance,
                structure_map,
                anchors,
                observed_high,
                band_map,
            ],
            dim=1,
        )
        g0 = self.global_stem(global_input)
        g1 = self.global_enc1(self.global_down1(g0))
        gz = self.global_context(self.global_down2(g1))
        u1 = F.interpolate(gz, size=g1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.global_up1(torch.cat([u1, g1], dim=1))
        u0 = F.interpolate(u1, size=g0.shape[-2:], mode="bilinear", align_corners=False)
        u0 = self.global_up0(torch.cat([u0, g0, structure_full], dim=1))

        rad_features = torch.cat([u0, anchors, structure_map], dim=1)
        anchor_weights = torch.softmax(self.anchor_mix(rad_features), dim=1)
        anchor = (anchor_weights * anchors).sum(dim=1, keepdim=True)
        low_delta = self.low_delta_scale * torch.tanh(self.radiometry_head(rad_features))
        low_prediction = anchor + low_delta

        texture_input = torch.cat(
            [
                spectral,
                structure_full,
                u0,
                structure_map,
                observed_high,
                mask,
                distance,
            ],
            dim=1,
        )
        tex = self.texture_body(self.texture_stem(texture_input))
        raw_hf = self.hf_residual_scale * torch.tanh(self.texture_head(tex))
        amp = torch.sigmoid(self.texture_amplitude(torch.cat([tex, structure_map, distance], dim=1)))
        high_frequency = amp * raw_hf

        direct_prediction = low_prediction + high_frequency
        prediction = observed_target * (1.0 - mask) + direct_prediction * mask

        return {
            "prediction": prediction,
            "direct_prediction": direct_prediction,
            "coarse_prediction": low_prediction,
            "low_prediction": low_prediction,
            "high_frequency": high_frequency,
            "texture_amplitude": amp,
            "anchor": anchor,
            "anchor_weights": anchor_weights,
        }
