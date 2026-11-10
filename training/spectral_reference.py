"""Select an explicitly approved spectral source without faking gate acceptance.

The SWIR inference contract is unchanged. For a VNIR experiment whose band
confidence gate was explicitly REJECTED, the conservative source is the
observed-band linear interpolant, not a rejected neural checkpoint. The
backbone diagnostic file is used only for training-set normalization and
checkpoint provenance; its neural weights are NEVER loaded for predictions.
"""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

ACCEPTED_GATE = "accepted_band_gate"
INTERPOLATION = "interpolation_after_rejected_gate"


class FrozenInterpolation(nn.Module):
    """Return the normalized linear interpolation already in channel 1."""
    def forward(self, features):
        if features.ndim != 3 or features.shape[-1] != 4:
            raise ValueError("Spectral interpolation expects [N, bands, 4] features")
        return features[..., 1]


def select_spectral_source_checkpoint(parent, sensor):
    """Return (source checkpoint path, mode); never accept a diagnostic gate."""
    base = Path(parent)
    gate = base / 'spectral_gate' / 'band' / 'best_model.pt'
    if gate.is_file():
        return gate, ACCEPTED_GATE
    if str(sensor).upper() != 'VNIR':
        raise FileNotFoundError(f'Accepted band confidence gate missing: {gate}')
    summary = base / 'spectral_gate' / 'GATE_SUMMARY.txt'
    diagnostic = base / 'spectral_gate' / 'band' / 'diagnostic_best_val.pt'
    if not summary.is_file() or not diagnostic.is_file():
        raise RuntimeError('VNIR has no accepted gate and no completed rejection report')
    report = summary.read_text(encoding='utf-8')
    if not any(line.strip().startswith('band: accepted=False;') for line in report.splitlines()):
        raise RuntimeError('VNIR fallback requires an explicit rejected band-gate validation')
    backbone = base / 'spectral_backbone' / 'hybrid' / 'diagnostic_best_val.pt'
    if not backbone.is_file():
        raise FileNotFoundError('VNIR fallback requires frozen-backbone normalization metadata: ' + str(backbone))
    return backbone, INTERPOLATION


def source_mode_for_path(path):
    path = Path(path)
    if path.parent.name == 'hybrid' and path.name == 'diagnostic_best_val.pt':
        return INTERPOLATION
    if path.parent.name == 'band' and path.name == 'best_model.pt':
        return ACCEPTED_GATE
    raise ValueError(f'Unknown spectral source checkpoint type: {path}')
