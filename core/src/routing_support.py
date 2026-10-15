"""Determine expert eligibility using observed spectra and mask structure.

Routes depend on spectral support and measured neighboring pixels, not on
scenario names, severity labels, masked targets or evaluation metrics.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Fixed routing thresholds; do not calibrate on evaluation results.
MIN_OBSERVED = 2
MIN_LONG_GAP_BANDS = 24
MIN_LONG_GAP_FRACTION = 0.10
MIN_MISSING_FRACTION = 0.12
NEIGHBOR_WINDOW_RADIUS = 1  # eight immediate spatial neighbors
NEIGHBOR_OBSERVED_GAP_FRACTION = 0.90
MIN_SPATIAL_SUPPORT_FRACTION = 0.75
MIN_AVAILABLE_NEIGHBORS = 4


def policy_spec():
    return dict(
        policy="support_long_gap_with_local_observed_neighbors",
        min_observed=MIN_OBSERVED,
        min_long_gap_bands=MIN_LONG_GAP_BANDS,
        min_long_gap_fraction=MIN_LONG_GAP_FRACTION,
        min_missing_fraction=MIN_MISSING_FRACTION,
        neighbor_window_radius=NEIGHBOR_WINDOW_RADIUS,
        neighbor_observed_gap_fraction=NEIGHBOR_OBSERVED_GAP_FRACTION,
        min_spatial_support_fraction=MIN_SPATIAL_SUPPORT_FRACTION,
        min_available_neighbors=MIN_AVAILABLE_NEIGHBORS,
    )


def _longest_target_gap(missing_row, target_row):
    """Longest contiguous unavailable run containing an actual masked target."""
    missing_row = np.asarray(missing_row, dtype=bool)
    target_row = np.asarray(target_row, dtype=bool)
    edges = np.flatnonzero(np.diff(np.pad(missing_row.astype(np.int8), (1, 1))))
    longest = (0, 0)
    for start, end in edges.reshape(-1, 2):
        start, end = int(start), int(end)
        if end - start > longest[1] - longest[0] and np.any(target_row[start:end]):
            longest = (start, end)
    return longest


def expert_eligibility_maps(mask, cube):
    """Return (Hybrid map, Region-no-anchors map, Region-long-gap map, audit).

    Conditions for the *additional* Region route: a long continuous missing
    spectral run *and* local spatial support of observed (unmasked, finite)
    neighbor values at the same band range. This avoids indiscriminately sending
    the interior of spatial/spectro-spatial missing patches to Region.
    """
    mask = np.asarray(mask, dtype=bool)
    cube = np.asarray(cube)
    if mask.shape != cube.shape or mask.ndim != 3:
        raise ValueError("Expected identical 3D mask/cube arrays")
    h, w, bands = mask.shape
    spatial_mask = np.any(mask, axis=2)
    # Never consider a ground-truth sample at a masked location observed.
    finite_observed = (~mask) & np.isfinite(cube)
    observed = np.sum(finite_observed, axis=2, dtype=np.int32)
    missing_counts = bands - observed
    forced_region = spatial_mask & (observed < MIN_OBSERVED)
    candidate = spatial_mask & (observed >= MIN_OBSERVED)
    long_gap_region = np.zeros((h, w), dtype=bool)

    min_len = max(MIN_LONG_GAP_BANDS, int(np.ceil(MIN_LONG_GAP_FRACTION * bands)))
    min_miss = max(min_len, int(np.ceil(MIN_MISSING_FRACTION * bands)))
    candidate_yx = np.argwhere(candidate & (missing_counts >= min_miss))
    with_long_gap = 0
    with_support = 0
    for y, x in candidate_yx:
        y, x = int(y), int(x)
        a, b = _longest_target_gap(~finite_observed[y, x, :], mask[y, x, :])
        if b - a < min_len:
            continue
        with_long_gap += 1
        available = 0
        good = 0
        for yy in range(max(0, y - NEIGHBOR_WINDOW_RADIUS), min(h, y + NEIGHBOR_WINDOW_RADIUS + 1)):
            for xx in range(max(0, x - NEIGHBOR_WINDOW_RADIUS), min(w, x + NEIGHBOR_WINDOW_RADIUS + 1)):
                if yy == y and xx == x:
                    continue
                available += 1
                fraction = float(np.mean(finite_observed[yy, xx, a:b]))
                if fraction >= NEIGHBOR_OBSERVED_GAP_FRACTION:
                    good += 1
        if (available >= MIN_AVAILABLE_NEIGHBORS and
                good / available >= MIN_SPATIAL_SUPPORT_FRACTION):
            long_gap_region[y, x] = True
            with_support += 1
    hybrid = candidate & ~long_gap_region
    if np.any(hybrid & forced_region) or np.any(long_gap_region & forced_region):
        raise AssertionError("Overlapping two-Expert pixel assignments")
    if not np.array_equal(spatial_mask, hybrid | forced_region | long_gap_region):
        raise AssertionError("Missing pixel assignments in routing policy")
    audit = dict(
        masked_pixels=int(spatial_mask.sum()),
        region_no_anchor_pixels=int(forced_region.sum()),
        long_gap_candidate_pixels=int(with_long_gap),
        region_long_gap_pixels=int(with_support),
        hybrid_pixels=int(hybrid.sum()),
        min_gap_length_bands=int(min_len),
    )
    return hybrid, forced_region, long_gap_region, audit
