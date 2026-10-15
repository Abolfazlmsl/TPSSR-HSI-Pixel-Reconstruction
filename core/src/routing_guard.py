"""Require contiguous masked spectral gaps before assigning spatial routing.

The guard distinguishes target loss from pre-existing invalid measurements
and uses only masks and observed values.
"""
from __future__ import annotations
import numpy as np
from .routing_support import (
    expert_eligibility_maps as support_eligibility,
    _longest_target_gap,
    policy_spec as support_policy_spec,
)

# Pre-specified heuristic defaults; NOT fitted to any test metric.
MIN_MASKED_FRACTION_INSIDE_GAP = 0.85
MIN_TARGET_DOMINANCE_IN_GAP = 0.85
MIN_CONTIGUOUS_TRUE_MASK_FRACTION = 0.80


def policy_spec():
    return {
        "policy": "guard_support_plus_true_mask_gap_purity_guard",
        "base_policy": support_policy_spec(),
        "min_masked_fraction_inside_gap": MIN_MASKED_FRACTION_INSIDE_GAP,
        "min_target_dominance_in_gap": MIN_TARGET_DOMINANCE_IN_GAP,
        "min_contiguous_true_mask_fraction": MIN_CONTIGUOUS_TRUE_MASK_FRACTION,
        "validation_status": "exploratory_post_support_test_analysis",
    }


def _longest_true_mask_run(row):
    row = np.asarray(row, dtype=bool)
    edges = np.flatnonzero(np.diff(np.pad(row.astype(np.int8), (1, 1))))
    return max((int(b - a) for a, b in edges.reshape(-1, 2)), default=0)


def expert_eligibility_maps(mask, cube):
    """Returns Hybrid, forced Region, long-gap Region, diagnostics.

    The guard cannot expand initial geometry assignments. All pixels
    with fewer than two actually observed finite bands remain forced-Region.
    """
    mask = np.asarray(mask, dtype=bool)
    cube = np.asarray(cube)
    candidate_hybrid, forced_region, support_long_gap, audit_support = (
        support_eligibility(mask, cube)
    )
    observed = (~mask) & np.isfinite(cube)
    kept_gap = np.zeros_like(support_long_gap, dtype=bool)
    rejected_insufficient_true_mask = 0
    rejected_fragmented_target = 0
    for y, x in np.argwhere(support_long_gap):
        y, x = int(y), int(x)
        start, end = _longest_target_gap(~observed[y, x], mask[y, x])
        gap_len = end - start
        total_target = int(mask[y, x].sum())
        if gap_len <= 0 or total_target <= 0:
            raise AssertionError('Proposed gap has no masked targets')
        target_inside = np.asarray(mask[y, x, start:end], dtype=bool)
        n_target_inside = int(target_inside.sum())

        # Guard A: this is actually a synthetic *missing-band* gap, not just
        # a long run of invalid measurements with a few target entries.
        if (n_target_inside / gap_len < MIN_MASKED_FRACTION_INSIDE_GAP or
                n_target_inside / total_target < MIN_TARGET_DOMINANCE_IN_GAP):
            rejected_insufficient_true_mask += 1
            continue

        # Guard B: the target bands should be one continuous block. Multiple
        # short/disconnected missing segments inside one invalid run are not
        # confidently handled by the spatial gap rule.
        if _longest_true_mask_run(target_inside) / gap_len < MIN_CONTIGUOUS_TRUE_MASK_FRACTION:
            rejected_fragmented_target += 1
            continue
        kept_gap[y, x] = True

    hybrid = candidate_hybrid | (support_long_gap & ~kept_gap)
    spatial_mask = np.any(mask, axis=2)
    if np.any(hybrid & forced_region) or np.any(hybrid & kept_gap):
        raise AssertionError('V2.2 pixel routes overlap')
    if not np.array_equal(spatial_mask, hybrid | forced_region | kept_gap):
        raise AssertionError('V2.2 left pixels without an Expert')
    audit = dict(audit_support)
    audit.update(
        policy="guard_true_mask_gap_purity",
        support_region_long_gap_pixels=int(support_long_gap.sum()),
        guard_long_gap_rejected_non_dominant=int(rejected_insufficient_true_mask),
        guard_long_gap_rejected_incoherent=int(rejected_fragmented_target),
        guard_long_gap_returned_to_hybrid=int((support_long_gap & ~kept_gap).sum()),
        region_long_gap_pixels=int(kept_gap.sum()),
        hybrid_pixels=int(hybrid.sum()),
    )
    assert (int(support_long_gap.sum()) == int(kept_gap.sum())
            + rejected_insufficient_true_mask + rejected_fragmented_target)
    return hybrid, forced_region, kept_gap, audit
