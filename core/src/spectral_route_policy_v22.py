"""TPSSR 2-Expert V2.2: observed-only TRUE-mask long-gap purity guard.

Start from the exact V2.1 rule. A proposed long-gap Region route is retained
only if its long gap is mostly composed of contiguous, ACTUALLY MASKED target
bands, rather than (for example) a long run of pre-existing non-finite bands
with a few synthetic targets interspersed.

No true masked reflectance, scenario/severity labels or test IDs are used.
The *choice* to add this guard was made after inspecting development-test
results on SWIR5/10. Treat evaluation there as exploratory, not untouched.
"""
from __future__ import annotations
import numpy as np
from .spectral_route_policy_v21 import (
    expert_eligibility_maps as v21_eligibility,
    _longest_target_gap,
    policy_spec as v21_policy_spec,
)

# Pre-specified heuristic defaults; NOT fitted to any test metric.
MIN_MASKED_FRACTION_INSIDE_GAP = 0.85
MIN_TARGET_DOMINANCE_IN_GAP = 0.85
MIN_CONTIGUOUS_TRUE_MASK_FRACTION = 0.80


def policy_spec():
    return {
        "policy": "v22_v21_plus_true_mask_gap_purity_guard",
        "base_policy": v21_policy_spec(),
        "min_masked_fraction_inside_gap": MIN_MASKED_FRACTION_INSIDE_GAP,
        "min_target_dominance_in_gap": MIN_TARGET_DOMINANCE_IN_GAP,
        "min_contiguous_true_mask_fraction": MIN_CONTIGUOUS_TRUE_MASK_FRACTION,
        "validation_status": "exploratory_post_v21_test_analysis",
    }


def _longest_true_mask_run(row):
    row = np.asarray(row, dtype=bool)
    edges = np.flatnonzero(np.diff(np.pad(row.astype(np.int8), (1, 1))))
    return max((int(b - a) for a, b in edges.reshape(-1, 2)), default=0)


def expert_eligibility_maps(mask, cube):
    """Returns Hybrid, forced Region, long-gap Region, diagnostics.

    V2.2 never expands V2.1 Region assignment. In particular, all pixels
    with fewer than two actually observed finite bands remain forced-Region.
    """
    mask = np.asarray(mask, dtype=bool)
    cube = np.asarray(cube)
    candidate_hybrid, forced_region, v21_long_gap, audit_v21 = (
        v21_eligibility(mask, cube)
    )
    observed = (~mask) & np.isfinite(cube)
    kept_gap = np.zeros_like(v21_long_gap, dtype=bool)
    rejected_insufficient_true_mask = 0
    rejected_fragmented_target = 0
    for y, x in np.argwhere(v21_long_gap):
        y, x = int(y), int(x)
        start, end = _longest_target_gap(~observed[y, x], mask[y, x])
        gap_len = end - start
        total_target = int(mask[y, x].sum())
        if gap_len <= 0 or total_target <= 0:
            raise AssertionError('V2.1 gap Region lacked any masked targets')
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
        # confidently handled by the V2.1 spatial Region gap rule.
        if _longest_true_mask_run(target_inside) / gap_len < MIN_CONTIGUOUS_TRUE_MASK_FRACTION:
            rejected_fragmented_target += 1
            continue
        kept_gap[y, x] = True

    hybrid = candidate_hybrid | (v21_long_gap & ~kept_gap)
    spatial_mask = np.any(mask, axis=2)
    if np.any(hybrid & forced_region) or np.any(hybrid & kept_gap):
        raise AssertionError('V2.2 pixel routes overlap')
    if not np.array_equal(spatial_mask, hybrid | forced_region | kept_gap):
        raise AssertionError('V2.2 left pixels without an Expert')
    audit = dict(audit_v21)
    audit.update(
        policy="v22_true_mask_gap_purity",
        v21_region_long_gap_pixels=int(v21_long_gap.sum()),
        v22_long_gap_rejected_non_dominant=int(rejected_insufficient_true_mask),
        v22_long_gap_rejected_incoherent=int(rejected_fragmented_target),
        v22_long_gap_returned_to_hybrid=int((v21_long_gap & ~kept_gap).sum()),
        region_long_gap_pixels=int(kept_gap.sum()),
        hybrid_pixels=int(hybrid.sum()),
    )
    assert (int(v21_long_gap.sum()) == int(kept_gap.sum())
            + rejected_insufficient_true_mask + rejected_fragmented_target)
    return hybrid, forced_region, kept_gap, audit
