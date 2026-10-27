import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_b2a_fixed import paired_audit, summarize, PROTOCOL, OLD_ROUTER, NEW_ROUTER


def example(name='mask_1.npz', rmse=0.006):
    return dict(dataset='SWIR1', mask_file=name, scenario='isolated_voxels', severity='1',
                repeat='1', target_voxels='100', full_mask_voxels='200',
                affected_bands='1', coverage='1.0', rmse=str(rmse),
                region_tiling_protocol=PROTOCOL)


def test_identical_keys_and_matched_protocol():
    old = [dict(example(rmse=0.006), router_variant=OLD_ROUTER, status='ok')]
    new = [dict(example(rmse=0.005), router=NEW_ROUTER)]
    paired = paired_audit(old, new, expected_cases=1)
    assert paired[0]['result'] == 'rankcv_better'
    agg = summarize(paired)[0]
    assert agg['cases'] == 1
    assert agg['mean_rmse_improvement_pct'] == pytest.approx(100/6)


def test_reject_unmatched_samples():
    old = [dict(example(), router_variant=OLD_ROUTER, status='ok')]
    new = [dict(example(), router=NEW_ROUTER, target_voxels='101')]
    with pytest.raises(ValueError, match='Mismatched target_voxels'):
        paired_audit(old, new, expected_cases=1)


def test_reject_wrong_legacy_tiling():
    old = [dict(example(), router_variant=OLD_ROUTER, status='ok', region_tiling_protocol='router_selected_coordinates')]
    new = [dict(example(), router=NEW_ROUTER)]
    with pytest.raises(ValueError, match='Wrong Region protocol'):
        paired_audit(old, new, expected_cases=1)


def test_reject_missing_case():
    old = [dict(example(), router_variant=OLD_ROUTER, status='ok')]
    new = [dict(example('mask_2.npz'), router=NEW_ROUTER)]
    with pytest.raises(ValueError, match='Mask mismatch'):
        paired_audit(old, new, expected_cases=1)


def test_reject_duplicated_cases():
    old = [dict(example(), router_variant=OLD_ROUTER, status='ok')] * 2
    new = [dict(example(), router=NEW_ROUTER)]
    with pytest.raises(ValueError, match='Duplicate'):
        paired_audit(old, new, expected_cases=2)


def test_reject_incomplete_coverage():
    old = [dict(example(), router_variant=OLD_ROUTER, status='ok', coverage='0.8')]
    new = [dict(example(), router=NEW_ROUTER)]
    with pytest.raises(ValueError, match='Incomplete coverage'):
        paired_audit(old, new, expected_cases=1)


def test_old_same_model_protocol_shift():
    from compare_b2a_fixed import protocol_shift
    original = [dict(example(rmse=0.005), router_variant=OLD_ROUTER, status='ok',
                     region_tiling_protocol='router_selected_coordinates')]
    fixed = [dict(example(rmse=0.006), router_variant=OLD_ROUTER, status='ok')]
    rows = protocol_shift(original, fixed, expected_cases=1)
    assert rows[0]['fixed_minus_routed'] == pytest.approx(0.001)
