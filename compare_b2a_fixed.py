"""Paired B-to-A audit: original Region + Aligned Router vs RankCV + Line-aware.

The RankCV case predictions are taken from the existing fixed-region expert
cache. The original Aligned Router must have been evaluated with the *same*
fixed-union Region support using test.py --region-fixed-eval.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENT = 'swir__train_6-7-8-9-10__test_1-2-3-4-5'
OLD_ROUTER = 'learned_router3_aligned'
NEW_ROUTER = 'learned_router3_lineaware_rankcv'
PROTOCOL = 'fixed_union_of_sampled_coordinates_v1'


def load_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing input file: {path}')
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('No rows to save')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def positive_int(value, label):
    result = int(float(value))
    if result < 1:
        raise ValueError(f'{label} must be positive, received {value}')
    return result


def num(row, column):
    value = float(row[column])
    if not math.isfinite(value):
        raise ValueError(f'Invalid {column} at {row.get("dataset")}/{row.get("mask_file")}')
    return value


def index_rows(rows, expected_router, *, require_status=False, require_fixed=True):
    out = {}
    for row in rows:
        if str(row.get('router_variant') or row.get('router', '')) != expected_router:
            continue
        if require_status and row.get('status') != 'ok':
            raise ValueError(f'Failed old case: {row.get("dataset")}/{row.get("mask_file")}: {row.get("status")}')
        key = (str(row['dataset']), str(row['mask_file']))
        if key in out:
            raise ValueError(f'Duplicate case: {key}')
        if require_fixed and row.get('region_tiling_protocol') != PROTOCOL:
            raise ValueError(f'Wrong Region protocol for {key}: {row.get("region_tiling_protocol")!r}')
        if abs(num(row, 'coverage') - 1.0) > 1e-9:
            raise ValueError(f'Incomplete coverage for {key}')
        out[key] = row
    return out


def paired_audit(old_rows, new_rows, *, expected_cases=200):
    old = index_rows(old_rows, OLD_ROUTER, require_status=True)
    new = index_rows(new_rows, NEW_ROUTER)
    if len(old) != expected_cases or len(new) != expected_cases:
        raise ValueError(f'Expected {expected_cases} cases per model, got old={len(old)}, rankcv={len(new)}')
    if set(old) != set(new):
        only_old = sorted(set(old) - set(new))[:5]
        only_new = sorted(set(new) - set(old))[:5]
        raise ValueError(f'Mask mismatch; old-only {only_old}, rankcv-only {only_new}')

    cases = []
    for key in sorted(old):
        a, b = old[key], new[key]
        for field in ('scenario', 'severity', 'repeat', 'target_voxels', 'full_mask_voxels', 'affected_bands'):
            if str(a.get(field, '')) != str(b.get(field, '')):
                raise ValueError(f'Mismatched {field} for {key}: {a.get(field)} vs {b.get(field)}')
        count = positive_int(a['target_voxels'], 'target_voxels')
        old_rmse = num(a, 'rmse')
        new_rmse = num(b, 'rmse')
        cases.append({
            'dataset': key[0], 'mask_file': key[1],
            'scenario': a['scenario'], 'severity': int(a['severity']),
            'repeat': int(a['repeat']), 'target_voxels': count,
            'original_region_rmse': old_rmse, 'rankcv_lineaware_rmse': new_rmse,
            'rmse_improvement': old_rmse - new_rmse,
            'pct_improvement': 100.0 * (old_rmse - new_rmse) / old_rmse if old_rmse > 0 else math.nan,
            'result': 'rankcv_better' if new_rmse < old_rmse - 1e-12 else ('original_better' if old_rmse < new_rmse - 1e-12 else 'tie'),
        })
    return cases


def protocol_shift(original_routed_rows, original_fixed_rows, *, expected_cases=200):
    """Diagnostic effect of changing only the original Region tiling support."""
    routed = index_rows(original_routed_rows, OLD_ROUTER, require_status=True, require_fixed=False)
    fixed = index_rows(original_fixed_rows, OLD_ROUTER, require_status=True)
    if len(routed) != expected_cases or len(fixed) != expected_cases or set(routed) != set(fixed):
        raise ValueError('Historical and fixed original-Region runs do not share the same complete mask set')
    rows = []
    for key in sorted(fixed):
        a, b = routed[key], fixed[key]
        for field in ('scenario', 'severity', 'repeat', 'target_voxels', 'full_mask_voxels', 'affected_bands'):
            if str(a.get(field, '')) != str(b.get(field, '')):
                raise ValueError(f'Mismatched historical {field} for {key}')
        x, y = num(a, 'rmse'), num(b, 'rmse')
        rows.append(dict(dataset=key[0], mask_file=key[1], scenario=a['scenario'],
                         severity=int(a['severity']), routed_only_rmse=x, fixed_union_rmse=y,
                         fixed_minus_routed=y-x))
    return rows


def summarize(rows, group_fields=()):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[x] for x in group_fields)].append(row)
    result = []
    for key in sorted(groups):
        group = groups[key]
        n = len(group)
        old_mean = sum(r['original_region_rmse'] for r in group) / n
        new_mean = sum(r['rankcv_lineaware_rmse'] for r in group) / n
        total_voxels = sum(r['target_voxels'] for r in group)
        old_pooled = math.sqrt(sum(r['original_region_rmse']**2*r['target_voxels'] for r in group) / total_voxels)
        new_pooled = math.sqrt(sum(r['rankcv_lineaware_rmse']**2*r['target_voxels'] for r in group) / total_voxels)
        out = {field: item for field, item in zip(group_fields, key)}
        out.update(cases=n, original_mean_rmse=old_mean, rankcv_mean_rmse=new_mean,
                   mean_rmse_improvement_pct=100.0*(old_mean-new_mean)/old_mean if old_mean > 0 else math.nan,
                   original_pooled_rmse=old_pooled, rankcv_pooled_rmse=new_pooled,
                   pooled_improvement_pct=100.0*(old_pooled-new_pooled)/old_pooled if old_pooled > 0 else math.nan,
                   rankcv_better=sum(r['result']=='rankcv_better' for r in group),
                   original_better=sum(r['result']=='original_better' for r in group),
                   ties=sum(r['result']=='tie' for r in group))
        result.append(out)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    baseline = ROOT / 'results' / EXPERIMENT / 'combined' / 'tpssr_learned_router3_aligned__fixed_union_region' / 'all_test_metrics.csv'
    score = ROOT / 'results' / EXPERIMENT / '_router_expert_cache_v1' / 'SCORE_ALL_CASES.csv'
    historical = ROOT / 'results' / EXPERIMENT / 'combined' / 'tpssr_learned_router3_aligned' / 'all_test_metrics.csv'
    parser.add_argument('--baseline', type=Path, default=baseline)
    parser.add_argument('--score', type=Path, default=score)
    parser.add_argument('--historical', type=Path, default=historical,
                        help='Optional old routed-only Aligned results; read if file exists')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / EXPERIMENT / 'combined' / '_paired_original_vs_rankcv_fixed')
    parser.add_argument('--expected-cases', type=int, default=200)
    args = parser.parse_args(argv)
    if args.expected_cases != 200:
        raise ValueError('This final B-to-A comparison requires the fixed 200-mask budget')
    paired = paired_audit(load_csv(args.baseline), load_csv(args.score), expected_cases=args.expected_cases)
    overall = summarize(paired)[0]
    save_csv(args.output_dir / 'PAIRED_CASES.csv', paired)
    save_csv(args.output_dir / 'PAIRED_BY_SCENARIO.csv', summarize(paired, ('scenario',)))
    save_csv(args.output_dir / 'PAIRED_BY_SCENARIO_SEVERITY.csv', summarize(paired, ('scenario','severity')))
    lines = [
        'B-to-A MATCHED FIXED-UNION COMPARISON',
        f'Cases: {overall["cases"]}; masks, severity, evaluated voxels, coverage and Region tiling matched',
        'Original Router3 Aligned + original Region vs RankCV + Region Line-aware',
        f'Mean mask RMSE: {overall["original_mean_rmse"]:.9f} -> {overall["rankcv_mean_rmse"]:.9f}',
        f'Mean RMSE change: {overall["mean_rmse_improvement_pct"]:+.3f}% (positive means RankCV better)',
        f'Pooled voxel RMSE: {overall["original_pooled_rmse"]:.9f} -> {overall["rankcv_pooled_rmse"]:.9f}',
        f'Pooled RMSE change: {overall["pooled_improvement_pct"]:+.3f}%',
        f'Masks: RankCV better {overall["rankcv_better"]}, original better {overall["original_better"]}, ties {overall["ties"]}',
        'The fixed-union protocol differs from the historical routed-only test. Do not compare these new values with historical 0.005418 as if they were matched.',
    ]
    if args.historical.is_file():
        shift = protocol_shift(load_csv(args.historical), load_csv(args.baseline),
                               expected_cases=args.expected_cases)
        save_csv(args.output_dir / 'PROTOCOL_SHIFT_ORIGINAL_REGION.csv', shift)
        mean_routed = sum(r['routed_only_rmse'] for r in shift) / len(shift)
        mean_fixed = sum(r['fixed_union_rmse'] for r in shift) / len(shift)
        lines.extend([
            '', 'ADDITIONAL PROTOCOL-SHIFT CHECK (same original Router and Region checkpoints)',
            f'Original routed-only mean RMSE: {mean_routed:.9f}',
            f'Original fixed-union mean RMSE: {mean_fixed:.9f}',
            f'Protocol effect on original baseline: {100.0*(mean_fixed-mean_routed)/mean_routed:+.3f}%',
        ])
    else:
        lines.append(f'Historical routed-only CSV not found: {args.historical}; optional protocol-shift check skipped')
    text = '\n'.join(lines)+'\n'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'PAIRED_SUMMARY.txt').write_text(text, encoding='utf-8')
    print(text)
    print('Report folder:', args.output_dir)


if __name__ == '__main__':
    main()
