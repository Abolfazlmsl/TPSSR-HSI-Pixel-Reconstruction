#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot REAL held-out RMSE vs severity for TPSSR and available reference methods.

Only reads existing CSV/ZIP; does NOT train models or use GPU.

Example (from the TPSSR project root):
    python plot_tpssr_scenario_rmse_v22.py \
      --experiment-dir 'results/swir__train_1-2-3-4-6-7-8-9__test_5-10' \
      --tpssr-csv 'results/swir__train_1-2-3-4-6-7-8-9__test_5-10/combined/final_tpssr_hybrid_v2_region_2expert_v22/all_test_metrics.csv'

Alternative: --baseline-zip '/path/to/swir__train_...zip' --tpssr-csv ...

Optional method(s): --extra DDS2M=/some/SWIR5/.../metrics.csv --extra DDS2M=/some/SWIR10/.../metrics.csv
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

METHODS = {
    'spectral': 'Spectral',
    'spatial': 'Spatial',
    'spatio_spectral': 'Spatio-spectral',
    'pca': 'PCA',
    'random_forest': 'RF',
    'pfgin': 'PFGIN',
}
METHOD_ORDER = ['TPSSR V2.2', 'Spectral', 'Spatio-spectral', 'Spatial', 'PCA', 'RF', 'PFGIN']
COLORS = {
    'TPSSR V2.2': '#101828', 'Spectral': '#2878B5',
    'Spatial': '#EF8D32', 'Spatio-spectral': '#299965',
    'PCA': '#9868BC', 'RF': '#CF547B', 'PFGIN': '#916849',
    'DDS2M': '#31A3A3', 'CCS2M': '#E1A83A',
}
SCENARIOS = [
    'band_wide_saturation', 'consecutive_spectral_bands',
    'isolated_voxels', 'mixed_saturation', 'multiple_detector_lines',
    'partial_vertical_lines', 'single_detector_line', 'spatial_patch',
    'spatio_spectral_block', 'whole_spectrum_pixels',
]
SEVERITIES = [1, 2, 3, 4, 5]
KEYS = ['dataset', 'scenario', 'severity', 'repeat']


def args_parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiment-dir', type=Path, help='Results/<experiment> directory with SWIR5/ and SWIR10/ baseline folders')
    p.add_argument('--baseline-zip', type=Path, help='ZIP containing <experiment>/<SWIR>/<method>/metrics.csv')
    p.add_argument('--tpssr-csv', type=Path, help='V2.2 combined all_test_metrics.csv. Preferred over auto-discovery')
    p.add_argument('--output-dir', type=Path, default=Path('rmse_scenario_comparison_v22'))
    p.add_argument('--extra', action='append', default=[], metavar='METHOD=CSV', help='Other method metrics; repeat for each core')
    p.add_argument('--no-zoom', action='store_true', help='Skip zoom view with closer-performing methods')
    p.add_argument('--dpi', type=int, default=160)
    return p.parse_args()


def find_tpssr(root):
    root = root or Path.cwd()
    matches = sorted(root.glob('combined/final_tpssr_hybrid_v2_region_2expert_v22/all_test_metrics.csv'))
    matches += sorted(root.glob('results/*/combined/final_tpssr_hybrid_v2_region_2expert_v22/all_test_metrics.csv'))
    # Allow launching from a directory holding project outputs, but never choose arbitrary older TPSSR.
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError('V2.2 all_test_metrics.csv not found automatically. Please supply --tpssr-csv PATH')
    raise RuntimeError('Multiple V2.2 CSV files found. Supply --tpssr-csv explicitly: ' + str(matches))


def standardize(df, method, fallback_dataset=None, source=''):
    d = df.copy()
    if 'dataset' not in d.columns:
        if not fallback_dataset:
            raise ValueError(f'{source}: missing dataset column, and not inside a SWIR5/SWIR10 folder')
        d['dataset'] = fallback_dataset
    required = {'scenario', 'severity', 'repeat', 'rmse'}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f'{source}: missing columns: {sorted(missing)}')
    if 'valid_n' not in d:
        if 'filled_voxels' in d:
            d['valid_n'] = d['filled_voxels']
        else:
            raise ValueError(f'{source}: missing valid_n / filled_voxels: cannot calculate pooled RMSE fairly')
    if 'target_n' not in d:
        if 'target_voxels' in d:
            d['target_n'] = d['target_voxels']
        else:
            raise ValueError(f'{source}: missing target_n / target_voxels')
    if 'status' in d:
        d = d[d['status'].astype(str).str.lower().isin(['ok', 'success', 'completed'])].copy()
    for col in ('severity', 'repeat', 'valid_n', 'target_n', 'rmse'):
        d[col] = pd.to_numeric(d[col], errors='coerce')
    d['dataset'] = d['dataset'].astype(str).str.upper().str.strip()
    d['scenario'] = d['scenario'].astype(str).str.strip().str.lower()
    d['method'] = method
    d = d[d['severity'].isin(SEVERITIES) & d['dataset'].isin(['SWIR5', 'SWIR10'])].copy()
    d = d[KEYS + ['method', 'rmse', 'valid_n', 'target_n']]
    if d.empty:
        raise ValueError(f'{source}: no SWIR5/SWIR10 data with severity 1..5')
    bad = d[(d['valid_n'] < 0) | (d['target_n'] <= 0) | (d['valid_n'] > d['target_n'] + 0.01)]
    if len(bad):
        raise ValueError(f'{source}: invalid voxel counts on {len(bad)} rows')
    if d.duplicated(KEYS).any():
        raise ValueError(f'{source}: duplicate dataset/scenario/severity/repeat masks')
    # No negative or infinity metrics. Zero-coverage masks have NaN RMSE.
    d.loc[~np.isfinite(d['rmse']) | (d['valid_n'] <= 0), 'rmse'] = np.nan
    return d


def dataset_from_path(path):
    for part in path.parts:
        if re.fullmatch(r'SWIR(?:5|10)', part, re.I):
            return part.upper()
    return None


def load_benchmark_zip(path):
    frames = []
    with zipfile.ZipFile(path) as z:
        for item in z.namelist():
            parts = Path(item).parts
            if len(parts) < 3 or parts[-1].lower() != 'metrics.csv':
                continue
            label = METHODS.get(parts[-2].lower())
            if label is None or parts[-3].upper() not in ('SWIR5', 'SWIR10'):
                continue
            d = pd.read_csv(io.BytesIO(z.read(item)))
            frames.append(standardize(d, label, parts[-3], source=item))
    return frames


def load_benchmark_dir(path):
    frames = []
    for ds in ('SWIR5', 'SWIR10'):
        for folder, label in METHODS.items():
            csv_path = path / ds / folder / 'metrics.csv'
            if not csv_path.exists():
                print(f'[NOTE] No {label} results for {ds}: {csv_path}')
                continue
            frames.append(standardize(pd.read_csv(csv_path), label, ds, source=str(csv_path)))
    return frames


def load_extra(spec):
    if '=' not in spec:
        raise ValueError(f'Expected --extra METHOD=CSV, got: {spec}')
    method, file = spec.split('=', 1)
    method = method.strip()
    file = Path(file.strip()).expanduser()
    if not method:
        raise ValueError(f'Empty method name in --extra={spec}')
    if not file.is_file():
        raise FileNotFoundError(file)
    if method.lower() in ('tpssr v2.2', 'tpssr'):
        raise ValueError('Use --tpssr-csv for the primary TPSSR V2.2 results')
    return standardize(pd.read_csv(file), method, dataset_from_path(file), source=str(file))


def check_keys(d):
    tpssr = d[d['method'] == 'TPSSR V2.2']
    expected = set(map(tuple, tpssr[KEYS].to_numpy()))
    print(f'[CHECK] TPSSR V2.2: {len(expected)} masks; {tpssr.groupby("dataset").size().to_dict()}')
    if len(expected) != 200:
        print('[WARNING] TPSSR has other than 200 masks; charts show what is actually present.')
    for method, group in d.groupby('method'):
        if method == 'TPSSR V2.2':
            continue
        keys = set(map(tuple, group[KEYS].to_numpy()))
        missing = expected - keys
        extra = keys - expected
        print(f'[CHECK] {method}: {len(keys)} masks; missing={len(missing)}, extra={len(extra)}')
        if missing or extra:
            print(f'[WARNING] Different mask keys for {method}; direct comparisons may not match.')


def summarize(d):
    out = []
    for (ds, sc, severity, method), part in d.groupby(['dataset', 'scenario', 'severity', 'method'], sort=True):
        rmse = part['rmse'].to_numpy(dtype=float)
        n = part['valid_n'].to_numpy(dtype=float)
        tar = part['target_n'].to_numpy(dtype=float)
        ok = np.isfinite(rmse) & np.isfinite(n) & (n > 0)
        value = float(np.sqrt(np.sum(n[ok] * rmse[ok]**2) / np.sum(n[ok]))) if ok.any() else np.nan
        out.append(dict(dataset=ds, scenario=sc, severity=int(severity), method=method,
                        pooled_rmse=value,
                        coverage=float(n.sum()/tar.sum()) if tar.sum() > 0 else np.nan,
                        valid_voxels=int(n.sum()), target_voxels=int(tar.sum()),
                        masks=int(len(part)), valid_masks=int(ok.sum())))
    return pd.DataFrame(out).sort_values(['dataset', 'scenario', 'severity', 'method'])


def pick_zoom_methods(data):
    # Always retain the two conceptual anchors; add 3 strongest *fully covered* references.
    mandatory = ['TPSSR V2.2', 'Spectral']
    candidates = []
    for method, rows in data.groupby('method'):
        if method in mandatory:
            continue
        good = rows[(rows['coverage'] >= .99999) & np.isfinite(rows['pooled_rmse'])]
        if len(good):
            candidates.append((float(good['pooled_rmse'].median()), method))
    candidates.sort()
    return mandatory + [m for _, m in candidates[:3]]


def plot_one(ax, dat, methods, title, legend=False):
    ax.set_title(title, fontsize=13, fontweight='bold')
    for method in methods:
        chunk = dat[dat['method'] == method].set_index('severity')
        if chunk.empty:
            continue
        xs = np.asarray(SEVERITIES, dtype=int)
        ys = np.asarray([chunk.at[s, 'pooled_rmse'] if s in chunk.index else np.nan for s in xs], dtype=float)
        cov = np.asarray([chunk.at[s, 'coverage'] if s in chunk.index else np.nan for s in xs], dtype=float)
        if not np.isfinite(ys).any():
            continue
        color = COLORS.get(method, None)
        strong = method == 'TPSSR V2.2'
        # Draw Spectral ABOVE TPSSR and with a different marker shape.
        # Otherwise the two curves can be nearly identical and the thick
        # black TPSSR line completely hides the blue Spectral line.
        is_spectral = method == 'Spectral'
        line, = ax.plot(
            xs, ys, color=color, label=method,
            lw=3 if strong else (2.1 if is_spectral else 1.8),
            ls='--' if is_spectral else '-',
            marker='s' if is_spectral else 'o',
            markersize=6.8 if is_spectral else (6.5 if strong else 5),
            markeredgecolor='white' if is_spectral else None,
            markeredgewidth=0.65 if is_spectral else 0.0,
            zorder=7 if is_spectral else (5 if strong else 3),
        )
        incomplete = np.isfinite(ys) & (cov < .99999)
        if incomplete.any():
            ax.scatter(xs[incomplete], ys[incomplete], s=75 if strong else 62,
                       facecolors='white', edgecolors=line.get_color(), linewidths=1.8, zorder=9)
    ax.set_xticks(SEVERITIES)
    ax.set_xlim(.8, 5.2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Severity (S1 - S5)')
    ax.set_ylabel('Pooled RMSE (lower is better)')
    ax.grid(True, alpha=.25, linestyle=':')
    if not dat['pooled_rmse'].notna().any():
        ax.text(.5, .5, 'No measurements', transform=ax.transAxes, ha='center', va='center')


def legend_on_fig(fig, axes, methods, fontsize=9):
    handles = {}
    for ax in axes:
        hs, ls = ax.get_legend_handles_labels()
        for h, l in zip(hs, ls):
            handles[l] = h
    if handles:
        ordered = [m for m in methods if m in handles]
        fig.legend([handles[m] for m in ordered], ordered, loc='lower center',
                   bbox_to_anchor=(.5, .005), ncol=min(4, len(ordered)), fontsize=fontsize,
                   frameon=True)


def figure_scenario(summary, scenario, methods, zoom=False):
    dat = summary[summary['scenario'] == scenario]
    if zoom:
        methods = pick_zoom_methods(dat)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.4))
    for ax, ds in zip(axes, ('SWIR5', 'SWIR10')):
        plot_one(ax, dat[dat['dataset'] == ds], methods, ds)
    fig.suptitle(scenario.replace('_', ' ').title() + (' — close comparison' if zoom else ' — all reported methods'),
                 fontsize=16, fontweight='bold', y=.99)
    legend_on_fig(fig, axes, methods)
    fig.text(.5, .165, 'Hollow marker = incomplete voxel coverage. No point = RMSE unavailable. '
             'Methods with different coverage are not strictly comparable.',
             ha='center', fontsize=9, color='#666666')
    fig.subplots_adjust(bottom=.27, top=.85, wspace=.22)
    return fig


def figure_overview(summary, ds, methods):
    fig, axes = plt.subplots(2, 5, figsize=(21, 9.4))
    for ax, sc in zip(axes.flat, SCENARIOS):
        dat = summary[(summary['dataset'] == ds) & (summary['scenario'] == sc)]
        plot_one(ax, dat, methods, sc.replace('_', ' ').title())
        ax.set_title(sc.replace('_', ' ').replace('spatio spectral', 'spatio-spectral').title(), fontsize=10.4)
        ax.set_ylabel('RMSE' if ax in (axes[0, 0], axes[1, 0]) else '')
        ax.set_xlabel('Severity')
        ax.tick_params(axis='both', labelsize=8)
    fig.suptitle(f'TPSSR V2.2 and Baselines — {ds} | RMSE vs Severity', fontsize=18, y=.98)
    legend_on_fig(fig, list(axes.flat), methods, fontsize=10)
    fig.text(.5, .135, 'Hollow marker = incomplete coverage; missing curves are not zero.',
             ha='center', fontsize=10, color='#666666')
    fig.subplots_adjust(bottom=.23, top=.9, wspace=.24, hspace=.44)
    return fig


def winner_table(summary):
    rows = []
    for (ds, sc, s), data in summary.groupby(['dataset', 'scenario', 'severity']):
        covered = data[(data['coverage'] >= .99999) & np.isfinite(data['pooled_rmse'])]
        winner = covered.sort_values('pooled_rmse').iloc[0] if not covered.empty else None
        tpssr = data[data['method'] == 'TPSSR V2.2']
        rows.append(dict(dataset=ds, scenario=sc, severity=s,
                         best_full_coverage_method=winner['method'] if winner is not None else '',
                         best_full_coverage_rmse=winner['pooled_rmse'] if winner is not None else np.nan,
                         tpssr_rmse=tpssr.iloc[0]['pooled_rmse'] if len(tpssr) else np.nan,
                         tpssr_coverage=tpssr.iloc[0]['coverage'] if len(tpssr) else np.nan,
                         number_of_full_coverage_methods=int(len(covered))))
    return pd.DataFrame(rows)


def main():
    a = args_parse()
    if a.experiment_dir is not None and not a.experiment_dir.is_dir():
        raise FileNotFoundError(f'Experiment directory missing: {a.experiment_dir}')
    tpssr_path = a.tpssr_csv or find_tpssr(a.experiment_dir)
    if not tpssr_path.is_file():
        raise FileNotFoundError(tpssr_path)
    frames = [standardize(pd.read_csv(tpssr_path), 'TPSSR V2.2', source=str(tpssr_path))]
    if a.baseline_zip:
        if not a.baseline_zip.is_file():
            raise FileNotFoundError(a.baseline_zip)
        frames += load_benchmark_zip(a.baseline_zip)
        print('[SOURCE] Baselines ZIP:', a.baseline_zip)
    elif a.experiment_dir:
        frames += load_benchmark_dir(a.experiment_dir)
        print('[SOURCE] Baselines directory:', a.experiment_dir)
    else:
        raise ValueError('Give --experiment-dir DIR or --baseline-zip FILE to load baselines')
    for spec in a.extra:
        frames.append(load_extra(spec))
    if len(frames) < 2:
        raise RuntimeError('No reference methods found. Check folder/zip paths.')
    all_data = pd.concat(frames, ignore_index=True)
    if all_data.duplicated(KEYS + ['method']).any():
        dupe = all_data[all_data.duplicated(KEYS + ['method'], keep=False)]
        raise ValueError('Duplicate method and mask keys (different runs mixed):\n' + dupe[KEYS + ['method']].head(8).to_string(index=False))
    check_keys(all_data)
    summary = summarize(all_data)
    methods = [m for m in METHOD_ORDER if m in set(all_data.method)]
    methods += sorted(set(all_data.method) - set(methods))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(a.output_dir/'rmse_and_coverage_by_scenario_severity.csv', index=False)
    wide = summary.pivot(index=['dataset','scenario','severity'], columns='method', values='pooled_rmse')
    wide.reindex(columns=methods).to_csv(a.output_dir/'rmse_wide_comparison.csv')
    summary.pivot(index=['dataset','scenario','severity'], columns='method', values='coverage').reindex(
        columns=methods).to_csv(a.output_dir/'coverage_wide_comparison.csv')
    winner_table(summary).to_csv(a.output_dir/'winner_full_coverage_by_case.csv', index=False)
    with PdfPages(a.output_dir/'ALL_10_SCENARIOS_ALL_METHODS.pdf') as pdf:
        for sc in SCENARIOS:
            fig = figure_scenario(summary, sc, methods, zoom=False)
            fig.savefig(a.output_dir/f'RMSE_{sc}.png', dpi=a.dpi, bbox_inches='tight')
            pdf.savefig(fig)
            plt.close(fig)
    if not a.no_zoom:
        (a.output_dir/'zoom').mkdir(exist_ok=True)
        with PdfPages(a.output_dir/'ALL_10_SCENARIOS_ZOOM.pdf') as pdf:
            for sc in SCENARIOS:
                fig = figure_scenario(summary, sc, methods, zoom=True)
                fig.savefig(a.output_dir/'zoom'/f'RMSE_{sc}_zoom.png', dpi=a.dpi, bbox_inches='tight')
                pdf.savefig(fig)
                plt.close(fig)
    for ds in ('SWIR5', 'SWIR10'):
        fig = figure_overview(summary, ds, methods)
        fig.savefig(a.output_dir/f'OVERVIEW_ALL_SCENARIOS_{ds}.png', dpi=a.dpi, bbox_inches='tight')
        plt.close(fig)
    print('[DONE] RMSE charts, two overview charts, PDF report(s), and numerical CSV tables saved to:')
    print('       ', a.output_dir.resolve())
    print('[NOTE] A lower RMSE for a method with <100% coverage does NOT establish an equal-coverage win.')
    print('[NOTE] Pooled RMSE values use valid_n-weighted SSE, not the arithmetic mean of RMSE masks.')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('[ERROR]', type(e).__name__ + ':', str(e), file=sys.stderr)
        sys.exit(1)
