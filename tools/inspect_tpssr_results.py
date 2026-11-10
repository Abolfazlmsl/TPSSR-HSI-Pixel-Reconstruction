#!/usr/bin/env python3
"""Lightweight TPSSR result inventory (standard library only).

Does not load images, HSI cubes, checkpoints or array contents. Reports the
folder layout, file paths, and limited CSV/JSON/TXT schema samples so a precise
Excel results extractor can be built without copying the whole results tree.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
from pathlib import Path
import re
import zipfile

DATA_EXTENSIONS = {'.csv', '.tsv', '.json', '.txt', '.md', '.xlsx', '.xls', '.parquet'}
SKIP_DIRS = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', 'site-packages', '.mypy_cache'}
METRIC_NAMES = ('metric', 'summary', 'score', 'result', 'case', 'paired', 'evaluate',
                'history', 'router', 'scenario', 'severity', 'audit', 'risk',
                'benchmark', 'fold', 'training', 'comparison')


def safe_open_text(path: Path):
    # Data is primarily ASCII/UTF-8; replacement prevents a locale-specific crash.
    return path.open('r', encoding='utf-8-sig', errors='replace', newline='')


def shorten(v, limit=250):
    v = str(v).replace('\r', ' ').replace('\n', ' ').strip()
    return v[:limit] + ('...' if len(v) > limit else '')


def limited_shape(value, depth=0):
    if depth >= 3:
        return type(value).__name__
    if isinstance(value, dict):
        return {str(k): limited_shape(v, depth + 1) for k, v in list(value.items())[:24]}
    if isinstance(value, list):
        return {'type': 'list', 'length': len(value), 'item': limited_shape(value[0], depth + 1) if value else None}
    return type(value).__name__


def read_schema(path: Path):
    ext = path.suffix.lower()
    try:
        if ext in {'.csv', '.tsv'}:
            with safe_open_text(path) as f:
                first = f.read(4096)
                f.seek(0)
                delim = '\t' if ext == '.tsv' else ','
                if ext == '.csv':
                    try:
                        delim = csv.Sniffer().sniff(first, delimiters=',;\t|').delimiter
                    except csv.Error:
                        pass
                reader = csv.reader(f, delimiter=delim)
                header = next(reader, [])
                rows = []
                for _ in range(2):
                    row = next(reader, None)
                    if row is None: break
                    rows.append([shorten(c, 105) for c in row[:12]])
                return ('tabular', {'delimiter': repr(delim),
                                     'columns': [shorten(x, 160) for x in header[:120]],
                                     'column_count': len(header), 'first_two_rows': rows})
        if ext == '.json' and path.stat().st_size <= 15_000_000:
            with safe_open_text(path) as f:
                data = json.load(f)
            return ('json', {'shape': limited_shape(data)})
        if ext in {'.txt', '.md'}:
            with safe_open_text(path) as f:
                lines = [shorten(line, 220) for _, line in zip(range(7), f)]
            return ('text', {'opening_lines': lines})
        return ('other', {})
    except Exception as exc:
        return ('error', {'message': shorten(f'{type(exc).__name__}: {exc}')})


def folder_summary_path(p: str, max_depth: int):
    parts = p.split('/') if p != '.' else []
    return '/'.join(parts[:max_depth]) + ('/... ' if len(parts) > max_depth else '') or '.'


def main():
    ap = argparse.ArgumentParser(description='Inventory a large TPSSR results folder without reading binary contents')
    ap.add_argument('--root', default='results', help='Results folder to inspect; usually results')
    ap.add_argument('--out', default='TPSSR_RESULTS_SCAN', help='Output filename prefix in current directory')
    ap.add_argument('--max-files', type=int, default=500000, help='Safety limit for file inventory')
    ap.add_argument('--sample-per-type', type=int, default=10, help='Maximum schema samples per filename')
    ap.add_argument('--max-report-folders', type=int, default=450)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f'Cannot find folder: {root}\nRun the script from the project folder or pass --root /full/path/to/results')
    output_prefix = Path(args.out).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    manifest_file = output_prefix.with_name(output_prefix.name + '_FILES.csv')
    schema_file = output_prefix.with_name(output_prefix.name + '_SCHEMAS.csv')
    report_file = output_prefix.with_name(output_prefix.name + '_STRUCTURE.txt')
    zip_file = output_prefix.with_suffix('.zip')

    extensions = collections.Counter()
    dir_counts = collections.Counter()
    dir_sizes = collections.Counter()
    folders = set(['.'])
    key_files = collections.defaultdict(list)
    sample_count = collections.Counter()
    skipped_dirs = []
    error_paths = []
    total_size = 0
    all_count = 0
    limit_hit = False
    with manifest_file.open('w', newline='', encoding='utf-8') as mf, schema_file.open('w', newline='', encoding='utf-8') as sf:
        manifest = csv.writer(mf)
        schemas = csv.writer(sf)
        manifest.writerow(['relative_path', 'size_bytes', 'extension', 'parent_folder'])
        schemas.writerow(['relative_path', 'kind', 'schema_json'])
        for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
            dirs.sort()
            allowed = []
            for d in dirs:
                if d in SKIP_DIRS:
                    skipped_dirs.append(str(Path(current, d).relative_to(root)))
                else:
                    allowed.append(d)
            dirs[:] = allowed
            rel_dir = Path(current).relative_to(root).as_posix()
            folders.add(rel_dir)
            for name in sorted(files):
                path = Path(current, name)
                if path.is_symlink():
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    error_paths.append(str(path.relative_to(root)))
                    continue
                relative = path.relative_to(root).as_posix()
                ext = path.suffix.lower() or '[no extension]'
                manifest.writerow([relative, size, ext, rel_dir])
                all_count += 1
                total_size += size
                extensions[ext] += 1
                dir_counts[rel_dir] += 1
                dir_sizes[rel_dir] += size
                key = name.lower()
                priority = any(word in key for word in METRIC_NAMES)
                if priority and len(key_files[rel_dir]) < 8:
                    key_files[rel_dir].append(name)
                # Capture a few representative *schemas* from each filename. Keep
                # first-row examples for identifying column meaning later.
                if ext in DATA_EXTENSIONS and ext not in {'.xlsx', '.xls', '.parquet'}:
                    norm_name = re.sub(r'\d{6,}', '#', key)
                    norm_name = re.sub(r'(?<=_)[0-9]+(?=\.|_)', '#', norm_name)
                    if sample_count[norm_name] < args.sample_per_type:
                        kind, schema = read_schema(path)
                        schemas.writerow([relative, kind, json.dumps(schema, ensure_ascii=False)])
                        sample_count[norm_name] += 1
                if all_count >= args.max_files:
                    limit_hit = True
                    break
            if limit_hit:
                break

    # Summarize the directory layout in a bounded, legible report.
    # Sorting metric-bearing dirs first keeps the report useful for huge trees.
    folder_list = sorted(folders, key=lambda p: (0 if dir_counts[p] and key_files[p] else 1,
                                                 p.count('/'), p.lower()))
    with report_file.open('w', encoding='utf-8') as f:
        f.write('TPSSR RESULTS DIRECTORY INSPECTION\n')
        f.write('=' * 60 + '\n')
        f.write(f'ROOT: {root}\n')
        f.write(f'FILES FOUND: {all_count:,}\n')
        f.write(f'FOLDERS FOUND: {len(folders):,}\n')
        f.write(f'TOTAL FILE SIZE: {total_size / (1024**3):.3f} GiB\n')
        f.write(f'SCAN COMPLETE: {not limit_hit}\n')
        if limit_hit:
            f.write('WARNING: File limit reached. Increase --max-files for full coverage.\n')
        if error_paths:
            f.write(f'STAT ERRORS: {len(error_paths)} (first few: {error_paths[:8]})\n')
        f.write('\nFILE TYPES (count):\n')
        for ext, count in extensions.most_common(35):
            f.write(f'  {ext}: {count:,}\n')
        f.write('\nTOP-LEVEL DIRECTORIES:\n')
        top_dirs = collections.Counter()
        for folder, count in dir_counts.items():
            top_dirs[folder.split('/')[0] if folder != '.' else '.'] += count
        for folder, count in top_dirs.most_common(100):
            f.write(f'  {folder}: {count:,} files\n')
        f.write('\nDIRECTORY MAP (files directly inside each directory):\n')
        for folder in folder_list[:args.max_report_folders]:
            if not dir_counts[folder] and folder.count('/') > 2:
                continue
            f.write(f'  {folder}/  [{dir_counts[folder]:,} files, {dir_sizes[folder]/1048576:.1f} MiB direct]\n')
            if key_files[folder]:
                f.write('    important names: ' + ', '.join(key_files[folder]) + '\n')
        if len(folder_list) > args.max_report_folders:
            f.write(f'  ...and {len(folder_list)-args.max_report_folders:,} more folders. See _FILES.csv for every file.\n')
        if skipped_dirs:
            f.write('\nSKIPPED non-results directories: ' + ', '.join(skipped_dirs[:50]) + '\n')
        f.write('\nFULL INVENTORY: ' + manifest_file.name + '\n')
        f.write('TABULAR HEADERS, FIRST ROW EXAMPLES, JSON KEY SHAPES: ' + schema_file.name + '\n')
        f.write('No NumPy arrays, checkpoint tensors, PNG images, or other binary payloads were opened.\n')
    with zipfile.ZipFile(zip_file, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=7) as z:
        for path in (report_file, schema_file, manifest_file):
            z.write(path, arcname=path.name)
    print('\n[DONE] Results structure scan complete.')
    print(f'[INFO] Root: {root}')
    print(f'[INFO] Files scanned: {all_count:,}; folders: {len(folders):,}; scan_complete={not limit_hit}')
    print(f'[SEND THIS ZIP] {zip_file}')
    print('This ZIP contains only filenames, sizes, CSV headers/sample rows, and small JSON/TXT samples; no full results.')


if __name__ == '__main__':
    main()
