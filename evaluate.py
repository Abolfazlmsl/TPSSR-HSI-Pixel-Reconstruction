#!/usr/bin/env python3
"""Aggregate masked-pixel reconstruction results and export sample images."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from _launcher_common import ROOT, CORE, env_for
from experiment_config import add_experiment_args, experiment_cli_args, resolve_experiment


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict], metric: str) -> float:
    vals = []
    for row in rows:
        try:
            value = float(row.get(metric, "nan"))
        except (ValueError, TypeError):
            continue
        if np.isfinite(value):
            vals.append(value)
    return float(np.mean(vals)) if vals else float("nan")


def summary(rows: list[dict], label: str) -> dict:
    valid = [row for row in rows if row.get("status") == "ok"]
    return {
        "label": label,
        "requested_masks": len(rows),
        "successful_masks": len(valid),
        "failed_masks": len(rows) - len(valid),
        **{f"mean_{metric}": mean(valid, metric) for metric in
           ("rmse", "mae", "r2", "sam_deg", "ssim", "coverage",
            "final_spectral_fraction", "final_region_fraction", "runtime_sec")},
    }


def text_summary(result: dict) -> str:
    return "\n".join([
        f"TPSSR evaluation: {result['label']}",
        f"Successful cases: {result['successful_masks']}/{result['requested_masks']}",
        *(f"{name}: {result[f'mean_{name}']:.6f}" for name in
          ("rmse", "mae", "r2", "sam_deg", "ssim", "coverage")),
    ]) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_args(parser)
    parser.add_argument("--visuals", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--region-batch", type=int, default=32)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def combine(spec) -> None:
    all_rows = []
    seen = set()
    for dataset in spec.tests:
        path = ROOT / "results" / spec.experiment_name / dataset / "tpssr" / "test_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing held-out evaluation data: {path}")
        for row in read_csv(path):
            key = (dataset, row.get("mask_file"))
            if key in seen:
                raise RuntimeError(f"Duplicate case: {key}")
            seen.add(key)
            all_rows.append(row)
    output = ROOT / "results" / spec.experiment_name / "combined" / "tpssr"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "all_test_metrics.csv", all_rows)
    results = summary(all_rows, spec.experiment_name)
    results["train_datasets"] = list(spec.train)
    results["test_datasets"] = list(spec.tests)
    (output / "combined_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "COMBINED_SUMMARY.txt").write_text(text_summary(results), encoding="utf-8")
    groups = {}
    for row in all_rows:
        if row.get("status") != "ok":
            continue
        key = (row.get("dataset"), row.get("scenario"), row.get("severity"))
        groups.setdefault(key, []).append(row)
    records = []
    for (dataset, scenario, severity), rows in sorted(groups.items()):
        records.append({"dataset": dataset, "scenario": scenario, "severity": severity,
                        **summary(rows, label=f"{dataset}/{scenario}/S{severity}")})
    write_csv(output / "scenario_severity_metrics.csv", records)
    print(text_summary(results))
    print("Saved:", output)


def evaluate_one(spec, args):
    os.environ.update(env_for(spec, spec.active_test))
    if str(CORE) not in sys.path:
        sys.path.insert(0, str(CORE))
    global ARGS, DATASET, OUTDIR, METRICS, VISDIR, config, torch
    ARGS = args
    DATASET = spec.active_test
    import torch
    import config
    from src.evaluation_utils import load_dataset, split_blocks, list_mask_files, load_test_mask
    from src.pipeline import load_pipeline, predict_mask
    from src.local_methods import reconstruct as reconstruct_baseline
    # The visual export functions resolve these names in this module's globals.
    globals().update(load_dataset=load_dataset, split_blocks=split_blocks,
                     list_mask_files=list_mask_files, load_test_mask=load_test_mask,
                     load_pipeline=load_pipeline, predict_mask=predict_mask,
                     reconstruct_baseline=reconstruct_baseline)
    OUTDIR = Path(config.RESULTS_DIR) / "tpssr"
    METRICS = OUTDIR / "test_metrics.csv"
    VISDIR = OUTDIR / "visuals"
    if not METRICS.exists():
        raise FileNotFoundError(f"Run test.py before evaluate.py: {METRICS}")
    rows = read_csv(METRICS)
    if not rows:
        raise RuntimeError(f"No test masks found in {METRICS}")
    result = summary(rows, DATASET)
    result["visuals"] = export_visuals(args.visuals, rows) if args.visuals > 0 else []
    (OUTDIR / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUTDIR / "SUMMARY.txt").write_text(text_summary(result), encoding="utf-8")
    print(text_summary(result))


def find_case(mask_files, scenario, severity, repeat):
    for path in mask_files:
        _mask, meta = load_test_mask(path)
        if (
            str(meta["scenario"]) == scenario
            and int(meta["severity"]) == int(severity)
            and int(meta["repeat"]) == int(repeat)
        ):
            return path
    return None

def representative_band(mask):
    counts = np.count_nonzero(mask, axis=(0, 1))
    if not np.any(counts > 0):
        return int(mask.shape[2] // 2)
    maximum = int(counts.max())
    candidates = np.flatnonzero(counts == maximum)
    center = (mask.shape[2] - 1) / 2.0
    return int(candidates[np.argmin(np.abs(candidates - center))])

def _visual_texture_score(image, valid):
    image = np.asarray(image, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(image)
    if valid.mean() < 0.95:
        return 0.0
    gx = np.abs(image[:, 1:] - image[:, :-1])
    gy = np.abs(image[1:, :] - image[:-1, :])
    vx = valid[:, 1:] & valid[:, :-1]
    vy = valid[1:, :] & valid[:-1, :]
    values = np.concatenate((gx[vx], gy[vy]))
    if not len(values):
        return 0.0
    # A percentile suppresses isolated noise and rewards layered texture.
    return float(np.clip(np.percentile(values, 80) / 0.008, 0, 1))


def _visual_crop(center_y, center_x, size, height, width, guard):
    y0 = int(np.clip(round(center_y - size / 2), guard, height - guard - size))
    x0 = int(np.clip(round(center_x - size / 2), guard, width - guard - size))
    return y0, y0 + size, x0, x0 + size


def _visual_band_candidates(mask):
    counts = np.count_nonzero(mask, axis=(0, 1))
    active = np.flatnonzero(counts)
    if active.size <= 28:
        return active.tolist()
    # Inspect both the broad spectral range and the densest corrupted bands.
    spread = active[np.linspace(0, active.size - 1, 18).round().astype(int)]
    peaks = np.argsort(counts)[-12:]
    return sorted(set(int(b) for b in np.concatenate((spread, peaks)) if counts[b]))


def _best_visual_location(mask, band, test_cube, test_invalid, crop_size):
    from scipy import ndimage

    mask2d = np.asarray(mask[:, :, band], dtype=bool)
    if not mask2d.any():
        return None
    height, width = mask2d.shape
    guard = min(16, max(3, min(height, width) // 24))
    size = int(min(crop_size, height - 2 * guard, width - 2 * guard))
    if size < 32:
        return None
    image = np.asarray(test_cube[:, :, band], dtype=np.float32)
    valid = ~np.asarray(test_invalid[:, :, band], dtype=bool) & np.isfinite(image)
    labels, n_components = ndimage.label(mask2d, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    component_ids = np.argsort(sizes)[-min(n_components, 32):][::-1]
    best = None

    for component_id in component_ids:
        if sizes[component_id] < 8:
            continue
        yy, xx = np.nonzero(labels == component_id)
        cy, cx = float(np.median(yy)), float(np.median(xx))
        for dy, dx in [(0, 0), (-0.13, 0), (0.13, 0),
                       (0, -0.13), (0, 0.13)]:
            y0, y1, x0, x1 = _visual_crop(
                cy + dy * size, cx + dx * size, size, height, width, guard
            )
            local = mask2d[y0:y1, x0:x1]
            local_valid = valid[y0:y1, x0:x1]
            valid_fraction = float(local_valid.mean())
            # No silent fallback to the old median-centered crop.
            if valid_fraction < 0.97:
                continue
            n_masked = int(np.count_nonzero(local))
            fraction = n_masked / float(local.size)
            if n_masked < max(8, int(size * size * 0.003)) or fraction > 0.45:
                continue
            # Reject crops with a masked region cut off by the border.
            ys, xs = np.where(local)
            clearance = min(ys.min(), xs.min(),
                            size - 1 - ys.max(), size - 1 - xs.max())
            if clearance < max(3, size // 24):
                continue
            texture = _visual_texture_score(image[y0:y1, x0:x1], local_valid)
            if texture < 0.07:
                continue
            # The geometry and texture determine figure selection, never error
            # against a model prediction or the numerical benchmark ranking.
            crop_mean_y, crop_mean_x = float(ys.mean()), float(xs.mean())
            centrality = 1 - (abs(crop_mean_y - size/2) + abs(crop_mean_x - size/2)) / size
            occ = min(fraction / 0.08, 1.0)
            clearance_score = min(clearance / (size * 0.12), 1.0)
            score = (0.38 * texture + 0.21 * valid_fraction +
                     0.20 * occ + 0.12 * centrality + 0.09 * clearance_score)
            candidate = dict(score=float(score), band=int(band),
                             bounds=(y0, y1, x0, x1),
                             valid_fraction=valid_fraction,
                             texture=texture, masked_pixels=n_masked)
            if best is None or score > best['score']:
                best = candidate
    return best


def _choose_visual_case(paths, test_cube, test_invalid, crop_size):
    best = None
    for path in paths:
        mask, meta = load_test_mask(path)
        if mask.shape != test_cube.shape:
            raise ValueError(f"Mask and test cube shapes differ: {path}")
        for band in _visual_band_candidates(mask):
            candidate = _best_visual_location(
                mask, band, test_cube, test_invalid, crop_size
            )
            if candidate is not None and (best is None or candidate['score'] > best['score']):
                best = dict(candidate, path=path, mask=mask, meta=meta)
    return best


def export_visuals(n_visuals, rows):
    if n_visuals <= 0:
        return []
    if not torch.cuda.is_available():
        raise RuntimeError("Visual reconstruction export requires CUDA.")

    import matplotlib.pyplot as plt

    torch.cuda.set_device(int(ARGS.gpu))
    device = torch.device(f"cuda:{int(ARGS.gpu)}")
    pipeline = load_pipeline(
        device,
        region_batch=ARGS.region_batch,
        verify_region_batch=True,
            )
    cube, invalid, split, wavelengths = load_dataset(DATASET)
    _, _, test_cube = split_blocks(cube, split)
    _, _, test_invalid = split_blocks(invalid, split)
    if test_invalid.shape != test_cube.shape:
        raise ValueError("The valid-pixel mask does not match the held-out test cube.")
    wavelengths = np.asarray(wavelengths)
    mask_files = list_mask_files()
    # Keep previous figures separate; they must not be mistaken for new selections.
    visual_dir = VISDIR / "auto_interior"
    visual_dir.mkdir(parents=True, exist_ok=True)

    # Select among all completed, official mask repetitions and candidate bands.
    # Never pick by model accuracy; select only from observed masks and image texture.
    import re
    path_lookup = {}
    for path in mask_files:
        match = re.match(r"(.+)__s(\d+)__r(\d+)\.npz$", Path(path).name)
        if match:
            path_lookup[(match.group(1), int(match.group(2)), int(match.group(3)))] = path

    available = {}
    for item in rows:
        if item.get("status") != "ok":
            continue
        key = (str(item.get("scenario")), int(item.get("severity", -1)))
        repeat = int(item["repeat"])
        if (key[0], key[1], repeat) in path_lookup:
            available.setdefault(key, set()).add(repeat)

    candidate_cases = [
        ("spatio_spectral_block", 5),
        ("spatial_patch", 5),
        ("whole_spectrum_pixels", 5),
        ("consecutive_spectral_bands", 5),
        ("mixed_saturation", 5),
    ]
    requested = [case for case in candidate_cases if case in available][:int(n_visuals)]
    saved = []
    selection_records = []
    for number, (scenario, severity) in enumerate(requested, start=1):
        paths = [path_lookup[(scenario, severity, r)] for r in sorted(available[(scenario, severity)])]
        selection = _choose_visual_case(paths, test_cube, test_invalid, ARGS.crop_size)
        if selection is None:
            print(f"[{DATASET}] SKIP {scenario} S{severity}: no sufficiently interior, "
                  "textured masked crop among official cases; not using edge fallback.", flush=True)
            continue
        mask = selection['mask']
        repeat = int(selection['meta']['repeat'])
        band = selection['band']
        y0, y1, x0, x1 = selection['bounds']
        selection_records.append({
            'dataset': DATASET, 'scenario': scenario, 'severity': severity,
            'repeat': repeat, 'band': band, 'y0': y0, 'y1': y1,
            'x0': x0, 'x1': x1, 'valid_fraction': selection['valid_fraction'],
            'texture_score': selection['texture'], 'masked_pixels': selection['masked_pixels']
        })
        print(f"[{DATASET}] Chosen interior ROI: {scenario} S{severity} R{repeat} "
              f"band={band}, bounds=({y0}:{y1}, {x0}:{x1}), "
              f"valid={selection['valid_fraction']:.1%}, "
              f"texture={selection['texture']:.3f}", flush=True)
        local = mask[y0:y1, x0:x1, band]
        yx = np.argwhere(local)
        if len(yx) == 0:
            continue
        coords = np.column_stack([
            yx[:, 0] + y0,
            yx[:, 1] + x0,
            np.full(len(yx), band, dtype=np.int64),
        ]).astype(np.int64)

        print(
            f"[{DATASET}] Visual {number}/{len(requested)}: {scenario} "
            f"S{severity} R{repeat} | band {band} | {len(coords)} masked pixels",
            flush=True,
        )
        out = predict_mask(
            pipeline,
            test_cube,
            mask,
            coords,
            preview_progress=f"{DATASET} Visual {number}/{len(requested)} previews",
            region_progress=f"{DATASET} Visual {number}/{len(requested)} Region",
        )
        # Exactly the same spectral interpolation algorithm as the baseline.
        spectral_pred = reconstruct_baseline(test_cube, mask, coords, "spectral")
        tpssr_pred = np.asarray(out["prediction"], dtype=np.float32)

        original = np.asarray(test_cube[y0:y1, x0:x1, band], dtype=np.float32).copy()
        corrupted = original.copy()
        corrupted[local] = np.nan
        reconstructed = original.copy()
        spectral_image = corrupted.copy()
        for (y, x, _b), v_tp, v_sp in zip(coords, tpssr_pred, spectral_pred):
            yy, xx = int(y) - y0, int(x) - x0
            reconstructed[yy, xx] = v_tp
            if np.isfinite(v_sp):
                spectral_image[yy, xx] = v_sp
        # An unavailable Spectral prediction MUST stay blank, never ground truth.

        finite = original[np.isfinite(original)]
        if finite.size:
            vmin, vmax = np.quantile(finite, [0.01, 0.99])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        else:
            vmin, vmax = 0.0, 1.0
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0

        truth = np.asarray(test_cube[coords[:, 0], coords[:, 1], coords[:, 2]], dtype=float)
        good_spectral = np.isfinite(spectral_pred) & np.isfinite(truth)
        good_tp = np.isfinite(tpssr_pred) & np.isfinite(truth)
        def rmse(y, p, ok):
            return float(np.sqrt(np.mean((np.asarray(y)[ok] - np.asarray(p)[ok]) ** 2))) if np.any(ok) else float('nan')
        err_spec = np.full_like(original, np.nan)
        err_tp = np.full_like(original, np.nan)
        if np.any(good_spectral):
            good_coords = coords[good_spectral]
            err_spec[good_coords[:, 0]-y0, good_coords[:, 1]-x0] = np.abs(truth[good_spectral] - spectral_pred[good_spectral])
        if np.any(good_tp):
            good_coords = coords[good_tp]
            err_tp[good_coords[:, 0]-y0, good_coords[:, 1]-x0] = np.abs(truth[good_tp] - tpssr_pred[good_tp])

        cmap = plt.get_cmap("gray").copy()
        cmap.set_bad("white")
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
        for ax, image, title in zip(
            axes,
            [original, corrupted, spectral_image, reconstructed],
            ["Ground truth", "Masked input", "Spectral interpolation", "TPSSR V2.3"],
        ):
            ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(title)
            ax.axis("off")
        wl = float(wavelengths[band]) if band < len(wavelengths) else float("nan")
        fig.suptitle(
            f"{DATASET} | {scenario} | S{severity} R{repeat} | "
            f"band {band} ({wl:.1f} nm) | official-mask auto-selected interior crop | same crop and grayscale for all methods"
        )
        fig.tight_layout()
        stem = f"{number:02d}_{DATASET}_{scenario}_S{severity}_R{repeat}_band{band}_y{y0}_x{x0}"
        outpath = visual_dir / (stem + "_comparison.png")
        fig.savefig(outpath, dpi=180, bbox_inches="tight")
        plt.close(fig)

        valid_errors = np.concatenate([err_spec[np.isfinite(err_spec)], err_tp[np.isfinite(err_tp)]])
        maxerr = float(np.quantile(valid_errors, 0.99)) if valid_errors.size else 1.0
        maxerr = max(maxerr, 1e-6)
        ecmap = plt.get_cmap('magma').copy()
        ecmap.set_bad('white')
        errfig, erraxes = plt.subplots(1, 3, figsize=(14, 4.3))
        erraxes[0].imshow(local, cmap='gray', vmin=0, vmax=1, aspect='auto')
        erraxes[0].set_title('Mask (white = corrupted)')
        im = erraxes[1].imshow(err_spec, cmap=ecmap, vmin=0, vmax=maxerr, aspect='auto')
        erraxes[1].set_title(f"Spectral  |  RMSE {rmse(truth, spectral_pred, good_spectral):.5f}  |  Cov. {good_spectral.mean():.0%}", fontsize=10)
        erraxes[2].imshow(err_tp, cmap=ecmap, vmin=0, vmax=maxerr, aspect='auto')
        erraxes[2].set_title(f"TPSSR  |  RMSE {rmse(truth, tpssr_pred, good_tp):.5f}  |  Cov. {good_tp.mean():.0%}", fontsize=10)
        for ax in erraxes:
            ax.axis('off')
        errfig.colorbar(im, ax=erraxes[1:], label='Absolute reflectance error', fraction=0.03, pad=0.02)
        errfig.suptitle(f"{DATASET} | {scenario} S{severity} R{repeat} | band {band} | error only on masked pixels", fontsize=12)
        errfig.subplots_adjust(left=.02, right=.91, bottom=.07, top=.81, wspace=.10)
        errpath = visual_dir / (stem + "_error_maps.png")
        errfig.savefig(errpath, dpi=180, bbox_inches='tight')
        plt.close(errfig)
        saved.extend([str(outpath), str(errpath)])
        print('Saved:', outpath, 'and', errpath, flush=True)

    write_csv(visual_dir / "selected_rois.csv", selection_records)
    return saved

def main():
    args = parse_args()
    spec = resolve_experiment(args)
    if args.single:
        evaluate_one(spec, args)
        return
    if args.visuals < 0:
        raise ValueError("--visuals must be non-negative")
    allocation = [args.visuals // len(spec.tests)] * len(spec.tests)
    for index in range(args.visuals % len(spec.tests)):
        allocation[index] += 1
    for dataset, n_visuals in zip(spec.tests, allocation):
        command = [sys.executable, str(Path(__file__).resolve()),
                   *experiment_cli_args(spec, active_test=dataset),
                   "--visuals", str(n_visuals), "--gpu", str(args.gpu),
                   "--region-batch", str(args.region_batch),
                   "--crop-size", str(args.crop_size), "--single"]
        subprocess.run(command, cwd=ROOT, env=env_for(spec, dataset), check=True)
    combine(spec)


if __name__ == "__main__":
    main()
