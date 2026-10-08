#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate Final TPSSR metrics for one or many held-out acquisitions.

The same trained model is evaluated separately on every configured test core.
A combined cross-core summary is then written under results/<experiment>/combined/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"

from _launcher_common import env_for
from experiment_config import (
    add_experiment_args,
    describe,
    experiment_cli_args,
    resolve_experiment,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Final TPSSR test outputs.")
    add_experiment_args(p)
    p.add_argument(
        "--visuals",
        type=int,
        default=3,
        help="Total representative visuals across the configured test acquisitions.",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--deep-batch", type=int, default=16)
    p.add_argument("--region-batch", type=int, default=32)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmean(rows, key):
    values = []
    for r in rows:
        try:
            x = float(r[key])
        except Exception:
            continue
        if np.isfinite(x):
            values.append(x)
    return float(np.mean(values)) if values else float("nan")


def distribute_visuals(total, n_tests):
    total = max(0, int(total))
    result = [0] * max(1, int(n_tests))
    for i in range(total):
        result[i % len(result)] += 1
    return result


def combined_summary(spec):
    all_rows = []
    dataset_rows = []
    for test_name in spec.tests:
        test_dir = ROOT / "results" / spec.experiment_name / test_name / "final_tpssr"
        metrics = test_dir / "test_metrics.csv"
        summary_path = test_dir / "summary.json"
        if not metrics.exists():
            raise FileNotFoundError(
                f"Missing Final TPSSR metrics for {test_name}: {metrics}"
            )
        rows = read_csv(metrics)
        ok = [r for r in rows if r.get("status") == "ok"]
        for row in ok:
            row = dict(row)
            row.setdefault("dataset", test_name)
            all_rows.append(row)
        if summary_path.exists():
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            dataset_rows.append({
                "dataset": test_name,
                "masks": int(s.get("n_ok", len(ok))),
                "rmse": s.get("mean_rmse"),
                "mae": s.get("mean_mae"),
                "r2": s.get("mean_r2"),
                "sam_deg": s.get("mean_sam_deg"),
                "ssim": s.get("mean_ssim"),
                "coverage": s.get("mean_coverage"),
                "spectral_fraction": s.get("mean_final_spectral_fraction"),
                "deep_fraction": s.get("mean_final_deep_fraction"),
                "region_fraction": s.get("mean_final_region_fraction"),
            })

    outdir = ROOT / "results" / spec.experiment_name / "combined" / "final_tpssr"
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "all_test_metrics.csv", all_rows)
    write_csv(outdir / "dataset_summary.csv", dataset_rows)

    summary = {
        "experiment": spec.experiment_name,
        "train_datasets": list(spec.train),
        "test_datasets": list(spec.tests),
        "n_test_datasets": len(spec.tests),
        "n_ok_masks": len(all_rows),
        "mean_rmse": fmean(all_rows, "rmse"),
        "mean_mae": fmean(all_rows, "mae"),
        "mean_r2": fmean(all_rows, "r2"),
        "mean_sam_deg": fmean(all_rows, "sam_deg"),
        "mean_ssim": fmean(all_rows, "ssim"),
        "mean_coverage": fmean(all_rows, "coverage"),
        "mean_final_spectral_fraction": fmean(all_rows, "final_spectral_fraction"),
        "mean_final_deep_fraction": fmean(all_rows, "final_deep_fraction"),
        "mean_final_region_fraction": fmean(all_rows, "final_region_fraction"),
    }
    (outdir / "combined_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    lines = [
        "FINAL TPSSR COMBINED HELD-OUT EVALUATION",
        "=" * 76,
        f"Experiment: {spec.experiment_name}",
        f"Train: {', '.join(spec.train)}",
        f"Test: {', '.join(spec.tests)}",
        f"Successful masks across all test acquisitions: {len(all_rows)}",
        f"RMSE={summary['mean_rmse']:.6f}",
        f"MAE={summary['mean_mae']:.6f}",
        f"R2={summary['mean_r2']:.6f}",
        f"SAM={summary['mean_sam_deg']:.4f} deg",
        f"SSIM={summary['mean_ssim']:.4f}",
        f"Coverage={summary['mean_coverage']:.4f}",
    ]
    (outdir / "COMBINED_SUMMARY.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n" + "\n".join(lines))
    print("Saved combined results:", outdir)


def run_all_tests(args, spec):
    print(describe(spec))
    allocations = distribute_visuals(args.visuals, len(spec.tests))
    for i, (test_name, n_visuals) in enumerate(zip(spec.tests, allocations), start=1):
        print(
            f"\n{'='*90}\nEVALUATE TEST DATASET {i}/{len(spec.tests)}: {test_name}\n{'='*90}"
        )
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            *experiment_cli_args(spec, active_test=test_name),
            "--visuals",
            str(n_visuals),
            "--gpu",
            str(args.gpu),
            "--deep-batch",
            str(args.deep_batch),
            "--region-batch",
            str(args.region_batch),
            "--crop-size",
            str(args.crop_size),
            "--single",
        ]
        print("RUN:", " ".join(map(str, cmd)), flush=True)
        subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env_for(spec, test_name),
            check=True,
        )
    combined_summary(spec)


ARGS = parse_args()
SPEC = resolve_experiment(ARGS)
if not ARGS.single:
    run_all_tests(ARGS, SPEC)
    raise SystemExit(0)

DATASET = SPEC.active_test
os.environ.update(env_for(SPEC, DATASET))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import torch
import config
from src.evaluation_utils import load_dataset, split_blocks, list_mask_files, load_test_mask
from src.final_pipeline import load_pipeline, predict_mask

OUTDIR = Path(config.RESULTS_DIR) / "final_tpssr"
METRICS = OUTDIR / "test_metrics.csv"
VISDIR = OUTDIR / "visuals"


def aggregate(rows, key_name):
    out = []
    keys = sorted({str(r[key_name]) for r in rows})
    for key in keys:
        rr = [r for r in rows if str(r[key_name]) == key]
        out.append({
            key_name: key,
            "masks": len(rr),
            "rmse": fmean(rr, "rmse"),
            "mae": fmean(rr, "mae"),
            "r2": fmean(rr, "r2"),
            "sam_deg": fmean(rr, "sam_deg"),
            "ssim": fmean(rr, "ssim"),
            "coverage": fmean(rr, "coverage"),
            "spectral_fraction": fmean(rr, "final_spectral_fraction"),
            "deep_fraction": fmean(rr, "final_deep_fraction"),
            "region_fraction": fmean(rr, "final_region_fraction"),
        })
    return out


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


def crop_bounds(mask2d, crop_size):
    ys, xs = np.where(mask2d)
    h, w = mask2d.shape
    size = int(min(crop_size, h, w))
    if len(ys) == 0:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = int(np.median(ys)), int(np.median(xs))
    y0 = max(0, min(cy - size // 2, h - size))
    x0 = max(0, min(cx - size // 2, w - size))
    return y0, y0 + size, x0, x0 + size


def export_visuals(n_visuals):
    if n_visuals <= 0:
        return []
    if not torch.cuda.is_available():
        raise RuntimeError("Visual reconstruction export requires CUDA.")

    import matplotlib.pyplot as plt

    torch.cuda.set_device(int(ARGS.gpu))
    device = torch.device(f"cuda:{int(ARGS.gpu)}")
    pipeline = load_pipeline(
        device,
        deep_batch=ARGS.deep_batch,
        region_batch=ARGS.region_batch,
        verify_region_batch=True,
    )
    cube, _, split, wavelengths = load_dataset(DATASET)
    _, _, test_cube = split_blocks(cube, split)
    wavelengths = np.asarray(wavelengths)
    mask_files = list_mask_files()
    VISDIR.mkdir(parents=True, exist_ok=True)

    candidate_cases = [
        ("whole_spectrum_pixels", 5, 1),
        ("spatio_spectral_block", 5, 1),
        ("mixed_saturation", 5, 1),
        ("consecutive_spectral_bands", 5, 1),
        ("spatial_patch", 5, 1),
    ]
    requested = candidate_cases[: int(n_visuals)]

    saved = []
    for number, (scenario, severity, repeat) in enumerate(requested, start=1):
        path = find_case(mask_files, scenario, severity, repeat)
        if path is None:
            continue
        mask, meta = load_test_mask(path)
        mask = np.asarray(mask, dtype=bool)
        band = representative_band(mask)
        y0, y1, x0, x1 = crop_bounds(mask[:, :, band], ARGS.crop_size)
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
            f"[{DATASET}] Visual {number}/{n_visuals}: {scenario} "
            f"S{severity} R{repeat} | band {band} | {len(coords)} masked pixels"
        )
        out = predict_mask(
            pipeline,
            test_cube,
            mask,
            coords,
            preview_progress=f"{DATASET} Visual {number}/{n_visuals} previews",
            region_progress=f"{DATASET} Visual {number}/{n_visuals} Region",
        )
        original = np.asarray(test_cube[y0:y1, x0:x1, band], dtype=np.float32).copy()
        corrupted = original.copy()
        corrupted[local] = np.nan
        reconstructed = original.copy()
        for (y, x, _b), value in zip(coords, out["prediction"]):
            reconstructed[int(y) - y0, int(x) - x0] = float(value)

        finite = original[np.isfinite(original)]
        if finite.size:
            vmin, vmax = np.quantile(finite, [0.01, 0.99])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        else:
            vmin, vmax = 0.0, 1.0

        cmap = plt.get_cmap("gray").copy()
        cmap.set_bad("white")
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
        for ax, image, title in zip(
            axes,
            [original, corrupted, reconstructed],
            ["Original", "Corrupted", "Final TPSSR"],
        ):
            ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(title)
            ax.axis("off")
        wl = float(wavelengths[band]) if band < len(wavelengths) else float("nan")
        route = out["final_route"]
        fig.suptitle(
            f"{DATASET} | {scenario} | Severity {severity} | Repeat {repeat} | "
            f"band {band} ({wl:.1f} nm) | "
            f"routes S/D/R={np.mean(route==0):.2f}/{np.mean(route==1):.2f}/{np.mean(route==2):.2f}"
        )
        fig.tight_layout()
        outpath = VISDIR / (
            f"{number:02d}_{DATASET}_{scenario}_S{severity}_R{repeat}_band{band}.png"
        )
        fig.savefig(outpath, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(outpath))
    return saved


def main_single():
    if not METRICS.exists():
        raise SystemExit(
            f"Test metrics not found: {METRICS}\nRun test.py for {DATASET} first."
        )
    rows = read_csv(METRICS)
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        raise RuntimeError("No successful test rows were found.")

    write_csv(OUTDIR / "scenario_summary.csv", aggregate(ok, "scenario"))
    write_csv(OUTDIR / "severity_summary.csv", aggregate(ok, "severity"))

    worker0 = {}
    p0 = OUTDIR / "worker.rank0.json"
    if p0.exists():
        worker0 = json.loads(p0.read_text(encoding="utf-8"))

    summary = {
        "architecture": (
            "Stage1(Spectral vs Deep-v3.1) -> hard-case gate -> "
            "Frozen V27 Stage-A -> V13 TopologyFirst V1"
        ),
        "experiment": config.EXPERIMENT_NAME,
        "train_datasets": list(config.TRAIN_DATASET_FOLDERS),
        "test_datasets": list(config.TEST_DATASET_FOLDERS),
        "dataset": DATASET,
        "n_requested": len(rows),
        "n_ok": len(ok),
        "n_failed": len(rows) - len(ok),
        "mean_rmse": fmean(ok, "rmse"),
        "mean_mae": fmean(ok, "mae"),
        "mean_r2": fmean(ok, "r2"),
        "mean_sam_deg": fmean(ok, "sam_deg"),
        "mean_ssim": fmean(ok, "ssim"),
        "mean_coverage": fmean(ok, "coverage"),
        "mean_final_spectral_fraction": fmean(ok, "final_spectral_fraction"),
        "mean_final_deep_fraction": fmean(ok, "final_deep_fraction"),
        "mean_final_region_fraction": fmean(ok, "final_region_fraction"),
        "mean_runtime_sec_per_mask": fmean(ok, "runtime_sec"),
        **worker0,
    }

    visual_paths = export_visuals(int(ARGS.visuals))
    summary["visuals"] = visual_paths
    (OUTDIR / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )

    lines = [
        "FINAL TPSSR EVALUATION",
        "=" * 72,
        summary["architecture"],
        f"Experiment: {config.EXPERIMENT_NAME}",
        f"Dataset: {DATASET}",
        f"Masks: {summary['n_ok']}/{summary['n_requested']} successful",
        f"RMSE={summary['mean_rmse']:.6f}",
        f"MAE={summary['mean_mae']:.6f}",
        f"R2={summary['mean_r2']:.6f}",
        f"SAM={summary['mean_sam_deg']:.4f} deg",
        f"SSIM={summary['mean_ssim']:.4f}",
        f"Coverage={summary['mean_coverage']:.4f}",
        "",
        "Mean final route fractions:",
        f"  Spectral={summary['mean_final_spectral_fraction']:.4f}",
        f"  Deep-v3.1={summary['mean_final_deep_fraction']:.4f}",
        f"  Final Region={summary['mean_final_region_fraction']:.4f}",
        "",
        f"Gate threshold: {summary.get('gate_threshold', float('nan'))}",
        f"Visuals exported for this dataset: {len(visual_paths)}",
    ]
    (OUTDIR / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main_single()
