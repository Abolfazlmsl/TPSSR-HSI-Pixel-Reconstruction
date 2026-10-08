#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate Final TPSSR test metrics and export representative visuals."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"
SUPPORTED = [*(f"SWIR{i}" for i in range(1, 6)), *(f"VNIR{i}" for i in range(1, 6))]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Final TPSSR test outputs.")
    p.add_argument("--dataset", required=True, help="SWIR1..SWIR5 or VNIR1..VNIR5")
    p.add_argument("--visuals", type=int, default=3, choices=[0, 1, 2, 3])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--deep-batch", type=int, default=16)
    p.add_argument("--region-batch", type=int, default=32)
    p.add_argument("--crop-size", type=int, default=128)
    return p.parse_args()


ARGS = parse_args()
DATASET = ARGS.dataset.upper()
if DATASET not in SUPPORTED:
    raise SystemExit(f"Dataset must be one of: {', '.join(SUPPORTED)}")

os.environ["TPSSR_PROJECT_DIR"] = str(ROOT)
os.environ["HSI_TEST_DATASET"] = DATASET
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import torch
import config
from src.evaluation_utils import load_dataset, split_blocks, list_mask_files, load_test_mask
from src.final_pipeline import load_pipeline, predict_mask

OUTDIR = Path(config.RESULTS_DIR) / "final_tpssr"
METRICS = OUTDIR / "test_metrics.csv"
VISDIR = OUTDIR / "visuals"


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    keys = list(rows[0].keys())
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


def aggregate(rows, key_name):
    out = []
    keys = sorted({str(r[key_name]) for r in rows})
    for key in keys:
        rr = [r for r in rows if str(r[key_name]) == key]
        out.append(
            {
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
            }
        )
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
    cube, _, split, wavelengths = load_dataset()
    _, _, test_cube = split_blocks(cube, split)
    wavelengths = np.asarray(wavelengths)
    mask_files = list_mask_files()
    VISDIR.mkdir(parents=True, exist_ok=True)

    requested = [
        ("whole_spectrum_pixels", 5, 1),
        ("spatio_spectral_block", 5, 1),
        ("mixed_saturation", 5, 1),
    ][:n_visuals]

    saved = []
    for number, (scenario, severity, repeat) in enumerate(requested, start=1):
        path = find_case(mask_files, scenario, severity, repeat)
        if path is None:
            print("Visual case not found, skipping:", scenario, severity, repeat)
            continue
        mask, meta = load_test_mask(path)
        mask = np.asarray(mask, dtype=bool)
        band = representative_band(mask)
        y0, y1, x0, x1 = crop_bounds(mask[:, :, band], ARGS.crop_size)
        local = mask[y0:y1, x0:x1, band]
        yx = np.argwhere(local)
        if len(yx) == 0:
            print("Visual case has no masked pixels in representative band, skipping:", path.name)
            continue
        coords = np.column_stack(
            [
                yx[:, 0] + y0,
                yx[:, 1] + x0,
                np.full(len(yx), band, dtype=np.int64),
            ]
        ).astype(np.int64)

        print(
            f"Visual {number}/{n_visuals}: {scenario} S{severity} R{repeat} | "
            f"band {band} | {len(coords)} masked pixels"
        )
        out = predict_mask(
            pipeline,
            test_cube,
            mask,
            coords,
            preview_progress=f"Visual {number}/{n_visuals} previews",
            region_progress=f"Visual {number}/{n_visuals} Region",
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
            f"{scenario} | Severity {severity} | Repeat {repeat} | "
            f"band {band} ({wl:.1f} nm) | "
            f"routes S/D/R={np.mean(route==0):.2f}/{np.mean(route==1):.2f}/{np.mean(route==2):.2f}"
        )
        fig.tight_layout()
        outpath = VISDIR / f"{number:02d}_{scenario}_S{severity}_R{repeat}_band{band}.png"
        fig.savefig(outpath, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(outpath))
    return saved


def main():
    if not METRICS.exists():
        raise SystemExit(
            f"Test metrics not found: {METRICS}\n"
            f"Run: python test.py --dataset {DATASET} --gpus 2 --max-masks 200"
        )
    rows = read_csv(METRICS)
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        raise RuntimeError("No successful test rows were found.")

    scenario_rows = aggregate(ok, "scenario")
    severity_rows = aggregate(ok, "severity")
    write_csv(OUTDIR / "scenario_summary.csv", scenario_rows)
    write_csv(OUTDIR / "severity_summary.csv", severity_rows)

    worker0 = {}
    p0 = OUTDIR / "worker.rank0.json"
    if p0.exists():
        worker0 = json.loads(p0.read_text(encoding="utf-8"))

    summary = {
        "architecture": "Stage1(Spectral vs Deep-v3.1) -> hard-case gate -> Frozen V27 Stage-A -> V13 TopologyFirst V1",
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
        f"Visuals exported: {len(visual_paths)}",
    ]
    (OUTDIR / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print("Saved:", OUTDIR / "summary.json")
    print("Saved:", OUTDIR / "scenario_summary.csv")
    print("Saved:", OUTDIR / "severity_summary.csv")
    if visual_paths:
        print("Visuals:")
        for path in visual_paths:
            print(" ", path)


if __name__ == "__main__":
    main()
