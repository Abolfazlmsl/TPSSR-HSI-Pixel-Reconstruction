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
from experiment_config import add_experiment_args, experiment_cli_args, resolve_experiment, selected_results_folder


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
    parser.add_argument("--region-fixed-eval", action="store_true",
                        help="Read results produced by test.py --region-fixed-eval")
    parser.add_argument("--router", choices=["v23", "neighbor_v1", "neighbor_v2", "learned_fusion", "learned_fusion_guarded", "learned_router", "learned_router3", "learned_router3_rmse", "learned_router3_aligned", "learned_router3_lineaware", "learned_router3_lineaware_mse", "learned_router3_lineaware_rankcv", "learned_router3_lineaware_regretcv", "learned_router2_lineaware"], default="learned_router3_aligned")
    return parser.parse_args()


def results_folder(spec, router, region_fixed_eval=False):
    return selected_results_folder(spec, router) + (
        '__fixed_union_region' if region_fixed_eval else '')


def combine(spec, router="v23", region_fixed_eval=False) -> None:
    folder = results_folder(spec, router, region_fixed_eval)
    all_rows = []
    seen = set()
    for dataset in spec.tests:
        path = ROOT / "results" / spec.experiment_name / dataset / folder / "test_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing held-out evaluation data: {path}")
        for row in read_csv(path):
            key = (dataset, row.get("mask_file"))
            if key in seen:
                raise RuntimeError(f"Duplicate case: {key}")
            seen.add(key)
            all_rows.append(row)
    output = ROOT / "results" / spec.experiment_name / "combined" / folder
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
    from src.pipeline import load_pipeline, predict_mask, predict_mask_candidate, predict_mask_candidate_v2, predict_mask_learned_fusion, predict_mask_learned_fusion_guarded, predict_mask_learned_router, predict_mask_learned_router3
    if args.router == "neighbor_v1":
        predict_mask = predict_mask_candidate
    elif args.router == "neighbor_v2":
        predict_mask = predict_mask_candidate_v2
    elif args.router == "learned_fusion":
        predict_mask = predict_mask_learned_fusion
    elif args.router == "learned_fusion_guarded":
        predict_mask = predict_mask_learned_fusion_guarded
    elif args.router == "learned_router":
        predict_mask = predict_mask_learned_router
    elif args.router in ("learned_router3", "learned_router3_rmse", "learned_router3_aligned", "learned_router3_lineaware", "learned_router3_lineaware_mse", "learned_router3_lineaware_rankcv", "learned_router3_lineaware_regretcv", "learned_router2_lineaware"):
        predict_mask = predict_mask_learned_router3
    # The visual export functions resolve these names in this module's globals.
    globals().update(load_dataset=load_dataset, split_blocks=split_blocks,
                     list_mask_files=list_mask_files, load_test_mask=load_test_mask,
                     load_pipeline=load_pipeline, predict_mask=predict_mask)
    OUTDIR = Path(config.RESULTS_DIR) / results_folder(spec, args.router, args.region_fixed_eval)
    METRICS = OUTDIR / "test_metrics.csv"
    VISDIR = OUTDIR / "visuals"
    if not METRICS.exists():
        raise FileNotFoundError(f"Run test.py before evaluate.py: {METRICS}")
    rows = read_csv(METRICS)
    if not rows:
        raise RuntimeError(f"No test masks found in {METRICS}")
    result = summary(rows, DATASET)
    result["visuals"] = export_visuals(args.visuals) if args.visuals > 0 else []
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
        region_batch=ARGS.region_batch,
        verify_region_batch=True,
        learned_fusion=(ARGS.router in ("learned_fusion", "learned_fusion_guarded", "learned_router", "learned_router3", "learned_router3_rmse", "learned_router3_aligned", "learned_router3_lineaware", "learned_router3_lineaware_mse", "learned_router3_lineaware_rankcv", "learned_router3_lineaware_regretcv", "learned_router2_lineaware")),
        learned_router=(ARGS.router == "learned_router"),
        learned_router3=(ARGS.router == "learned_router3"),
        learned_router3_rmse=(ARGS.router == "learned_router3_rmse"),
        learned_router3_aligned=(ARGS.router == "learned_router3_aligned"),
        learned_router3_lineaware=(ARGS.router == "learned_router3_lineaware"),
        learned_router3_lineaware_mse=(ARGS.router == "learned_router3_lineaware_mse"),
        learned_router3_lineaware_rankcv=(ARGS.router == "learned_router3_lineaware_rankcv"),
        learned_router3_lineaware_regretcv=(ARGS.router == "learned_router3_lineaware_regretcv"),
        learned_router2_lineaware=(ARGS.router == "learned_router2_lineaware"),
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
        kw = dict(region_full_eval=True) if ARGS.region_fixed_eval else {}
        out = predict_mask(
            pipeline,
            test_cube,
            mask,
            coords,
            preview_progress=f"{DATASET} Visual {number}/{n_visuals} previews",
            region_progress=f"{DATASET} Visual {number}/{n_visuals} Region",
            **kw,
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
            ["Original", "Corrupted", "TPSSR"],
        ):
            ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(title)
            ax.axis("off")
        wl = float(wavelengths[band]) if band < len(wavelengths) else float("nan")
        route = out["final_route"]
        fig.suptitle(
            f"{DATASET} | {scenario} | Severity {severity} | Repeat {repeat} | "
            f"band {band} ({wl:.1f} nm) | "
            f"routes spectral/region={np.mean(route==0):.2f}/{np.mean(route==2):.2f}"
        )
        fig.tight_layout()
        outpath = VISDIR / (
            f"{number:02d}_{DATASET}_{scenario}_S{severity}_R{repeat}_band{band}.png"
        )
        fig.savefig(outpath, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(outpath))
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
                   "--crop-size", str(args.crop_size),
                   "--router", args.router, "--single"]
        if args.region_fixed_eval:
            command.append('--region-fixed-eval')
        subprocess.run(command, cwd=ROOT, env=env_for(spec, dataset), check=True)
    combine(spec, args.router, args.region_fixed_eval)


if __name__ == "__main__":
    main()
