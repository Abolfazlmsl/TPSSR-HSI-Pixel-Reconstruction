#!/usr/bin/env python3
"""Compute mean-reflectance-normalized errors for the frozen SWIR 8x2 benchmark.

This reads existing metrics, saved benchmark masks, and calibrated reflectance.
It does not load reconstruction networks, change predictions, or use a GPU.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EXPERIMENT = "swir__train_1-2-3-4-6-7-8-9__test_5-10"
DATASETS = ("SWIR5", "SWIR10")
METHODS = ("tpssr", "spectral", "spatial", "spatio_spectral", "pca", "random_forest", "pfgin")
SIMPLE = {"isolated_voxels", "single_detector_line", "multiple_detector_lines", "partial_vertical_lines"}


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."), help="Project root")
    args = parser.parse_args()
    root = args.project.resolve()

    # The original index/coordinate selection functions are reused rather than
    # reimplementing the benchmark sampling procedure.
    os.environ["TPSSR_PROJECT_DIR"] = str(root)
    os.environ["HSI_TEST_DATASET"] = "SWIR5"
    os.environ["HSI_ACTIVE_TEST_DATASET"] = "SWIR5"
    os.environ["HSI_TEST_DATASETS"] = "SWIR5,SWIR10"
    os.environ["HSI_TRAIN_DATASETS"] = "SWIR1,SWIR2,SWIR3,SWIR4,SWIR6,SWIR7,SWIR8,SWIR9"
    os.environ["HSI_EXPERIMENT_NAME"] = EXPERIMENT
    sys.path.insert(0, str(root / "core"))
    import config
    from src.evaluation_utils import load_test_mask, benchmark_case_index, select_eval_coords

    results_root = root / "results" / EXPERIMENT
    all_cases = []
    for dataset in DATASETS:
        gt_path = root / "processed_data" / dataset / "calibrated_reflectance.npy"
        tp_path = results_root / dataset / "tpssr" / "test_metrics.csv"
        masks_dir = root / "simulated_saturation" / EXPERIMENT / dataset / "masks"
        for path in (gt_path, tp_path, masks_dir):
            if not path.exists():
                raise SystemExit(f"Missing required input: {path}")

        rows = [r for r in load_rows(tp_path) if r.get("status") == "ok"]
        if len(rows) != 100:
            raise SystemExit(f"Expected 100 successful TPSSR rows for {dataset}, got {len(rows)}")
        cube = np.load(gt_path, mmap_mode="r")
        test_size = max(1, int(round(cube.shape[0] * config.HELD_OUT_TEST_FRACTION)))
        gt_test = cube[-test_size:]
        print(f"\n{dataset}: {len(rows)} masks; held-out shape {gt_test.shape}", flush=True)

        for i, row in enumerate(rows, start=1):
            mask_path = masks_dir / row["mask_file"]
            if not mask_path.is_file():
                raise SystemExit(f"Missing mask: {mask_path}")
            mask, meta = load_test_mask(mask_path)
            if mask.shape != gt_test.shape:
                raise SystemExit(f"Mask/GT shape mismatch: {mask_path.name} {mask.shape} vs {gt_test.shape}")
            idx = benchmark_case_index(mask_path)
            coords = select_eval_coords(mask, int(config.SEED) + idx)
            expected = int(float(row["target_voxels"]))
            if len(coords) != expected:
                raise SystemExit(f"Evaluation coords mismatch for {dataset}/{mask_path.name}: {len(coords)} != {expected}")
            gt = np.asarray(gt_test[coords[:, 0], coords[:, 1], coords[:, 2]], dtype=np.float64)
            gt = gt[np.isfinite(gt)]
            if len(gt) != expected:
                raise SystemExit(f"Nonfinite ground truth for {dataset}/{mask_path.name}: {len(gt)}/{expected}")
            mean_abs = float(np.mean(np.abs(gt)))
            if mean_abs <= 1e-12:
                raise SystemExit(f"Zero mean absolute reflectance: {dataset}/{mask_path.name}")
            all_cases.append({
                "dataset": dataset,
                "scenario": meta["scenario"],
                "severity": int(meta["severity"]),
                "repeat": int(meta["repeat"]),
                "mask_file": mask_path.name,
                "target_voxels": expected,
                "mean_true_reflectance": float(np.mean(gt)),
                "mean_abs_true_reflectance": mean_abs,
                "std_true_reflectance": float(np.std(gt)),
            })
            del mask, coords, gt
            if i % 20 == 0:
                print(f"  Ground truth {i}/100 checked", flush=True)
        del cube, gt_test

    out = root / "NRMSE_PERCENT_AUDIT"
    out.mkdir(exist_ok=True)
    write_rows(out / "ground_truth_reference_by_mask.csv", all_cases)
    by_key = {(c["dataset"], c["scenario"], c["severity"], c["repeat"]): c for c in all_cases}

    details = []
    for dataset in DATASETS:
        for method in METHODS:
            path = results_root / dataset / method / ("test_metrics.csv" if method == "tpssr" else "metrics.csv")
            if not path.is_file():
                print(f"Skipping missing results: {path}")
                continue
            for r in load_rows(path):
                if method == "tpssr" and r.get("status") != "ok":
                    continue
                key = (dataset, r["scenario"], int(r["severity"]), int(r["repeat"]))
                if key not in by_key:
                    raise SystemExit(f"Unexpected method case for {method}: {key}")
                ref = by_key[key]
                rmse = float(r["rmse"]) if r.get("rmse", "") else float("nan")
                mae = float(r["mae"]) if r.get("mae", "") else float("nan")
                coverage = float(r["coverage"])
                total = int(float(r.get("target_voxels", r.get("target_n", 0))))
                if total != ref["target_voxels"]:
                    raise SystemExit(f"Target count mismatch: {method} {key} {total} != {ref['target_voxels']}")
                denominator = ref["mean_abs_true_reflectance"]
                details.append({
                    "dataset": dataset, "method": method,
                    "scenario": r["scenario"], "severity": int(r["severity"]),
                    "repeat": int(r["repeat"]), "target_voxels": total,
                    "mean_abs_true_reflectance": denominator,
                    "coverage": coverage,
                    "rmse": rmse, "mae": mae,
                    "nrmse_percent": float(rmse / denominator * 100) if np.isfinite(rmse) else float("nan"),
                    "nmae_percent": float(mae / denominator * 100) if np.isfinite(mae) else float("nan"),
                })

    write_rows(out / "normalized_errors_by_mask.csv", details)
    summaries = []
    for method in METHODS:
        data = [r for r in details if r["method"] == method]
        if not data:
            continue
        for subset_name, subset in (
            ("ALL", data),
            ("SWIR5", [r for r in data if r["dataset"] == "SWIR5"]),
            ("SWIR10", [r for r in data if r["dataset"] == "SWIR10"]),
            ("SIMPLE_4_SCENARIOS", [r for r in data if r["scenario"] in SIMPLE]),
            ("FULL_COVERAGE_ONLY", [r for r in data if r["coverage"] >= 1 - 1e-9]),
        ):
            valid = [r for r in subset if np.isfinite(r["nrmse_percent"])]
            if not valid:
                continue
            summaries.append({
                "method": method, "subset": subset_name,
                "total_masks": len(subset), "valid_rmse_masks": len(valid),
                "full_coverage_masks": sum(r["coverage"] >= 1 - 1e-9 for r in subset),
                "mean_coverage_percent": float(100 * np.mean([r["coverage"] for r in subset])),
                "mean_per_mask_nrmse_percent": float(np.mean([r["nrmse_percent"] for r in valid])),
                "mean_per_mask_nmae_percent": float(np.mean([r["nmae_percent"] for r in valid])),
            })
    write_rows(out / "normalized_errors_summary.csv", summaries)

    print("\nMean per-mask normalized errors (% of mean absolute true reflectance):")
    print(f"{'Method':18} {'Subset':21} {'NRMSE %':>10} {'NMAE %':>10} {'Valid':>7} {'Full':>7}")
    for r in summaries:
        if r["subset"] in ("ALL", "SIMPLE_4_SCENARIOS"):
            print(f"{r['method']:18} {r['subset']:21} {r['mean_per_mask_nrmse_percent']:10.3f} {r['mean_per_mask_nmae_percent']:10.3f} {r['valid_rmse_masks']:7} {r['full_coverage_masks']:7}")
    print("\nSaved:", out)
    print("Note: These are RMSE and MAE divided by the average absolute true reflectance per mask,")
    print("NOT MAPE. Errors of methods with incomplete coverage are evaluated only where they predicted.")


if __name__ == "__main__":
    main()
