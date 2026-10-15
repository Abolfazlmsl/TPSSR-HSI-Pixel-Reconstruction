#!/usr/bin/env python3
"""Compare TPSSR with completed baselines across scenarios and severities.

Run:
    python plot_method_comparison.py --experiment-dir /path/to/experiment

Optional arguments:
    --tpssr-csv /path/to/all_test_metrics.csv
    --baseline-zip /path/to/previous_results.zip
    --output-dir /path/to/plots
    --exclude-pfgin
    --metrics rmse mae sam_deg r2 ssim

When no paths are specified, the script looks for an experiment directory and
TPSSR metrics under commonly used results folders. DDS2M is never included.

Pooled RMSE uses the number of actually reconstructed voxels as weights:
    sqrt(sum(n_valid * rmse**2) / sum(n_valid))

R2 and SSIM are averaged over masks with finite reported values. These means
are NOT global pooled R2/SSIM. The output CSV includes valid-mask counts.
The baseline metrics.csv files do not contain SSIM in this experiment, so
only TPSSR has an SSIM curve unless other methods actually report SSIM.
Missing metrics are never estimated or filled with zeros.

Methods with incomplete reconstruction coverage are shown with hollow markers.
Do not interpret a low RMSE from incomplete coverage as directly comparable
to complete reconstruction.
"""

from __future__ import annotations

import argparse
import io
import math
import re
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Optional path overrides. Leave these as None for automatic discovery.
EXPERIMENT_DIR = None
TPSSR_CSV = None
BASELINE_ZIP = None
OUTPUT_DIR = None

METHOD_FOLDERS = {
    "spectral": "Spectral",
    "spatial": "Spatial",
    "spatio_spectral": "Spatio-spectral",
    "pca": "PCA",
    "random_forest": "RF",
    "pfgin": "PFGIN",
}
METHOD_COLORS = {
    "Spectral": "#1f77b4",
    "Spatial": "#ff7f0e",
    "Spatio-spectral": "#2ca02c",
    "PCA": "#d62728",
    "RF": "#9467bd",
    "PFGIN": "#8c564b",
    "TPSSR": "#111111",
}

SCENARIOS = [
    "band_wide_saturation",
    "consecutive_spectral_bands",
    "isolated_voxels",
    "mixed_saturation",
    "multiple_detector_lines",
    "partial_vertical_lines",
    "single_detector_line",
    "spatial_patch",
    "spatio_spectral_block",
    "whole_spectrum_pixels",
]
SEVERITIES = (1, 2, 3, 4, 5)
METRIC_LABELS = {
    "rmse": "Pooled RMSE",
    "mae": "Pooled MAE",
    "sam_deg": "Mean SAM (degrees)",
    "r2": "Mean per-mask R²",
    "ssim": "Mean per-mask SSIM",
}
METRIC_FIELDS = {
    "rmse": "pooled_rmse",
    "mae": "pooled_mae",
    "sam_deg": "mean_sam_deg",
    "r2": "mean_r2",
    "ssim": "mean_ssim",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_DIR)
    parser.add_argument("--baseline-zip", type=Path, default=BASELINE_ZIP)
    parser.add_argument("--tpssr-csv", type=Path, default=TPSSR_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--exclude-pfgin", action="store_true")
    parser.add_argument("--metrics", nargs="+", choices=tuple(METRIC_LABELS),
                        default=["rmse", "mae", "sam_deg", "r2", "ssim"])
    return parser.parse_args()


def search_roots():
    roots = [Path.cwd(), Path(__file__).resolve().parent]
    parents = []
    for root in roots:
        parents.extend([root, root / "results", root / "evaluation_results", root / "outputs"])
    return list(dict.fromkeys(parents))


def discover_experiment():
    for root in search_roots():
        if (root / "SWIR5" / "spectral" / "metrics.csv").is_file():
            return root
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "SWIR5" / "spectral" / "metrics.csv").is_file():
                return child
    return None


def discover_zip():
    candidates = []
    for root in search_roots():
        if not root.is_dir():
            continue
        candidates.extend(root.glob("swir__train_*__test_*.zip"))
    return sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)[0] if candidates else None


def choose_tpssr_csv(experiment_dir, explicitly_given):
    if explicitly_given is not None:
        path = Path(explicitly_given).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"TPSSR metrics not found: {path}")
        return path

    candidates = []
    if experiment_dir is not None:
        combined = experiment_dir / "combined"
        for name in ("final_tpssr_stage1_rank_v4", "final_tpssr_spectral_safe_stage1_v2",
                     "final_tpssr_joint_v3", "final_tpssr_spectral_safe", "final_tpssr"):
            path = combined / name / "all_test_metrics.csv"
            if path.is_file():
                return path
        if combined.is_dir():
            candidates.extend(combined.glob("*/all_test_metrics.csv"))

    for root in search_roots():
        if not root.is_dir():
            continue
        candidates.extend(root.glob("all_test_metrics*.csv"))
        candidates.extend(root.glob("combined/*/all_test_metrics.csv"))
        for child in root.iterdir():
            if child.is_dir():
                candidates.extend((child / "combined").glob("*/all_test_metrics.csv"))
    candidates = sorted(set(p for p in candidates if p.is_file()),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            "Cannot find TPSSR all_test_metrics.csv. Supply --tpssr-csv /path/to/file."
        )
    if len(candidates) > 1:
        print(f"[INFO] Selected TPSSR metrics: {candidates[0]}")
        print("[INFO] Use --tpssr-csv to select a different Router version.")
    return candidates[0]


def safe_file_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def baseline_frames(experiment_dir, baseline_zip, include_pfgin=True):
    wanted = {key: label for key, label in METHOD_FOLDERS.items()
              if include_pfgin or key != "pfgin"}
    frames = []

    if baseline_zip is not None:
        zip_path = Path(baseline_zip).expanduser()
        if not zip_path.is_file():
            raise FileNotFoundError(f"Baseline ZIP not found: {zip_path}")
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                parts = Path(member).parts
                if len(parts) < 3 or parts[-1] != "metrics.csv":
                    continue
                method_folder = parts[-2]
                dataset = parts[-3]
                if method_folder not in wanted or not re.fullmatch(r"SWIR\d+|VNIR\d+", dataset, re.I):
                    continue
                df = pd.read_csv(io.BytesIO(archive.read(member)))
                df["dataset"] = dataset
                df["method"] = wanted[method_folder]
                frames.append(df)
    else:
        if experiment_dir is None or not experiment_dir.is_dir():
            raise FileNotFoundError("Baseline results not found. Supply --experiment-dir or --baseline-zip.")
        for dataset_path in sorted(experiment_dir.iterdir()):
            if not dataset_path.is_dir() or not re.fullmatch(r"SWIR\d+|VNIR\d+", dataset_path.name, re.I):
                continue
            for folder, name in wanted.items():
                csv_path = dataset_path / folder / "metrics.csv"
                if not csv_path.is_file():
                    print(f"[WARNING] Missing baseline: {csv_path}")
                    continue
                df = pd.read_csv(csv_path)
                df["dataset"] = dataset_path.name
                df["method"] = name
                frames.append(df)

    if not frames:
        raise RuntimeError("No baseline metrics found. Check the experiment path or ZIP.")
    return pd.concat(frames, ignore_index=True, sort=False)


def standardize(df, *, is_tpssr):
    df = df.copy()
    required = {"dataset", "scenario", "severity", "repeat", "rmse"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metrics CSV is missing required fields: {sorted(missing)}")

    if is_tpssr:
        df["method"] = "TPSSR"
        if "filled_voxels" in df:
            df["valid_n"] = df["filled_voxels"]
        if "target_voxels" in df:
            df["target_n"] = df["target_voxels"]
        if "status" in df:
            df = df[df["status"].astype(str).str.lower().eq("ok")].copy()
    if "valid_n" not in df or "target_n" not in df:
        raise ValueError("Metrics must supply valid_n/target_n or filled_voxels/target_voxels.")

    for col in ("severity", "repeat", "rmse", "mae", "sam_deg", "r2", "ssim",
                "valid_n", "target_n", "sam_pixels"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["scenario"] = df["scenario"].astype(str).str.strip()
    df["dataset"] = df["dataset"].astype(str).str.strip()
    df = df[df["severity"].isin(SEVERITIES)].copy()
    if "mae" not in df:
        df["mae"] = np.nan
    if "sam_deg" not in df:
        df["sam_deg"] = np.nan
    if "sam_pixels" not in df:
        df["sam_pixels"] = np.nan
    for metric in ("r2", "ssim"):
        if metric not in df:
            df[metric] = np.nan
    return df


def check_mask_keys(baselines, tpssr):
    join_columns = ["dataset", "scenario", "severity", "repeat"]
    tpssr_keys = set(map(tuple, tpssr[join_columns].drop_duplicates().to_numpy()))
    out = []
    for method, group in baselines.groupby("method"):
        keys = set(map(tuple, group[join_columns].drop_duplicates().to_numpy()))
        out.append((method, len(keys), len(keys - tpssr_keys), len(tpssr_keys - keys)))
    for method, n, extra, missing in sorted(out):
        print(f"[CHECK] {method}: {n} masks, {extra} extra keys, {missing} missing keys vs TPSSR")
    if any(extra or missing for _, _, extra, missing in out):
        print("[WARNING] Some baselines use different mask keys. Interpret comparisons cautiously.")


def weighted_rmse(g):
    errors = pd.to_numeric(g["rmse"], errors="coerce").to_numpy(dtype=float)
    count = pd.to_numeric(g["valid_n"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(errors) & np.isfinite(count) & (count > 0)
    return float(np.sqrt(np.sum(errors[keep] ** 2 * count[keep]) / np.sum(count[keep]))) if keep.any() else np.nan


def weighted_mean_metric(g, name, weight_name="valid_n"):
    vals = pd.to_numeric(g[name], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(g[weight_name], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
    return float(np.average(vals[keep], weights=weights[keep])) if keep.any() else np.nan


def aggregate_group(g, dataset, scenario, severity, method):
    valid = pd.to_numeric(g["valid_n"], errors="coerce").fillna(0).clip(lower=0)
    target = pd.to_numeric(g["target_n"], errors="coerce").fillna(0).clip(lower=0)
    valid_count = float(valid.sum())
    target_count = float(target.sum())
    return {
        "dataset": dataset,
        "scenario": scenario,
        "severity": int(severity),
        "method": method,
        "pooled_rmse": weighted_rmse(g),
        "pooled_mae": weighted_mean_metric(g, "mae"),
        "mean_sam_deg": weighted_mean_metric(g, "sam_deg", "sam_pixels")
        if np.isfinite(g["sam_pixels"].to_numpy(dtype=float)).any()
        else weighted_mean_metric(g, "sam_deg"),
        # Per-mask means: pooled R2/SSIM cannot be recovered from these CSVs.
        "mean_r2": float(pd.to_numeric(g["r2"], errors="coerce").mean()),
        "mean_ssim": float(pd.to_numeric(g["ssim"], errors="coerce").mean()),
        "valid_r2_masks": int(np.isfinite(pd.to_numeric(g["r2"], errors="coerce")).sum()),
        "valid_ssim_masks": int(np.isfinite(pd.to_numeric(g["ssim"], errors="coerce")).sum()),
        "coverage": valid_count / target_count if target_count > 0 else np.nan,
        "valid_voxels": int(valid_count),
        "target_voxels": int(target_count),
        "n_masks": len(g),
        "n_valid_rmse_masks": int(np.isfinite(pd.to_numeric(g["rmse"], errors="coerce")).sum()),
    }


def build_summary(all_df):
    rows = []
    cols = ["dataset", "scenario", "severity", "method"]
    for keys, group in all_df.groupby(cols, dropna=False):
        rows.append(aggregate_group(group, *keys))
    for keys, group in all_df.groupby(["scenario", "severity", "method"], dropna=False):
        scenario, severity, method = keys
        rows.append(aggregate_group(group, "ALL", scenario, severity, method))
    return pd.DataFrame(rows)


def plot_single(summary, dataset, scenario, metric, methods, output):
    data = summary[(summary["dataset"] == dataset) & (summary["scenario"] == scenario)]
    if data.empty:
        return False
    field = METRIC_FIELDS[metric]
    fig, ax = plt.subplots(figsize=(11, 6.4))
    drawn = 0
    for method in methods:
        subset = data[data["method"] == method].set_index("severity")
        if subset.empty:
            continue
        x = np.asarray(SEVERITIES)
        y = np.array([subset.at[k, field] if k in subset.index else np.nan for k in x], dtype=float)
        cov = np.array([subset.at[k, "coverage"] if k in subset.index else np.nan for k in x], dtype=float)
        if not np.isfinite(y).any():
            continue
        color = METHOD_COLORS[method]
        is_tpssr = method == "TPSSR"
        ax.plot(x, y, color=color, lw=3.4 if is_tpssr else 2.2,
                marker="o", markersize=7.5 if is_tpssr else 6.6,
                label=method, zorder=4 if is_tpssr else 2)
        incomplete = np.isfinite(y) & np.isfinite(cov) & (cov < 0.9999)
        if incomplete.any():
            ax.scatter(x[incomplete], y[incomplete], marker="o", s=90,
                       edgecolors=color, facecolors="white", linewidths=2,
                       zorder=8)
        drawn += 1
    ax.set_title(scenario.replace("_", " ").title() + f"  |  {dataset}", fontsize=17, pad=15)
    ax.set_xlabel("Severity", fontsize=13)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=13)
    ax.set_xticks(SEVERITIES)
    ax.set_xlim(0.8, 5.2)
    if metric in ("rmse", "mae", "sam_deg", "ssim"):
        ax.set_ylim(bottom=0)
    ax.grid(alpha=0.26)
    if drawn:
        ax.legend(ncol=2, loc="upper left", fontsize=10, framealpha=0.94)
    else:
        ax.text(0.5, 0.5, "No valid reported values", ha="center", va="center",
                transform=ax.transAxes, fontsize=15, color="gray")
    note = "Open circles: incomplete voxel coverage. Missing metric values are not treated as zero."
    if metric == "ssim":
        note += " Only methods reporting SSIM are shown."
    if metric == "r2":
        note += " R² is averaged per mask, not globally pooled."
    ax.text(0.01, -0.17, note, transform=ax.transAxes, fontsize=9.3, alpha=0.75)
    fig.tight_layout()
    fig.savefig(output, dpi=210, bbox_inches="tight")
    plt.close(fig)
    return drawn > 0


def plot_overview(summary, dataset, metric, methods, output):
    data = summary[summary["dataset"] == dataset]
    field = METRIC_FIELDS[metric]
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), constrained_layout=True)
    for ax, scenario in zip(axes.ravel(), SCENARIOS):
        subset = data[data["scenario"] == scenario]
        any_curve = False
        for method in methods:
            m = subset[subset["method"] == method].set_index("severity")
            if m.empty:
                continue
            ys = [m.at[k, field] if k in m.index else np.nan for k in SEVERITIES]
            if not np.isfinite(ys).any():
                continue
            ax.plot(SEVERITIES, ys, marker="o", ms=3.5,
                    lw=2.3 if method == "TPSSR" else 1.5,
                    color=METHOD_COLORS[method], label=method)
            any_curve = True
            incomplete = [k for k in SEVERITIES if k in m.index and
                          np.isfinite(m.at[k, field]) and m.at[k, "coverage"] < 0.9999]
            if incomplete:
                ax.scatter(incomplete, [m.at[k, field] for k in incomplete],
                           s=42, facecolors="white", edgecolors=METHOD_COLORS[method],
                           lw=1.4, zorder=8)
        if not any_curve:
            ax.text(0.5, 0.5, "Not available", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
        ax.set_title(scenario.replace("_", " ").title(), fontsize=11)
        ax.set_xticks(SEVERITIES)
        if metric in ("rmse", "mae", "sam_deg", "ssim"):
            ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=9)
    fig.supxlabel("Severity", fontsize=13)
    fig.supylabel(METRIC_LABELS[metric], fontsize=13)
    fig.suptitle(f"Method comparison across 10 scenarios  |  {dataset}", fontsize=19)
    available_methods = [
        m for m in methods
        if np.isfinite(pd.to_numeric(data.loc[data["method"] == m, field],
                                     errors="coerce")).any()
    ]
    handles = [plt.Line2D([], [], color=METHOD_COLORS[m], marker="o",
                          lw=3 if m == "TPSSR" else 2, label=m) for m in available_methods]
    if handles:
        fig.legend(handles=handles, loc="outside lower center",
                   ncol=min(len(handles), 7), fontsize=11)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    exp_dir = Path(args.experiment_dir).expanduser() if args.experiment_dir else discover_experiment()
    zip_path = Path(args.baseline_zip).expanduser() if args.baseline_zip else None
    if exp_dir is None and zip_path is None:
        zip_path = discover_zip()
    if exp_dir is None and zip_path is None:
        raise FileNotFoundError(
            "Cannot locate baseline results. Supply --experiment-dir or --baseline-zip."
        )
    tpssr_path = choose_tpssr_csv(exp_dir, args.tpssr_csv)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else (
        (exp_dir if exp_dir is not None else tpssr_path.parent) / "method_comparison_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Baselines: {exp_dir if zip_path is None else zip_path}")
    print(f"[INFO] TPSSR: {tpssr_path}")
    print(f"[INFO] Output: {output_dir}")

    baseline = standardize(baseline_frames(exp_dir, zip_path, not args.exclude_pfgin), is_tpssr=False)
    tpssr = standardize(pd.read_csv(tpssr_path), is_tpssr=True)
    check_mask_keys(baseline, tpssr)

    datasets = set(tpssr["dataset"].unique())
    baseline = baseline[baseline["dataset"].isin(datasets)].copy()
    if baseline.empty:
        raise RuntimeError(f"No baseline rows match TPSSR datasets: {sorted(datasets)}")
    all_metrics = pd.concat([baseline, tpssr], ignore_index=True, sort=False)
    methods = [label for key, label in METHOD_FOLDERS.items()
               if (not args.exclude_pfgin or key != "pfgin") and label in set(baseline["method"])]
    methods += ["TPSSR"]
    summary = build_summary(all_metrics)
    summary.to_csv(output_dir / "scenario_severity_pooled_metrics.csv", index=False)
    for metric in args.metrics:
        valid_by_method = all_metrics.groupby("method")[metric].apply(
            lambda values: int(np.isfinite(pd.to_numeric(values, errors="coerce")).sum())
        )
        missing_methods = [method for method in methods if valid_by_method.get(method, 0) == 0]
        if missing_methods:
            print(f"[INFO] No {metric} data for: {', '.join(missing_methods)}. "
                  "Their curves will be absent for that metric.")

    for dataset in [*sorted(datasets), "ALL"]:
        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(exist_ok=True)
        for metric in args.metrics:
            metric_dir = dataset_dir / metric.upper()
            metric_dir.mkdir(exist_ok=True)
            for scenario in SCENARIOS:
                plot_single(summary, dataset, scenario, metric, methods,
                            metric_dir / f"{safe_file_name(scenario)}.png")
            plot_overview(summary, dataset, metric, methods,
                          dataset_dir / f"all_scenarios_{metric}.png")
    print(f"[DONE] Saved summary CSV and figures to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)
