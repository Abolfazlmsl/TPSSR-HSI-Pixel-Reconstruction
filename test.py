#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run FINAL TPSSR on every configured held-out acquisition.

The --max-masks value is a GLOBAL experiment budget, not a per-acquisition
budget.  For example, 200 total masks means 100+100 for two held-out cores or
40 each for five held-out cores.

Progress reports both experiment-wide and local acquisition progress.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

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

EVAL_PROTOCOL = "historical_sampled_spatial_v2_hierarchical_end_to_end"


def parse_args():
    p = argparse.ArgumentParser(description="Test FINAL TPSSR end-to-end")
    add_experiment_args(p)
    p.add_argument("--gpus", type=int, default=2, choices=[1, 2])
    p.add_argument("--max-masks", type=int, default=200, help="TOTAL masks across all held-out acquisitions; 0 = all masks on every test acquisition")
    p.add_argument("--deep-batch", type=int, default=16)
    p.add_argument("--region-batch", type=int, default=32)
    p.add_argument("--no-verify-region-batch", action="store_true")
    p.add_argument("--force", action="store_true", help="Recompute completed masks")
    p.add_argument("--routing-mode", choices=["hybrid_v2_region_2expert_v22", "hybrid_v2_region_2expert_v23"],
                   default="hybrid_v2_region_2expert_v23", help="Keep baseline intact or test the spectral guard")
    p.add_argument("--spectral-guard-mae", type=float, default=0.08,
                   help="Maximum mean pseudo-gap error, in normalized units")
    p.add_argument("--spectral-guard-q80", type=float, default=0.14,
                   help="Maximum 80th-percentile pseudo-gap error, normalized")
    p.add_argument("--spectral-guard-confidence", type=float, default=0.50)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def _planned_counts(total_budget, n_tests):
    if int(total_budget) <= 0:
        return [None] * int(n_tests)
    base, remainder = divmod(int(total_budget), int(n_tests))
    return [base + (1 if i < remainder else 0) for i in range(int(n_tests))]


def launch_all_tests(args, spec):
    print(describe(spec))
    counts = _planned_counts(args.max_masks, len(spec.tests))
    offset = 0
    for i, test_name in enumerate(spec.tests, start=1):
        local_planned = counts[i - 1]
        print(
            f"\n{'='*94}\nFINAL TPSSR TEST DATASET {i}/{len(spec.tests)}: {test_name}\n{'='*94}",
            flush=True,
        )
        if local_planned is not None:
            print(
                f"Global experiment budget: {args.max_masks} masks | "
                f"planned share for {test_name}: {local_planned}",
                flush=True,
            )
        common = [
            str(Path(__file__).resolve()),
            *experiment_cli_args(spec, active_test=test_name),
            "--gpus",
            "1",
            "--max-masks",
            str(args.max_masks),
            "--deep-batch",
            str(args.deep_batch),
            "--region-batch",
            str(args.region_batch),
            "--routing-mode", args.routing_mode,
            "--spectral-guard-mae", str(args.spectral_guard_mae),
            "--spectral-guard-q80", str(args.spectral_guard_q80),
            "--spectral-guard-confidence", str(args.spectral_guard_confidence),
            "--worker",
        ]
        if args.no_verify_region_batch:
            common.append("--no-verify-region-batch")
        if args.force:
            common.append("--force")

        if args.gpus == 2:
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                *common,
            ]
        else:
            cmd = [sys.executable, *common]

        child_env = env_for(spec, test_name)
        if int(args.max_masks) > 0:
            child_env["HSI_TOTAL_TEST_MASKS"] = str(int(args.max_masks))
            child_env["HSI_EXPERIMENT_MASK_OFFSET"] = str(int(offset))
            child_env["HSI_EXPERIMENT_MASK_TOTAL"] = str(int(args.max_masks))

        print("RUN:", " ".join(map(str, cmd)), flush=True)
        subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=child_env,
            check=True,
        )
        if local_planned is not None:
            offset += int(local_planned)
    print("\nFinal TPSSR testing complete for:", ", ".join(spec.tests))


ARGS = parse_args()
SPEC = resolve_experiment(ARGS)
if not ARGS.worker:
    launch_all_tests(ARGS, SPEC)
    raise SystemExit(0)

DATASET = SPEC.active_test
# Worker environment is already populated by the parent, but these assignments
# keep direct worker use deterministic as well.
os.environ.update(env_for(SPEC, DATASET))
if int(ARGS.max_masks) > 0:
    os.environ["HSI_TOTAL_TEST_MASKS"] = str(int(ARGS.max_masks))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import torch
from tqdm import tqdm
import config
from src.evaluation_utils import (
    load_dataset,
    split_blocks,
    list_mask_files,
    load_test_mask,
    select_eval_coords,
    benchmark_case_index,
    experiment_progress,
)
from src.final_pipeline import load_pipeline, predict_mask, sparse_band_predictions, subset_rmse
from src.final_tpssr_runtime import evaluate_reconstruction

RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", str(RANK)))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)
    DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
else:
    DEVICE = torch.device("cpu")

OUTDIR = Path(config.RESULTS_DIR) / {"baseline": "final_tpssr", "spectral_safe": "final_tpssr_spectral_safe_stage1_v2", "joint_v3": "final_tpssr_joint_v3", "stage1_rank_v4": "final_tpssr_stage1_rank_v4", "spectral_region": "final_tpssr_spectral_region", "hard60_v5": "final_tpssr_hard60_v5", "hard60_v6": "final_tpssr_hard60_v6", "hard60_v6_hybrid_v2_fixed": "final_tpssr_hard60_v6_hybrid_v2_fixed", "hybrid_v2_region_2expert": "final_tpssr_hybrid_v2_region_2expert", "hybrid_v2_region_2expert_v21": "final_tpssr_hybrid_v2_region_2expert_v21", "hybrid_v2_region_2expert_v22": "final_tpssr_hybrid_v2_region_2expert_v22", "hybrid_v2_region_2expert_v23": "final_tpssr_hybrid_v2_region_2expert_v23"}[ARGS.routing_mode]
OUTDIR.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    keys, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    tmp = Path(path).with_suffix(Path(path).suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def evaluate_worker(mask_files):
    pipeline = load_pipeline(
        DEVICE,
        deep_batch=ARGS.deep_batch,
        region_batch=ARGS.region_batch,
        verify_region_batch=not ARGS.no_verify_region_batch,
        routing_mode=ARGS.routing_mode,
        spectral_guard_mae=ARGS.spectral_guard_mae,
        spectral_guard_q80=ARGS.spectral_guard_q80,
        spectral_guard_confidence=ARGS.spectral_guard_confidence,
    )
    cube, _, split, _ = load_dataset(DATASET)
    _, _, test_cube = split_blocks(cube, split)

    assigned = [
        (i, p)
        for i, p in enumerate(mask_files, start=1)
        if (i - 1) % WORLD_SIZE == RANK
    ]
    shard = OUTDIR / f"test_metrics.rank{RANK}.csv"
    component_shard = OUTDIR / f"component_decisions.rank{RANK}.csv"

    assigned_names = {p.name for _i, p in assigned}
    completed = {}
    if shard.exists() and not ARGS.force:
        for row in read_csv(shard):
            name = str(row.get("mask_file", ""))
            if (
                name in assigned_names
                and row.get("status") == "ok"
                and float(row.get("coverage", 0.0)) >= 0.999999
            ):
                completed[name] = row

    rows = list(completed.values())
    component_rows = [r for r in read_csv(component_shard) if r.get("mask_file") in completed]
    pending = [(i, p) for i, p in assigned if p.name not in completed]
    bar = tqdm(
        pending,
        total=len(assigned),
        initial=len(completed),
        desc=f"{DATASET} GPU{RANK} masks",
        unit="mask",
        dynamic_ncols=True,
        position=RANK,
        leave=True,
    )

    consecutive_errors = 0
    total_global = len(mask_files)
    for mask_index, mask_path in bar:
        t0 = time.time()
        global_index, global_total = experiment_progress(mask_index, total_global)
        stable_case_index = benchmark_case_index(mask_path)
        new_component_rows = []
        try:
            mask, meta = load_test_mask(mask_path)
            pct = 100.0 * float(global_index) / max(global_total, 1)
            print(
                f"[{DATASET}][GPU {RANK}] Experiment Mask "
                f"{global_index}/{global_total} ({pct:5.1f}%) | "
                f"local {mask_index}/{total_global} | {meta['scenario']} "
                f"S{meta['severity']} R{meta['repeat']}",
                flush=True,
            )
            mask = np.asarray(mask, dtype=bool)
            coords = np.asarray(
                select_eval_coords(mask, int(config.SEED) + int(stable_case_index)),
                dtype=np.int64,
            )
            if len(coords) == 0:
                raise RuntimeError("No evaluation coordinates selected.")

            out = predict_mask(
                pipeline,
                test_cube,
                mask,
                coords,
                preview_progress=(
                    f"{DATASET} GPU{RANK} Mask {mask_index}/{total_global} previews"
                ),
                region_progress=(
                    f"{DATASET} GPU{RANK} Mask {mask_index}/{total_global} Region"
                ),
            )
            pred = out["prediction"]
            reference_v6_metrics = None
            if ARGS.routing_mode == "hard60_v6_hybrid_v2_fixed":
                ref = out["v6_fixed_reference_prediction"]
                if ref is None or not np.isfinite(ref).all():
                    raise RuntimeError('Internal fixed-route V6 reference incomplete.')
                reference_v6_metrics = evaluate_reconstruction(
                    test_cube, sparse_band_predictions(coords, ref), mask, eval_coords=coords
                )
            band_predictions = sparse_band_predictions(coords, pred)
            metrics = evaluate_reconstruction(
                test_cube, band_predictions, mask, eval_coords=coords
            )
            if float(metrics.get("coverage", 0.0)) < 0.999999:
                raise RuntimeError(
                    f"Incomplete final coverage: {metrics.get('filled_voxels', 0)}/"
                    f"{metrics.get('target_voxels', 0)}"
                )

            if ARGS.routing_mode in ("hybrid_v2_region_2expert", "hybrid_v2_region_2expert_v21", "hybrid_v2_region_2expert_v22", "hybrid_v2_region_2expert_v23") and np.any(out["final_route"] == 1):
                raise RuntimeError("Deep Expert unexpectedly selected in two-Expert mode.")

            truth = np.asarray(test_cube)[coords[:, 0], coords[:, 1], coords[:, 2]]
            stage1 = out["stage1_route"]
            route = out["final_route"]
            new_component_rows = [
                {"dataset": DATASET, "mask_file": mask_path.name,
                 "scenario": str(meta["scenario"]), "severity": int(meta["severity"]),
                 "repeat": int(meta["repeat"]), **decision}
                for decision in out["component_records"]
            ]
            component_count = max(len(out["component_records"]), 1)
            safe_override_count = sum(int(r["spectral_safe_override"]) for r in out["component_records"])
            row = {
                "_case_index": int(stable_case_index),
                "experiment_case_index": int(global_index),
                "dataset_local_index": int(mask_index),
                "dataset": DATASET,
                "experiment": config.EXPERIMENT_NAME,
                "eval_protocol": EVAL_PROTOCOL,
                "mask_file": mask_path.name,
                "scenario": str(meta["scenario"]),
                "severity": int(meta["severity"]),
                "repeat": int(meta["repeat"]),
                **metrics,
                **({"reference_v6_rmse": float(reference_v6_metrics['rmse']),
                    "reference_v6_mae": float(reference_v6_metrics['mae']),
                    "reference_v6_r2": float(reference_v6_metrics['r2']),
                    "reference_v6_sam_deg": float(reference_v6_metrics['sam_deg']),
                    "reference_v6_ssim": float(reference_v6_metrics['ssim']),
                    "delta_rmse_vs_fixed_v6": float(metrics['rmse'])-float(reference_v6_metrics['rmse']),
                    "hybrid_changed_voxels": int(out['hybrid_changed_voxels']),
                    "hybrid_fallback_voxels": int(out['hybrid_fallback_voxels'])}
                   if reference_v6_metrics is not None else {}),
                "stage1_spectral_fraction": float(np.mean(stage1 == 0)),
                "stage1_deep_fraction": float(np.mean(stage1 == 1)),
                "final_spectral_fraction": float(np.mean(route == 0)),
                "final_deep_fraction": float(np.mean(route == 1)),
                "final_region_fraction": float(np.mean(route == 2)),
                "mean_gate_probability": float(np.nanmean(out["gate_probability"])),
                "component_count": int(len(out["component_records"])),
                "routing_mode": ARGS.routing_mode,
                "router_policy": {"baseline": "baseline", "spectral_safe": "spectral_safe_stage1_v2", "joint_v3": "joint_bandwise_v3", "stage1_rank_v4": "legacy_rank_stage1_v4", "spectral_region": "availability_only_two_expert_ablation", "hard60_v5": "legacy_v4_manual_hard60_region_gate", "hard60_v6": "legacy_v4_hard60_mask_geometry_exceptions", "hard60_v6_hybrid_v2_fixed": "hard60_v6_frozen_router_hybrid_v2_band_outputs", "hybrid_v2_region_2expert": "pure_two_expert_observed_band_availability", "hybrid_v2_region_2expert_v21": "v21_long_gap_neighbor_support_frozen_policy", "hybrid_v2_region_2expert_v22": "v22_true_mask_gap_purity_exploratory_policy", "hybrid_v2_region_2expert_v23": "v23_longgap_region_plus_local_median"}[ARGS.routing_mode],
                "stage1_spectral_override_components": int(sum(r["stage1_spectral_override"] for r in out["component_records"])),
                "spectral_safe_override_components": int(safe_override_count),
                "spectral_safe_override_component_fraction": float(safe_override_count / component_count),
                "baseline_region_component_count": int(sum(r["baseline_final"] == 2 for r in out["component_records"])),
                "spectral_guard_pass_components": int(sum(r["guard_protected"] for r in out["component_records"])),
                "region_component_count": int(
                    sum(r["final"] == 2 for r in out["component_records"])
                ),
                "spectral_fallback_components": int(out["spectral_fallback_components"]),
                "spectral_route_rmse": subset_rmse(truth, pred, route == 0),
                "hybrid_route_rmse": subset_rmse(truth, pred, route == 0) if ARGS.routing_mode in ("hybrid_v2_region_2expert", "hybrid_v2_region_2expert_v21", "hybrid_v2_region_2expert_v22", "hybrid_v2_region_2expert_v23") else float("nan"),
                "final_hybrid_fraction": float(np.mean(route == 0)) if ARGS.routing_mode in ("hybrid_v2_region_2expert", "hybrid_v2_region_2expert_v21", "hybrid_v2_region_2expert_v22", "hybrid_v2_region_2expert_v23") else float("nan"),
                "deep_route_rmse": subset_rmse(truth, pred, route == 1),
                "region_route_rmse": subset_rmse(truth, pred, route == 2),
                "v21_region_long_gap_eval_voxels": int(out.get("v21_region_gap_eval_voxels", 0)),
                "v21_region_long_gap_pixels": int(out.get("v21_policy_audit", {}).get("region_long_gap_pixels", 0)),
                "v21_region_no_anchor_pixels": int(out.get("v21_policy_audit", {}).get("region_no_anchor_pixels", 0)),
                "v21_gap_candidates": int(out.get("v21_policy_audit", {}).get("long_gap_candidate_pixels", 0)),
                "v22_longgap_region_eval_voxels": int(out.get("v22_region_gap_eval_voxels", 0)),
                "v22_region_long_gap_pixels": int(out.get("v22_policy_audit", {}).get("region_long_gap_pixels", 0)),
                "v22_region_returned_to_hybrid_pixels": int(out.get("v22_policy_audit", {}).get("v22_long_gap_returned_to_hybrid", 0)),
                "v22_rejected_fragmented_pixels": int(out.get("v22_policy_audit", {}).get("v22_long_gap_rejected_incoherent", 0)),
                "v22_rejected_non_dominant_pixels": int(out.get("v22_policy_audit", {}).get("v22_long_gap_rejected_non_dominant", 0)),

                "v23_additional_region_eval_voxels": int(out.get("v23_policy_audit", {}).get("additional_region_voxels", 0)),
                "v23_local_median_eval_voxels": int(out.get("v23_policy_audit", {}).get("local_median_voxels", 0)),
                "runtime_sec": float(time.time() - t0),
                "gpu": int(LOCAL_RANK) if DEVICE.type == "cuda" else -1,
                "status": "ok",
            }
        except Exception as exc:
            print(traceback.format_exc(), flush=True)
            row = {
                "_case_index": int(stable_case_index),
                "experiment_case_index": int(global_index),
                "dataset_local_index": int(mask_index),
                "dataset": DATASET,
                "experiment": config.EXPERIMENT_NAME,
                "eval_protocol": EVAL_PROTOCOL,
                "mask_file": mask_path.name,
                "scenario": "",
                "severity": "",
                "repeat": "",
                "coverage": 0.0,
                "mae": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "sam_deg": np.nan,
                "ssim": np.nan,

                "runtime_sec": float(time.time() - t0),
                "gpu": int(LOCAL_RANK) if DEVICE.type == "cuda" else -1,
                "status": f"ERROR: {type(exc).__name__}: {exc}",
            }

        rows = [r for r in rows if str(r.get("mask_file")) != mask_path.name]
        rows.append(row)
        write_csv(shard, rows)
        component_rows = [r for r in component_rows if r.get("mask_file") != mask_path.name]
        component_rows.extend(new_component_rows)
        write_csv(component_shard, component_rows)
        if row["status"] == "ok":
            consecutive_errors = 0
            bar.set_postfix(
                global_mask=f"{experiment_progress(mask_index, total_global)[0]}/{experiment_progress(mask_index, total_global)[1]}",
                rmse=f"{float(row['rmse']):.4g}",
                R=f"{100.0*float(row['final_region_fraction']):.1f}%",
            )
        else:
            consecutive_errors += 1
            gi, gt = experiment_progress(mask_index, total_global)
            bar.set_postfix(global_mask=f"{gi}/{gt}", status="ERROR")
            if consecutive_errors >= 3:
                raise RuntimeError("Stopped after 3 consecutive mask errors; see tracebacks above.")

    worker_meta = {
        "dataset": DATASET,
        "experiment": config.EXPERIMENT_NAME,
        "train_datasets": list(config.TRAIN_DATASET_FOLDERS),
        "test_datasets": list(config.TEST_DATASET_FOLDERS),
        "rank": RANK,
        "device": str(DEVICE),
        "deep_checkpoint": str(pipeline.deep_path),
        "hierarchical_router_checkpoint": str(pipeline.router_path),
        "gate_threshold": float(pipeline.gate_threshold),
        "routing_mode": str(ARGS.routing_mode),
        **({"v21_policy": __import__("src.spectral_route_policy_v21", fromlist=["policy_spec"]).policy_spec()} if ARGS.routing_mode == "hybrid_v2_region_2expert_v21" else {}),
        **({"v22_policy": __import__("src.spectral_route_policy_v22", fromlist=["policy_spec"]).policy_spec()} if ARGS.routing_mode == "hybrid_v2_region_2expert_v22" else {}),
        **({"v23_policy": __import__("src.spectral_route_policy_v23", fromlist=["policy_spec"]).policy_spec()} if ARGS.routing_mode == "hybrid_v2_region_2expert_v23" else {}),
        "spectral_guard_mae_limit": float(ARGS.spectral_guard_mae),
        "spectral_guard_q80_limit": float(ARGS.spectral_guard_q80),
        "spectral_guard_confidence_limit": float(ARGS.spectral_guard_confidence),
        "topology_checkpoint": str(pipeline.topology_path),
        "v27_stage_a_checkpoint": str(pipeline.v27_path),
    }
    (OUTDIR / f"worker.rank{RANK}.json").write_text(
        json.dumps(worker_meta, indent=2), encoding="utf-8"
    )


def merge(mask_files):
    expected = {p.name for p in mask_files}
    deadline = time.monotonic() + 24 * 3600
    while True:
        merged, seen = [], set()
        for rank in range(WORLD_SIZE):
            p = OUTDIR / f"test_metrics.rank{rank}.csv"
            if p.exists():
                for row in read_csv(p):
                    name = str(row.get("mask_file", ""))
                    if name not in expected:
                        continue
                    merged.append(row)
                    seen.add(name)
        if expected.issubset(seen):
            break
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for {len(expected-seen)} mask results."
            )
        time.sleep(3)

    order = {p.name: i for i, p in enumerate(mask_files)}
    merged.sort(key=lambda r: order.get(str(r.get("mask_file")), 10**9))
    rows = []
    for row in merged:
        row = dict(row)
        row.pop("_case_index", None)
        rows.append(row)
    path = OUTDIR / "test_metrics.csv"
    write_csv(path, rows)
    all_components = []
    for rank in range(WORLD_SIZE):
        for record in read_csv(OUTDIR / f"component_decisions.rank{rank}.csv"):
            if str(record.get("mask_file", "")) in expected:
                all_components.append(record)
    if all_components:
        all_components.sort(key=lambda r: (order.get(r.get("mask_file", ""), 10**9), int(r.get("component", 0))))
        write_csv(OUTDIR / "component_decisions.csv", all_components)
    ok = sum(r.get("status") == "ok" for r in rows)
    print(f"\n{DATASET} test complete: {ok}/{len(mask_files)} masks successful")
    print("Saved:", path)


def main_worker():
    # list_mask_files() already applies --max-masks as a GLOBAL experiment
    # budget through HSI_TOTAL_TEST_MASKS.  Do not slice again per acquisition.
    mask_files = list_mask_files()
    if not mask_files:
        raise RuntimeError("No benchmark masks found. Run prepare_dataset.py first.")
    if RANK == 0:
        print("=" * 94)
        print("FINAL TPSSR TEST")
        print("Experiment:", config.EXPERIMENT_NAME)
        print("Active held-out dataset:", DATASET)
        print("Training datasets:", ", ".join(config.TRAIN_DATASET_FOLDERS))
        if ARGS.routing_mode in ("hybrid_v2_region_2expert", "hybrid_v2_region_2expert_v21", "hybrid_v2_region_2expert_v22", "hybrid_v2_region_2expert_v23"):
            print("TWO EXPERTS ONLY: Hybrid V2 BAND / Frozen Region (mask-only sufficiency policy)")
            print("No Deep model or V6 Router is loaded or run.")
        else:
            print("Stage 1: V6 frozen Router / Deep-v3.1 (unchanged)")
        if ARGS.routing_mode == "hybrid_v2_region_2expert_v21":
            print("V2.1 FROZEN MASK GEOMETRY: >=2 observed + long-gap neighbor-support guard")
        if ARGS.routing_mode == "hybrid_v2_region_2expert_v22":
            print("V2.2 EXPLORATORY: V2.1 + true-target spectral gap purity guard")
        if ARGS.routing_mode == "hard60_v6_hybrid_v2_fixed":
            print("After routing: replace Spectral-selected predictions with Hybrid V2 BAND")
        if ARGS.routing_mode == "hybrid_v2_region_2expert_v23":
            print("V2.3 FINAL: frozen V2.2 + extra Region on supported long gaps + local median")
        print("Routing mode:", ARGS.routing_mode)
        print("Region: Frozen V27 Stage-A -> V13 TopologyFirst V1")
        print("Allocated masks for this acquisition:", len(mask_files), "| GPUs:", WORLD_SIZE)
        if int(ARGS.max_masks) > 0:
            print("GLOBAL experiment mask budget:", int(ARGS.max_masks))
        print("=" * 94)
    evaluate_worker(mask_files)
    if RANK == 0:
        merge(mask_files)


if __name__ == "__main__":
    main_worker()
