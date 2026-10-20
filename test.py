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
from mask_budget import planned_counts
from experiment_config import (
    add_experiment_args,
    describe,
    experiment_cli_args,
    resolve_experiment,
)

EVAL_PROTOCOL = "sampled_masked_voxel_evaluation"


def parse_args():
    p = argparse.ArgumentParser(description="Test FINAL TPSSR end-to-end")
    add_experiment_args(p)
    p.add_argument("--gpus", type=int, default=2, choices=[1, 2])
    p.add_argument("--max-masks", type=int, default=200, help="TOTAL masks across all held-out acquisitions; 0 = all masks on every test acquisition")
    p.add_argument("--region-batch", type=int, default=32)
    p.add_argument("--no-verify-region-batch", action="store_true")
    p.add_argument("--force", action="store_true", help="Recompute completed masks")
    p.add_argument("--router", choices=["v23", "neighbor_v1", "neighbor_v2", "learned_fusion", "learned_fusion_guarded", "learned_router", "learned_router3", "learned_router3_rmse", "learned_router3_aligned", "learned_router3_adaptive"], default="v23",
                   help="Select original V2.3, fixed Neighbor V1/V2, or learned spectral fusion")
    p.add_argument("--focus-band-guard", action="store_true",
                   help="Evaluate only the 12 mask cases where V2 band guard was activated")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()




def launch_all_tests(args, spec):
    if args.focus_band_guard and args.router != 'learned_fusion_guarded':
        raise ValueError('--focus-band-guard requires --router learned_fusion_guarded')
    print(describe(spec))
    counts = planned_counts(args.max_masks, len(spec.tests))
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
            "--region-batch",
            str(args.region_batch),
            "--worker",
        ]
        common.extend(["--router", args.router])
        if args.no_verify_region_batch:
            common.append("--no-verify-region-batch")
        if args.force:
            common.append("--force")
        if args.focus_band_guard:
            common.append("--focus-band-guard")

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
    _mask_case_key, load_dataset,
    split_blocks,
    list_mask_files,
    load_test_mask,
    select_eval_coords,
    benchmark_case_index,
    experiment_progress,
)
from src.pipeline import load_pipeline, predict_mask, predict_mask_candidate, predict_mask_candidate_v2, predict_mask_learned_fusion, predict_mask_learned_fusion_guarded, predict_mask_learned_router, predict_mask_learned_router3, sparse_band_predictions, subset_rmse
from src.spatial_inference import evaluate_reconstruction

RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", str(RANK)))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)
    DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
else:
    DEVICE = torch.device("cpu")

OUTDIR = Path(config.RESULTS_DIR) / ({'v23': 'tpssr', 'neighbor_v1': 'tpssr_neighbor_v1', 'neighbor_v2': 'tpssr_neighbor_v2', 'learned_fusion': 'tpssr_learned_fusion', 'learned_fusion_guarded': 'tpssr_learned_fusion_guarded', 'learned_router': 'tpssr_learned_router', 'learned_router3': 'tpssr_learned_router3', 'learned_router3_rmse': 'tpssr_learned_router3_rmse', 'learned_router3_aligned': 'tpssr_learned_router3_aligned', 'learned_router3_adaptive': 'tpssr_learned_router3_adaptive'}[ARGS.router])
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
        region_batch=ARGS.region_batch,
        verify_region_batch=not ARGS.no_verify_region_batch,
        learned_fusion=(ARGS.router in ("learned_fusion", "learned_fusion_guarded", "learned_router", "learned_router3", "learned_router3_rmse", "learned_router3_aligned", "learned_router3_adaptive")),
        learned_router=(ARGS.router == "learned_router"),
        learned_router3=(ARGS.router == "learned_router3"),
        learned_router3_rmse=(ARGS.router == "learned_router3_rmse"),
        learned_router3_aligned=(ARGS.router == "learned_router3_aligned"),
        learned_router3_adaptive=(ARGS.router == "learned_router3_adaptive"),
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

            predictor = {"v23": predict_mask, "neighbor_v1": predict_mask_candidate,
                         "neighbor_v2": predict_mask_candidate_v2,
                         "learned_fusion": predict_mask_learned_fusion,
                         "learned_fusion_guarded": predict_mask_learned_fusion_guarded,
                         "learned_router": predict_mask_learned_router,
                         "learned_router3": predict_mask_learned_router3, "learned_router3_rmse": predict_mask_learned_router3, "learned_router3_aligned": predict_mask_learned_router3, "learned_router3_adaptive": predict_mask_learned_router3}[ARGS.router]
            out = predictor(
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
            band_predictions = sparse_band_predictions(coords, pred)
            metrics = evaluate_reconstruction(
                test_cube, band_predictions, mask, eval_coords=coords
            )
            if float(metrics.get("coverage", 0.0)) < 0.999999:
                raise RuntimeError(
                    f"Incomplete final coverage: {metrics.get('filled_voxels', 0)}/"
                    f"{metrics.get('target_voxels', 0)}"
                )

            if np.any(out["final_route"] == 1):
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
                "router_variant": ARGS.router,
                **({f"candidate_{k}_voxels": v for k, v in out.get("candidate_method_voxels", {}).items()} if ARGS.router != "v23" else {}),
                **({f"candidate_{k}": v for k, v in out.get("candidate_policy_audit", {}).items()} if ARGS.router != "v23" else {}),
                "mask_file": mask_path.name,
                "scenario": str(meta["scenario"]),
                "severity": int(meta["severity"]),
                "repeat": int(meta["repeat"]),
                **metrics,
                "stage1_spectral_fraction": float(np.mean(stage1 == 0)),
                "stage1_deep_fraction": float(np.mean(stage1 == 1)),
                "final_spectral_fraction": float(np.mean(route == 0)),
                "final_deep_fraction": float(np.mean(route == 1)),
                "final_region_fraction": float(np.mean(route == 2)),
                "mean_gate_probability": float(np.nanmean(out["gate_probability"])),
                "component_count": int(len(out["component_records"])),
                "routing_mode": ARGS.router,
                "router_policy": "learned_three_expert_quality" if ARGS.router in ("learned_router3", "learned_router3_rmse", "learned_router3_aligned", "learned_router3_adaptive") else "mask_geometry_and_local_consistency",
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
                "hybrid_route_rmse": subset_rmse(truth, pred, route == 0),
                "final_hybrid_fraction": float(np.mean(route == 0)),
                "deep_route_rmse": subset_rmse(truth, pred, route == 1),
                "region_route_rmse": subset_rmse(truth, pred, route == 2),
                "support_region_long_gap_eval_voxels": int(out.get("support_region_gap_eval_voxels", 0)),
                "support_region_long_gap_pixels": int(out.get("support_policy_audit", {}).get("region_long_gap_pixels", 0)),
                "support_region_no_anchor_pixels": int(out.get("support_policy_audit", {}).get("region_no_anchor_pixels", 0)),
                "support_gap_candidates": int(out.get("support_policy_audit", {}).get("long_gap_candidate_pixels", 0)),
                "guard_longgap_region_eval_voxels": int(out.get("guard_region_gap_eval_voxels", 0)),
                "guard_region_long_gap_pixels": int(out.get("guard_policy_audit", {}).get("region_long_gap_pixels", 0)),
                "guard_region_returned_to_hybrid_pixels": int(out.get("guard_policy_audit", {}).get("guard_long_gap_returned_to_hybrid", 0)),
                "guard_rejected_fragmented_pixels": int(out.get("guard_policy_audit", {}).get("guard_long_gap_rejected_incoherent", 0)),
                "guard_rejected_non_dominant_pixels": int(out.get("guard_policy_audit", {}).get("guard_long_gap_rejected_non_dominant", 0)),

                "policy_additional_region_eval_voxels": int(out.get("routing_audit", {}).get("additional_region_voxels", 0)),
                "policy_local_median_eval_voxels": int(out.get("routing_audit", {}).get("local_median_voxels", 0)),
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
        "routing_mode": ARGS.router,
        "routing_parameters": __import__("src.routing_policy", fromlist=["policy_spec"]).policy_spec(),
        "spectral_backbone_checkpoint": str(pipeline.hybrid_backbone_checkpoint_path),
        "spectral_gate_checkpoint": str(pipeline.hybrid_gate_checkpoint_path),
        "topology_checkpoint": str(pipeline.topology_path),
        "geometry_stage_a_checkpoint": str(pipeline.geometry_path),
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
    if ARGS.focus_band_guard:
        selected = {('band_wide_saturation', 4), ('band_wide_saturation', 5),
                    ('mixed_saturation', 5)}
        mask_files = [p for p in mask_files if _mask_case_key(p)[:2] in selected]
    if not mask_files:
        raise RuntimeError("No benchmark masks found. Run prepare_dataset.py first.")
    if RANK == 0:
        print("=" * 94)
        print("FINAL TPSSR TEST")
        print("Experiment:", config.EXPERIMENT_NAME)
        print("Active held-out dataset:", DATASET)
        print("Training datasets:", ", ".join(config.TRAIN_DATASET_FOLDERS))
        print("TPSSR: spectral backbone, geometry guidance, topology completion")
        print("Routing: observed-only spectral-gap policy with local consistency checks")
        print("Allocated masks for this acquisition:", len(mask_files), "| GPUs:", WORLD_SIZE)
        if int(ARGS.max_masks) > 0:
            print("GLOBAL experiment mask budget:", int(ARGS.max_masks))
        print("=" * 94)
    evaluate_worker(mask_files)
    if RANK == 0:
        merge(mask_files)


if __name__ == "__main__":
    main_worker()
