#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the FINAL TPSSR end-to-end benchmark.

The progress output explicitly reports the GLOBAL benchmark mask number, e.g.
  [GPU 0] Mask 37/200 (18.5%) | spatio_spectral_block S4 R1
so a two-GPU run is easy to follow.
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

import numpy as np

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"
SUPPORTED = [*(f"SWIR{i}" for i in range(1, 6)), *(f"VNIR{i}" for i in range(1, 6))]
EVAL_PROTOCOL = "historical_sampled_spatial_v2_hierarchical_end_to_end"


def parse_args():
    p = argparse.ArgumentParser(description="Test FINAL TPSSR end-to-end")
    p.add_argument("--dataset", required=True, help="SWIR1..SWIR5 or VNIR1..VNIR5")
    p.add_argument("--gpus", type=int, default=2, choices=[1, 2])
    p.add_argument("--max-masks", type=int, default=200, help="0 = all benchmark masks")
    p.add_argument("--deep-batch", type=int, default=16)
    p.add_argument("--region-batch", type=int, default=32)
    p.add_argument("--no-verify-region-batch", action="store_true")
    p.add_argument("--force", action="store_true", help="Recompute completed masks")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def launch_if_needed(args):
    if args.worker or args.gpus == 1 or "RANK" in os.environ:
        return False
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.gpus}",
        str(Path(__file__).resolve()),
        "--dataset",
        args.dataset,
        "--gpus",
        "1",
        "--max-masks",
        str(args.max_masks),
        "--deep-batch",
        str(args.deep_batch),
        "--region-batch",
        str(args.region_batch),
        "--worker",
    ]
    if args.no_verify_region_batch:
        cmd.append("--no-verify-region-batch")
    if args.force:
        cmd.append("--force")
    print("Launching Final TPSSR test on two GPUs:")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return True


ARGS = parse_args()
DATASET = str(ARGS.dataset).upper()
if DATASET not in SUPPORTED:
    raise SystemExit(f"Dataset must be one of: {', '.join(SUPPORTED)}")
if launch_if_needed(ARGS):
    raise SystemExit(0)

os.environ["TPSSR_PROJECT_DIR"] = str(ROOT)
os.environ["HSI_TEST_DATASET"] = DATASET
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

OUTDIR = Path(config.RESULTS_DIR) / "final_tpssr"
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
    )
    cube, _, split, _ = load_dataset()
    _, _, test_cube = split_blocks(cube, split)

    assigned = [
        (i, p)
        for i, p in enumerate(mask_files, start=1)
        if (i - 1) % WORLD_SIZE == RANK
    ]
    shard = OUTDIR / f"test_metrics.rank{RANK}.csv"

    completed = {}
    if shard.exists() and not ARGS.force:
        for row in read_csv(shard):
            if row.get("status") == "ok" and float(row.get("coverage", 0.0)) >= 0.999999:
                completed[str(row["mask_file"])] = row

    rows = list(completed.values())
    pending = [(i, p) for i, p in assigned if p.name not in completed]
    bar = tqdm(
        pending,
        total=len(assigned),
        initial=len(completed),
        desc=f"GPU{RANK} masks",
        unit="mask",
        dynamic_ncols=True,
        position=RANK,
        leave=True,
    )

    total_global = len(mask_files)
    for mask_index, mask_path in bar:
        t0 = time.time()
        try:
            mask, meta = load_test_mask(mask_path)
            pct = 100.0 * float(mask_index) / max(total_global, 1)
            print(
                f"[GPU {RANK}] Mask {mask_index}/{total_global} ({pct:5.1f}%) | "
                f"{meta['scenario']} S{meta['severity']} R{meta['repeat']}",
                flush=True,
            )
            mask = np.asarray(mask, dtype=bool)
            coords = np.asarray(
                select_eval_coords(mask, int(config.SEED) + int(mask_index)),
                dtype=np.int64,
            )
            if len(coords) == 0:
                raise RuntimeError("No evaluation coordinates selected.")

            out = predict_mask(
                pipeline,
                test_cube,
                mask,
                coords,
                preview_progress=f"GPU{RANK} Mask {mask_index}/{total_global} previews",
                region_progress=f"GPU{RANK} Mask {mask_index}/{total_global} Region",
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

            truth = np.asarray(test_cube)[coords[:, 0], coords[:, 1], coords[:, 2]]
            stage1 = out["stage1_route"]
            route = out["final_route"]
            row = {
                "_case_index": int(mask_index),
                "dataset": DATASET,
                "eval_protocol": EVAL_PROTOCOL,
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
                "region_component_count": int(
                    sum(r["final"] == 2 for r in out["component_records"])
                ),
                "spectral_fallback_components": int(out["spectral_fallback_components"]),
                "spectral_route_rmse": subset_rmse(truth, pred, route == 0),
                "deep_route_rmse": subset_rmse(truth, pred, route == 1),
                "region_route_rmse": subset_rmse(truth, pred, route == 2),
                "runtime_sec": float(time.time() - t0),
                "gpu": int(LOCAL_RANK) if DEVICE.type == "cuda" else -1,
                "status": "ok",
            }
        except Exception as exc:
            row = {
                "_case_index": int(mask_index),
                "dataset": DATASET,
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
        if row["status"] == "ok":
            bar.set_postfix(
                global_mask=f"{mask_index}/{total_global}",
                rmse=f"{float(row['rmse']):.4g}",
                R=f"{100.0*float(row['final_region_fraction']):.1f}%",
            )
        else:
            bar.set_postfix(global_mask=f"{mask_index}/{total_global}", status="ERROR")

    worker_meta = {
        "rank": RANK,
        "device": str(DEVICE),
        "deep_checkpoint": str(pipeline.deep_path),
        "hierarchical_router_checkpoint": str(pipeline.router_path),
        "gate_threshold": float(pipeline.gate_threshold),
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
                    merged.append(row)
                    seen.add(str(row.get("mask_file")))
        if expected.issubset(seen):
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for {len(expected-seen)} mask results.")
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
    ok = sum(r.get("status") == "ok" for r in rows)
    print(f"\nTest complete: {ok}/{len(mask_files)} masks successful")
    print("Saved:", path)
    print(f"Next: python evaluate.py --dataset {DATASET} --visuals 3")


def main():
    mask_files = list_mask_files()
    if ARGS.max_masks and ARGS.max_masks > 0:
        mask_files = mask_files[: int(ARGS.max_masks)]
    if not mask_files:
        raise RuntimeError("No benchmark masks found. Run prepare_dataset.py first.")
    if RANK == 0:
        print("=" * 94)
        print("FINAL TPSSR TEST")
        print("Stage 1: Spectral vs Deep-v3.1")
        print("Stage 2: hard-case gate -> Frozen V27 Stage-A -> V13 TopologyFirst V1")
        print("Masks:", len(mask_files), "| GPUs:", WORLD_SIZE)
        print("=" * 94)
    evaluate_worker(mask_files)
    if RANK == 0:
        merge(mask_files)


if __name__ == "__main__":
    main()
