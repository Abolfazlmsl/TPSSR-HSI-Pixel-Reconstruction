#!/usr/bin/env python3
"""Isolated two-GPU line-aware Region training for any configured split.

The accepted TPSSR checkpoints are never modified. Geometry is trained from
scratch, followed by TopologyFirst with the new frozen Geometry checkpoint.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from _launcher_common import ROOT, CORE, env_for
from experiment_config import resolve_experiment, add_experiment_args
from train import geometry_training_metadata


METHODS = CORE / "reconstruction_methods" / "tpssr"
TRIAL = "region_lineaware_trial_v1"


def paths(spec):
    base = ROOT / "trained_models" / spec.experiment_name / "tpssr" / TRIAL
    return base, base / f"geometry_{spec.model_tag}" / "best_model.pt", base / f"topology_{spec.model_tag}" / "best_model.pt"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_experiment_args(p)
    p.add_argument("--stage", choices=["all", "geometry", "topology"], default="all")
    p.add_argument("--gpus", type=int, choices=[2], default=2)
    p.add_argument("--geometry-train-samples", type=int, default=None)
    p.add_argument("--geometry-val-samples", type=int, default=None)
    p.add_argument("--topology-train-samples", type=int, default=None)
    p.add_argument("--topology-val-samples", type=int, default=None)
    p.add_argument("--hard-val-samples", type=int, default=200)
    p.add_argument("--hard-val-weight", type=float, default=.25)
    p.add_argument("--fine-line-probability", type=float, default=.35)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if any(x is not None and x < 1 for x in [a.geometry_train_samples, a.geometry_val_samples,
                           a.topology_train_samples, a.topology_val_samples]) or a.hard_val_samples < 1:
        p.error("All sample counts must be positive")
    if not (0 <= a.fine_line_probability <= 1) or not (0 <= a.hard_val_weight <= 1):
        p.error("Probability and validation weight must be between zero and one")
    # The fixed preset requires no environment overrides from earlier custom runs.
    old = {k: os.environ.pop(k, None) for k in (
        "HSI_TRAIN_DATASETS", "HSI_TEST_DATASETS", "HSI_ACTIVE_TEST_DATASET",
        "HSI_EXPERIMENT_NAME", "HSI_TEST_DATASET")}
    try:
        spec = resolve_experiment(a)
    finally:
        for k,v in old.items():
            if v is not None: os.environ[k] = v
    # Preserve per-acquisition exposure from the validated eight-source training.
    # Four source acquisitions were the historical 2200/4200-sample reference.
    count = len(spec.train)
    if a.geometry_train_samples is None:
        a.geometry_train_samples = int(round(2200 * count / 4))
    if a.topology_train_samples is None:
        a.topology_train_samples = int(round(4200 * count / 4))
    if a.geometry_val_samples is None:
        a.geometry_val_samples = max(550, 100 * count)
    if a.topology_val_samples is None:
        a.topology_val_samples = max(900, 150 * count)
    if any(x < 1 for x in (a.geometry_train_samples, a.geometry_val_samples,
                            a.topology_train_samples, a.topology_val_samples)):
        raise RuntimeError("Invalid sample counts")
    base, geometry_ck, topology_ck = paths(spec)
    geom_dir = geometry_ck.parent
    topo_dir = topology_ck.parent
    if geom_dir.resolve() == (base.parent / f"geometry_{spec.model_tag}").resolve():
        raise RuntimeError("Trial geometry path equals production path")
    env = env_for(spec, spec.primary_test)
    env.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "TPSSR_GEOM_TRAIN_SAMPLES": str(a.geometry_train_samples),
        "TPSSR_GEOM_VAL_SAMPLES": str(a.geometry_val_samples),
        "TPSSR_FINE_LINE_PROBABILITY": str(a.fine_line_probability),
        "TPSSR_GEOM_HARD_VAL_SAMPLES": str(a.hard_val_samples),
        "TPSSR_GEOM_HARD_VAL_WEIGHT": str(a.hard_val_weight),
        "TPSSR_REGION_GEOMETRY_OUTPUT_DIR": str(geom_dir.resolve()),
        "TPSSR_REGION_TOPOLOGY_OUTPUT_DIR": str(topo_dir.resolve()),
        "TPSSR_REGION_GEOMETRY_CHECKPOINT": str(geometry_ck.resolve()),
        "TPSSR_TOPOLOGY_TRAIN_SAMPLES": str(a.topology_train_samples),
        "TPSSR_TOPOLOGY_VAL_SAMPLES": str(a.topology_val_samples),
        "TPSSR_TOPOLOGY_HARD_VAL_SAMPLES": str(a.hard_val_samples),
        "TPSSR_TOPOLOGY_HARD_VAL_WEIGHT": str(a.hard_val_weight),
        "TPSSR_TOPOLOGY_NO_RESUME": "0",
    })
    for name in ("train_geometry.py", "train_topology.py"):
        if not (METHODS / name).is_file():
            raise FileNotFoundError(METHODS / name)
    print("Line-aware Region training", spec.experiment_name, ", ".join(spec.train), flush=True)
    print("HELD OUT", ", ".join(spec.tests), flush=True)
    print("Trial checkpoints", base, flush=True)
    print("Geometry", "EXISTS" if geometry_ck.is_file() else "MISSING", geometry_ck, flush=True)
    print("Topology", "EXISTS" if topology_ck.is_file() else "MISSING", topology_ck, flush=True)
    print(f"Fine line probability={a.fine_line_probability}, hard validation weight={a.hard_val_weight}",flush=True)
    if a.dry_run:
        print("DRY RUN: no files created and no training started", flush=True)
        return

    if a.stage in ("all", "geometry"):
        if geometry_ck.exists():
            if a.stage == "geometry":
                raise RuntimeError("Line-aware Geometry already exists. Never overwrite it. Use --stage topology.")
            print("SKIP: trial geometry already trained", flush=True)
        else:
            if geom_dir.exists() and any(geom_dir.iterdir()):
                raise RuntimeError(f"Partial Geometry run found at {geom_dir}. Move/archive it before retraining; no automatic overwrite.")
            stats = geometry_training_metadata(spec, base.parent)
            env["TPSSR_GEOMETRY_TRAIN_STATS"] = str(stats.resolve())
            print("Training trial Geometry from scratch...", flush=True)
            subprocess.run([sys.executable, str(METHODS / "train_geometry.py"), "--gpus", "2"],
                           cwd=ROOT, env=env, check=True)
            if not geometry_ck.exists():
                raise RuntimeError(f"Trial Geometry failed to save: {geometry_ck}")

    if a.stage in ("all", "topology"):
        if not geometry_ck.exists():
            raise RuntimeError(f"Train experimental Geometry first: {geometry_ck}")
        if topology_ck.exists():
            if a.stage == 'all':
                print('SKIP: line-aware Topology already trained', flush=True)
                print('Existing checkpoint:', topology_ck, flush=True)
                return
            raise RuntimeError(f"Line-aware Topology exists; refusing overwrite: {topology_ck}")
        if topo_dir.exists() and any(topo_dir.iterdir()):
            raise RuntimeError(f"Partial Topology run exists: {topo_dir}. Archive it before fresh training.")
        print("Training line-aware Topology with its frozen Geometry...", flush=True)
        subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                        "--nproc_per_node=2", str(METHODS / "train_topology.py")],
                       cwd=ROOT, env=env, check=True)
        if not topology_ck.exists():
            raise RuntimeError(f"Trial Topology failed to save: {topology_ck}")
    print("COMPLETE. Original TPSSR and Router checkpoints have not been replaced.",flush=True)
    print("Trial root:", base, flush=True)


if __name__ == "__main__":
    main()
