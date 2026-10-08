#!/usr/bin/env python3
"""Train the FINAL TPSSR architecture for one LOIO fold.

Order:
  1) Deep-v3 teacher
  2) Deep-v3.1 expert
  3) V27 learned-correspondence model (Stage-A structure is used by Final TPSSR)
  4) V13 TopologyFirst V1 region filler
  5) Final hierarchical router:
       Stage 1 = Spectral vs Deep-v3.1
       Stage 2 = hard-case gate -> V27 Stage-A -> V13 TopologyFirst V1

The held-out acquisition is never used for training.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from _launcher_common import ROOT, CORE, env_for

METHOD = CORE / "reconstruction_methods" / "texture_restoration_router_rf_v2"


def run_cmd(cmd, dataset, extra_env=None):
    env = env_for(dataset)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    print("\nRUN:", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(CORE), env=env, check=True)


def fold_model_root(dataset):
    ds = dataset.upper()
    trainset = [f"{ds[:-1]}{i}" for i in range(1, 6) if f"{ds[:-1]}{i}" != ds]
    exp = "global_" + "_".join(trainset) + "__test_" + ds
    return ROOT / "trained_models" / exp / "texture_restoration_router_rf_v2"


def main():
    p = argparse.ArgumentParser(description="Train Final TPSSR from scratch or resume.")
    p.add_argument("--dataset", required=True, help="SWIR1..SWIR5 or VNIR1..VNIR5")
    p.add_argument("--gpus", type=int, default=2, choices=[2], help="Final training currently uses exactly two GPUs.")
    p.add_argument("--force", action="store_true", help="Run every stage even if its best checkpoint already exists.")
    p.add_argument("--topology-epochs", type=int, default=22)
    p.add_argument("--topology-min-epochs", type=int, default=14)
    p.add_argument("--topology-patience", type=int, default=5)
    p.add_argument("--topology-train-samples", type=int, default=4200)
    p.add_argument("--topology-val-samples", type=int, default=900)
    p.add_argument("--router-epochs", type=int, default=24)
    a = p.parse_args()

    ds = a.dataset.upper()
    tag = ds.lower()
    base = fold_model_root(ds)

    stages = [
        (
            "1/5 Deep-v3 teacher",
            base / f"deep_v3_{tag}_dev_2gpu" / "best_model.pt",
            [sys.executable, METHOD / "train_swir1_deep_v3.py", "--gpus", "2"],
            {},
        ),
        (
            "2/5 Deep-v3.1 expert",
            base / f"deep_v31_{tag}_dev_2gpu" / "best_model.pt",
            [sys.executable, METHOD / "train_swir1_deep_v31.py", "--gpus", "2"],
            {},
        ),
        (
            "3/5 V27 Stage-A / correspondence model",
            base / f"region_v27_original_{tag}_dev_2gpu" / "best_model.pt",
            [sys.executable, METHOD / "train_swir1_v27_learned_correspondence.py", "--gpus", "2"],
            {},
        ),
        (
            "4/5 V13 TopologyFirst V1",
            base / f"new_v13_topology_first_{tag}" / "best_model.pt",
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                METHOD / "train_new_v13_topology_first.py",
            ],
            {
                "V13TF_EPOCHS": a.topology_epochs,
                "V13TF_MIN_EPOCHS": a.topology_min_epochs,
                "V13TF_PATIENCE": a.topology_patience,
                "V13TF_TRAIN_SAMPLES": a.topology_train_samples,
                "V13TF_VAL_SAMPLES": a.topology_val_samples,
                "V13TF_NO_RESUME": 1 if a.force else 0,
            },
        ),
        (
            "5/5 Final hierarchical router",
            base / f"hierarchical_router_{tag}_dev_2gpu" / "best_model.pt",
            [
                sys.executable,
                METHOD / "train_swir1_hierarchical_router.py",
                "--gpus",
                "2",
                "--epochs",
                str(a.router_epochs),
            ] + (["--force-cache"] if a.force else []),
            {},
        ),
    ]

    print("=" * 94)
    print("FINAL TPSSR TRAINING")
    print("Held-out dataset:", ds)
    print("Architecture: Spectral/Deep-v3.1 -> hard-case gate -> V27 Stage-A -> V13 TopologyFirst")
    print("=" * 94)

    for label, checkpoint, cmd, stage_env in stages:
        print(f"\n=== {label} ===")
        if checkpoint.exists() and not a.force:
            print("SKIP: checkpoint already exists ->", checkpoint)
            continue
        run_cmd(cmd, ds, stage_env)
        if not checkpoint.exists():
            raise RuntimeError(f"Stage completed but checkpoint was not found: {checkpoint}")
        print("OK ->", checkpoint)

    print("\nFINAL TPSSR training complete.")
    print("Router checkpoint:", stages[-1][1])
    print(f"Next: python test.py --dataset {ds} --gpus 2 --max-masks 200")


if __name__ == "__main__":
    main()
