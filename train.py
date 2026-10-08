#!/usr/bin/env python3
"""Train the FINAL TPSSR architecture once for an arbitrary train/test experiment.

Training is performed ONLY on the configured training acquisitions. All configured
held-out acquisitions share this one trained TPSSR model.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from _launcher_common import ROOT, CORE, env_for
from experiment_config import add_experiment_args, describe, resolve_experiment

METHOD = CORE / "reconstruction_methods" / "texture_restoration_router_rf_v2"


def run_cmd(cmd, spec, extra_env=None):
    env = env_for(spec, spec.primary_test)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    print("\nRUN:", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(CORE), env=env, check=True)


def main():
    p = argparse.ArgumentParser(
        description="Train Final TPSSR once for a custom 10-acquisition experiment."
    )
    add_experiment_args(p)
    p.add_argument("--gpus", type=int, default=2, choices=[2])
    p.add_argument(
        "--force",
        action="store_true",
        help="Run every stage even if its best checkpoint already exists.",
    )
    p.add_argument("--topology-epochs", type=int, default=22)
    p.add_argument("--topology-min-epochs", type=int, default=14)
    p.add_argument("--topology-patience", type=int, default=5)
    p.add_argument("--topology-train-samples", type=int, default=4200)
    p.add_argument("--topology-val-samples", type=int, default=900)
    p.add_argument("--router-epochs", type=int, default=24)
    a = p.parse_args()
    spec = resolve_experiment(a)

    # Existing TPSSR training scripts use the first held-out acquisition as a
    # stable internal checkpoint tag. It is never used for training data.
    tag = spec.model_tag
    base = (
        ROOT
        / "trained_models"
        / spec.experiment_name
        / "texture_restoration_router_rf_v2"
    )

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
            [
                sys.executable,
                METHOD / "train_swir1_v27_learned_correspondence.py",
                "--gpus",
                "2",
            ],
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
            ]
            + (["--force-cache"] if a.force else []),
            {},
        ),
    ]

    print("=" * 94)
    print("FINAL TPSSR TRAINING")
    print(describe(spec))
    print(
        "Architecture: Spectral/Deep-v3.1 -> hard-case gate -> "
        "V27 Stage-A -> V13 TopologyFirst"
    )
    print("IMPORTANT: none of the configured test acquisitions are used for training.")
    print("=" * 94)

    for label, checkpoint, cmd, stage_env in stages:
        print(f"\n=== {label} ===")
        if checkpoint.exists() and not a.force:
            print("SKIP: checkpoint already exists ->", checkpoint)
            continue
        run_cmd(cmd, spec, stage_env)
        if not checkpoint.exists():
            raise RuntimeError(
                f"Stage completed but checkpoint was not found: {checkpoint}"
            )
        print("OK ->", checkpoint)

    print("\nFINAL TPSSR training complete.")
    print("Router checkpoint:", stages[-1][1])
    print("Test acquisitions:", ", ".join(spec.tests))
    print("Next: python test.py --preset/--test-datasets ... --gpus 2 --max-masks 200")


if __name__ == "__main__":
    main()
