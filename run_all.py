#!/usr/bin/env python3
"""Convenience runner for arbitrary TPSSR 10-acquisition experiments."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from experiment_config import (
    add_experiment_args,
    describe,
    experiment_cli_args,
    resolve_experiment,
)

ROOT = Path(__file__).resolve().parent


def run(script, *extra):
    cmd = [sys.executable, str(script), *map(str, extra)]
    print("\nRUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


p = argparse.ArgumentParser(description="Run Final TPSSR and comparison workflows.")
add_experiment_args(p)
p.add_argument(
    "--everything",
    action="store_true",
    help="Prepare, Final TPSSR, five benchmarks, and both external methods.",
)
p.add_argument("--prepare", action="store_true")
p.add_argument("--final", action="store_true", help="Train, test, evaluate Final TPSSR.")
p.add_argument("--benchmarks", action="store_true", help="Run all five benchmark methods.")
p.add_argument("--external", action="store_true", help="Run PFGIN and DDS2M (very expensive).")
p.add_argument("--gpus", type=int, default=2, choices=[2])
p.add_argument("--max-masks", type=int, default=200)
p.add_argument("--visuals", type=int, default=3)
a = p.parse_args()

if not any((a.everything, a.prepare, a.final, a.benchmarks, a.external)):
    p.error("Choose at least one action or use --everything.")

spec = resolve_experiment(a)
exp_args = experiment_cli_args(spec)
print(describe(spec))

if a.everything or a.prepare:
    run(ROOT / "prepare_dataset.py", *exp_args)

if a.everything or a.final:
    run(ROOT / "train.py", *exp_args, "--gpus", str(a.gpus))
    run(
        ROOT / "test.py",
        *exp_args,
        "--gpus",
        str(a.gpus),
        "--max-masks",
        str(a.max_masks),
    )
    run(
        ROOT / "evaluate.py",
        *exp_args,
        "--visuals",
        str(a.visuals),
    )

if a.everything or a.benchmarks:
    run(
        ROOT / "benchmarks.py",
        *exp_args,
        "--method",
        "all",
        "--stage",
        "both",
        "--max-masks",
        str(a.max_masks),
    )

if a.everything or a.external:
    run(
        ROOT / "external_methods.py",
        *exp_args,
        "--method",
        "all",
        "--stage",
        "both",
        "--max-masks",
        str(a.max_masks),
    )
