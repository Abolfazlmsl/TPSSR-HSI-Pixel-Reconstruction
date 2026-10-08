#!/usr/bin/env python3
"""Convenience workflow runner for the clean Final TPSSR project."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(*args):
    cmd = [sys.executable, *map(str, args)]
    print("\nRUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


p = argparse.ArgumentParser(description="Run Final TPSSR and comparison workflows.")
p.add_argument("--dataset", required=True, help="SWIR1..SWIR5 or VNIR1..VNIR5")
p.add_argument("--everything", action="store_true", help="Prepare, train/test/evaluate TPSSR, five benchmarks, and both external methods.")
p.add_argument("--prepare", action="store_true")
p.add_argument("--final", action="store_true", help="Train, test, and evaluate Final TPSSR.")
p.add_argument("--benchmarks", action="store_true", help="Run all five benchmark methods.")
p.add_argument("--external", action="store_true", help="Run PFGIN and DDS2M (very expensive).")
p.add_argument("--gpus", type=int, default=2, choices=[2])
p.add_argument("--max-masks", type=int, default=200)
a = p.parse_args()

if not any((a.everything, a.prepare, a.final, a.benchmarks, a.external)):
    p.error("Choose at least one action or use --everything.")

if a.everything or a.prepare:
    run(ROOT / "prepare_dataset.py", "--dataset", a.dataset)

if a.everything or a.final:
    run(ROOT / "train.py", "--dataset", a.dataset, "--gpus", str(a.gpus))
    run(
        ROOT / "test.py",
        "--dataset",
        a.dataset,
        "--gpus",
        str(a.gpus),
        "--max-masks",
        str(a.max_masks),
    )
    run(ROOT / "evaluate.py", "--dataset", a.dataset, "--visuals", "3")

if a.everything or a.benchmarks:
    run(ROOT / "benchmarks.py", "--dataset", a.dataset, "--method", "all", "--stage", "both")

if a.everything or a.external:
    run(ROOT / "external_methods.py", "--dataset", a.dataset, "--method", "all", "--stage", "both")
