#!/usr/bin/env python3
"""Run the five classical/reference benchmarks on arbitrary train/test splits.

PCA and Random Forest are trained once on the configured training acquisitions.
All five methods are then tested independently on every held-out acquisition.
"""
from __future__ import annotations

import argparse

from _launcher_common import CORE, run
from experiment_config import add_experiment_args, describe, resolve_experiment

METHODS = {
    "spectral": (
        None,
        CORE / "reconstruction_methods" / "baselines" / "run_spectral.py",
    ),
    "spatial": (
        None,
        CORE / "reconstruction_methods" / "baselines" / "run_spatial.py",
    ),
    "spatio_spectral": (
        None,
        CORE / "reconstruction_methods" / "baselines" / "run_spatio_spectral.py",
    ),
    "pca": (
        CORE / "reconstruction_methods" / "pca" / "train.py",
        CORE / "reconstruction_methods" / "pca" / "test.py",
    ),
    "random_forest": (
        CORE / "reconstruction_methods" / "random_forest" / "train.py",
        CORE / "reconstruction_methods" / "random_forest" / "test.py",
    ),
}

p = argparse.ArgumentParser(description="Run Final TPSSR benchmark methods.")
add_experiment_args(p)
p.add_argument("--method", required=True, choices=[*sorted(METHODS), "all"])
p.add_argument("--stage", choices=["train", "test", "both"], default="both")
a = p.parse_args()
spec = resolve_experiment(a)

print(describe(spec))
names = sorted(METHODS) if a.method == "all" else [a.method]
for name in names:
    train_script, test_script = METHODS[name]
    print(f"\n{'='*84}\nBENCHMARK: {name}\n{'='*84}")
    if a.stage in ("train", "both"):
        if train_script is None:
            print("No separate training stage; analytical method.")
        else:
            # Train once. primary_test is only an internal checkpoint tag.
            run(train_script, spec, active_test=spec.primary_test)

    if a.stage in ("test", "both"):
        for i, test_name in enumerate(spec.tests, start=1):
            print(
                f"\n--- {name}: test acquisition {i}/{len(spec.tests)} = {test_name} ---"
            )
            run(test_script, spec, active_test=test_name)
