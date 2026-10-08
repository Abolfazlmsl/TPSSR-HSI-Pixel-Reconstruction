#!/usr/bin/env python3
"""Run the five classical/reference benchmarks used with Final TPSSR."""
import argparse
from _launcher_common import CORE, run

METHODS = {
    "spectral": (None, CORE / "reconstruction_methods" / "baselines" / "run_spectral.py"),
    "spatial": (None, CORE / "reconstruction_methods" / "baselines" / "run_spatial.py"),
    "spatio_spectral": (None, CORE / "reconstruction_methods" / "baselines" / "run_spatio_spectral.py"),
    "pca": (CORE / "reconstruction_methods" / "pca" / "train.py", CORE / "reconstruction_methods" / "pca" / "test.py"),
    "random_forest": (CORE / "reconstruction_methods" / "random_forest" / "train.py", CORE / "reconstruction_methods" / "random_forest" / "test.py"),
}

p = argparse.ArgumentParser(description="Run one of the five Final TPSSR benchmark methods.")
p.add_argument("--dataset", required=True, help="SWIR1..SWIR5 or VNIR1..VNIR5")
p.add_argument("--method", required=True, choices=[*sorted(METHODS), "all"])
p.add_argument("--stage", choices=["train", "test", "both"], default="both")
a = p.parse_args()

names = sorted(METHODS) if a.method == "all" else [a.method]
for name in names:
    train_script, test_script = METHODS[name]
    print(f"\n{'='*80}\nBENCHMARK: {name}\n{'='*80}")
    if a.stage in ("train", "both"):
        if train_script is None:
            print("No separate training stage; analytical method.")
        else:
            run(train_script, a.dataset)
    if a.stage in ("test", "both"):
        run(test_script, a.dataset)
