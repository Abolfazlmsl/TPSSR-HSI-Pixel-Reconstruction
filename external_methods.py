#!/usr/bin/env python3
"""Run PFGIN and DDS2M using the same arbitrary train/test experiment.

PFGIN is trained once on the configured training acquisitions and then tested on
all held-out acquisitions. DDS2M has no global training stage and is evaluated
independently on each held-out acquisition.
"""
from __future__ import annotations

import argparse

from _launcher_common import CORE, run
from experiment_config import add_experiment_args, describe, resolve_experiment

METHODS = {
    "pfgin": (
        CORE / "reconstruction_methods" / "pfgin" / "train.py",
        CORE / "reconstruction_methods" / "pfgin" / "test.py",
    ),
    "dds2m": (
        None,
        CORE / "reconstruction_methods" / "dds2m" / "test.py",
    ),
}

p = argparse.ArgumentParser(description="Run external TPSSR comparison methods.")
add_experiment_args(p)
p.add_argument("--method", required=True, choices=["pfgin", "dds2m", "all"])
p.add_argument("--stage", choices=["train", "test", "both"], default="both")
a = p.parse_args()
spec = resolve_experiment(a)

print(describe(spec))
names = ["pfgin", "dds2m"] if a.method == "all" else [a.method]
for name in names:
    train_script, test_script = METHODS[name]
    print(f"\n{'='*84}\nEXTERNAL METHOD: {name}\n{'='*84}")
    if a.stage in ("train", "both"):
        if train_script is None:
            print(
                "DDS2M has no global training stage; it is optimized/self-supervised "
                "per corrupted case."
            )
        else:
            run(train_script, spec, active_test=spec.primary_test)

    if a.stage in ("test", "both"):
        for i, test_name in enumerate(spec.tests, start=1):
            print(
                f"\n--- {name}: test acquisition {i}/{len(spec.tests)} = {test_name} ---"
            )
            run(test_script, spec, active_test=test_name)
