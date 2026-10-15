#!/usr/bin/env python3
"""Run PFGIN and DDS2M using the same arbitrary train/test experiment.

PFGIN is trained once on the configured training acquisitions and then tested on
all held-out acquisitions. DDS2M has no global training stage and is evaluated
independently on each held-out acquisition.
"""
from __future__ import annotations

import argparse
import os

from _launcher_common import CORE, run
from mask_budget import planned_counts
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
p.add_argument("--max-masks", type=int, default=200, help="TOTAL benchmark masks across all held-out acquisitions; 0 = all per acquisition")
a = p.parse_args()
spec = resolve_experiment(a)


print(describe(spec))
names = ["pfgin", "dds2m"] if a.method == "all" else [a.method]
for name in names:
    train_script, test_script = METHODS[name]
    print(f"\n{'='*84}\nEXTERNAL METHOD: {name}\n{'='*84}")
    if a.stage in ("train", "both"):
        # Mask budgeting is a testing-only concern; do not leak it into model training.
        os.environ.pop("HSI_TOTAL_TEST_MASKS", None)
        os.environ.pop("HSI_EXPERIMENT_MASK_OFFSET", None)
        os.environ.pop("HSI_EXPERIMENT_MASK_TOTAL", None)
        if train_script is None:
            print(
                "DDS2M has no global training stage; it is optimized/self-supervised "
                "per corrupted case."
            )
        else:
            run(train_script, spec, active_test=spec.primary_test)

    if a.stage in ("test", "both"):
        counts = planned_counts(a.max_masks, len(spec.tests))
        offset = 0
        for i, test_name in enumerate(spec.tests, start=1):
            print(
                f"\n--- {name}: test acquisition {i}/{len(spec.tests)} = {test_name} ---"
            )
            if int(a.max_masks) > 0:
                os.environ["HSI_TOTAL_TEST_MASKS"] = str(int(a.max_masks))
                os.environ["HSI_EXPERIMENT_MASK_OFFSET"] = str(int(offset))
                os.environ["HSI_EXPERIMENT_MASK_TOTAL"] = str(int(a.max_masks))
                print(
                    f"Global benchmark budget: {a.max_masks} | "
                    f"planned share for {test_name}: {counts[i-1]}"
                )
            else:
                os.environ.pop("HSI_TOTAL_TEST_MASKS", None)
                os.environ.pop("HSI_EXPERIMENT_MASK_OFFSET", None)
                os.environ.pop("HSI_EXPERIMENT_MASK_TOTAL", None)
            run(test_script, spec, active_test=test_name)
            if counts[i - 1] is not None:
                offset += int(counts[i - 1])
