#!/usr/bin/env python3
"""Calibrate acquisitions and create fixed 200-mask benchmarks for all held-out tests."""
from __future__ import annotations

import argparse

from _launcher_common import CORE, run
from experiment_config import add_experiment_args, describe, resolve_experiment

p = argparse.ArgumentParser(
    description="Prepare TPSSR data for a folder-based or predefined train/test split."
)
add_experiment_args(p)
a = p.parse_args()
spec = resolve_experiment(a)

print(describe(spec))
for i, test_name in enumerate(spec.tests, start=1):
    print(f"\n{'='*84}\nPREPARE TEST DATASET {i}/{len(spec.tests)}: {test_name}\n{'='*84}")
    run(
        CORE / "data_preparation" / "prepare_dataset.py",
        spec,
        active_test=test_name,
    )

print("\nPreparation complete for all configured held-out acquisitions.")
