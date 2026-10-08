#!/usr/bin/env python3
"""Run the two published external comparison methods: PFGIN and DDS2M."""
import argparse
from _launcher_common import CORE, run

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

p = argparse.ArgumentParser(description="Run an external TPSSR comparison method.")
p.add_argument("--dataset", required=True, help="SWIR1..SWIR5 or VNIR1..VNIR5")
p.add_argument("--method", required=True, choices=["pfgin", "dds2m", "all"])
p.add_argument("--stage", choices=["train", "test", "both"], default="both")
a = p.parse_args()

names = ["pfgin", "dds2m"] if a.method == "all" else [a.method]
for name in names:
    train_script, test_script = METHODS[name]
    print(f"\n{'='*80}\nEXTERNAL METHOD: {name}\n{'='*80}")
    if a.stage in ("train", "both"):
        if train_script is None:
            print("DDS2M has no global training stage; it is optimized/self-supervised per corrupted case.")
        else:
            run(train_script, a.dataset)
    if a.stage in ("test", "both"):
        run(test_script, a.dataset)
