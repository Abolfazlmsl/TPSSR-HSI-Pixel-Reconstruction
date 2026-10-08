#!/usr/bin/env python3
"""Prepare one LOIO fold and its fixed 200-mask benchmark."""
import argparse
from _launcher_common import CORE, run

p = argparse.ArgumentParser(description="Prepare one complete TPSSR benchmark fold.")
p.add_argument("--dataset", required=True, help="SWIR1..SWIR5 or VNIR1..VNIR5")
a = p.parse_args()
run(CORE / "data_preparation" / "prepare_dataset.py", a.dataset)
