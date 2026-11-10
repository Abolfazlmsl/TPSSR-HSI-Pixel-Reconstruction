#!/usr/bin/env python3
"""Command-line entry point for TPSSR workflows."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "prepare": "prepare_dataset.py",
    "preflight": "preflight.py",
    "train": "train.py",
    "train-region": "train_region_lineaware.py",
    "test": "test.py",
    "evaluate": "evaluate.py",
    "benchmarks": "benchmarks.py",
    "external": "external_methods.py",
    "roi": "tools/visualize_roi_benchmark_comparison.py",
    "real-lines": "tools/reconstruct_real_swir_lines.py",
    "cube": "tools/render_hsi_cube_front.py",
    "boxplots": "tools/plot_tpssr_scenario_boxplots.py",
    "aggregate": "tools/tpssr_results_aggregator.py",
}

FINAL_ROUTER = "learned_router3_lineaware_rankcv"
RESULTS_FOLDER = (
    "tpssr_learned_router3_lineaware_rankcv"
    "__minorityhybridgap10pp_two10pct__robustdonor__lazycachev1"
)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(
        description="TPSSR: reconstruction of missing hyperspectral pixels",
        usage="%(prog)s COMMAND [OPTIONS]",
    )
    parser.add_argument("command", choices=COMMANDS)

    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0

    command = argv[0]
    if command not in COMMANDS:
        parser.error(f"unknown command: {command}")
    options = argv[1:]
    cmd = [sys.executable, str(ROOT / COMMANDS[command])]
    env = os.environ.copy()

    if command == "test":
        cmd.extend([
            "--router", FINAL_ROUTER,
            "--gpus", "2",
            "--max-masks", "200",
            "--cache-mode", "lazy",
            "--minority-policy", "hybrid_gap",
            "--global-minority-percent", "10",
            "--three-expert-min-gap-percent", "10",
        ])
        env["TPSSR_FUSION_DONOR_MATCH"] = "robust"
    elif command == "evaluate":
        cmd.extend(["--router", FINAL_ROUTER, "--results-folder", RESULTS_FOLDER, "--visuals", "0"])
    elif command == "preflight" and "--data-only" not in options:
        cmd.extend(["--router", FINAL_ROUTER])
    elif command in ("train", "train-region", "real-lines"):
        cmd.extend(["--gpus", "2"])

    cmd.extend(options)
    print("RUN:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
