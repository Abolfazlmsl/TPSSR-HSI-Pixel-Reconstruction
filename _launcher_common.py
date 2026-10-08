from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

from experiment_config import ExperimentSpec, SUPPORTED_DATASETS

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"
SUPPORTED = SUPPORTED_DATASETS


def env_for(spec: ExperimentSpec, active_test: str | None = None):
    active = str(active_test or spec.active_test).upper()
    if active not in spec.tests:
        raise SystemExit(
            f"Active test {active} is not in experiment tests: {', '.join(spec.tests)}"
        )
    env = os.environ.copy()
    env["TPSSR_PROJECT_DIR"] = str(ROOT)
    env["HSI_TRAIN_DATASETS"] = ",".join(spec.train)
    env["HSI_TEST_DATASETS"] = ",".join(spec.tests)
    # Legacy training code uses HSI_TEST_DATASET only as the model/fold tag.
    # Keep it fixed to the first held-out acquisition for the whole experiment.
    env["HSI_TEST_DATASET"] = spec.primary_test
    # Evaluation/preparation may move across several held-out acquisitions.
    env["HSI_ACTIVE_TEST_DATASET"] = active
    env["HSI_EXPERIMENT_NAME"] = spec.experiment_name
    return env


def run(script, spec: ExperimentSpec, extra=None, *, active_test: str | None = None):
    cmd = [sys.executable, str(script)] + (list(extra) if extra else [])
    print("\nRUN:", " ".join(map(str, cmd)), flush=True)
    subprocess.run(
        cmd,
        cwd=str(CORE),
        env=env_for(spec, active_test),
        check=True,
    )
