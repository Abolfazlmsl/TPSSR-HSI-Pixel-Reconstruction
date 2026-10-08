from pathlib import Path
import os, subprocess, sys
ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"
SUPPORTED = [*(f"SWIR{i}" for i in range(1,6)), *(f"VNIR{i}" for i in range(1,6))]
def env_for(dataset):
    dataset=str(dataset).upper()
    if dataset not in SUPPORTED:
        raise SystemExit(f"Dataset must be one of: {', '.join(SUPPORTED)}")
    env=os.environ.copy(); env["TPSSR_PROJECT_DIR"]=str(ROOT); env["HSI_TEST_DATASET"]=dataset
    return env
def run(script, dataset, extra=None):
    cmd=[sys.executable,str(script)]+(list(extra) if extra else [])
    print("\nRUN:"," ".join(cmd),flush=True)
    subprocess.run(cmd,cwd=str(CORE),env=env_for(dataset),check=True)
