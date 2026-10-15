from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
for path in (ROOT, CORE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("TPSSR_PROJECT_DIR", str(ROOT))
os.environ.setdefault("HSI_TRAIN_DATASETS", "SWIR1,SWIR2,SWIR3,SWIR4,SWIR6,SWIR7,SWIR8,SWIR9")
os.environ.setdefault("HSI_TEST_DATASETS", "SWIR5,SWIR10")
os.environ.setdefault("HSI_TEST_DATASET", "SWIR5")
os.environ.setdefault("HSI_ACTIVE_TEST_DATASET", "SWIR5")
