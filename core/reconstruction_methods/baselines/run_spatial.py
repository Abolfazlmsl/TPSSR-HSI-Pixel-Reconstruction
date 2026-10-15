from pathlib import Path
import sys

HERE = Path(__file__).resolve()
PROJECT_ROOT = next(
    parent for parent in HERE.parents
    if (parent / "config.py").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm import tqdm

import config
from src.evaluation_utils import (
    load_dataset,
    split_blocks,
    list_mask_files,
    load_test_mask,
    select_eval_coords,
    regression_metrics,
    spectral_angle_metrics,
    sparse_ssim_metrics,
    save_method_metrics,
    benchmark_case_index,
    experiment_progress,
)
from src.local_methods import reconstruct


METHOD = "spatial"


def main():
    cube, _, split, _ = load_dataset()
    _, _, test = split_blocks(cube, split)

    rows = []
    mask_files = list_mask_files()
    total_masks = len(mask_files)

    for mask_index, mask_file in enumerate(
        tqdm(mask_files, desc=f"{METHOD} masks"),
        start=1,
    ):
        mask, meta = load_test_mask(mask_file)
        global_index, global_total = experiment_progress(mask_index, total_masks)
        print(
            f"[{METHOD}] Experiment Mask {global_index}/{global_total} "
            f"({100.0*global_index/max(global_total,1):5.1f}%) | "
            f"local {mask_index}/{total_masks} | {meta['scenario']} "
            f"S{meta['severity']} R{meta['repeat']}",
            flush=True,
        )

        stable_case_index = benchmark_case_index(mask_file)
        coords = select_eval_coords(
            mask,
            config.SEED + stable_case_index,
        )

        if len(coords) == 0:
            continue

        pred = reconstruct(
            test,
            mask,
            tqdm(
                coords,
                desc=f"{METHOD}: {mask_file.stem}",
                leave=False,
            ),
            METHOD,
        )

        truth = np.asarray(test)[
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
        ]

        reg = regression_metrics(truth, pred)
        sam = spectral_angle_metrics(test, coords, pred)
        ssim = sparse_ssim_metrics(test, mask, coords, pred)

        rows.append({
            "scenario": meta["scenario"],
            "severity": meta["severity"],
            "repeat": meta["repeat"],
            "method": METHOD,
            **reg,
            **sam,
            **ssim,
        })

    output = save_method_metrics(METHOD, rows)
    print("")
    print("Completed.")
    print(output)


if __name__ == "__main__":
    main()
