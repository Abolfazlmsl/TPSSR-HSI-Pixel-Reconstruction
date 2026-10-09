from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve()
PROJECT_ROOT = next(parent for parent in HERE.parents if (parent / "config.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
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
    save_method_metrics,
    benchmark_case_index,
    experiment_progress,
)
from reconstruction_methods.pfgin.common import (
    require_official_repo,
    import_official_models,
    make_official_config,
    unwrap,
    load_scaler,
    normalize_cube_patch,
    tile_origin,
    spectral_origin,
    extract_spatial,
)


def load_model(device):
    repo = require_official_repo(config)
    models = import_official_models(repo)
    model_dir = config.MODELS_DIR / "pfgin"
    checkpoint = model_dir / "best.pt"
    scaler = model_dir / "scaler.npz"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"PFGIN checkpoint not found: {checkpoint}\n"
            "Run python reconstruction_methods/pfgin/train.py first."
        )
    if not scaler.exists():
        raise FileNotFoundError(f"PFGIN scaler not found: {scaler}")

    official_cfg = make_official_config(config, model_dir, device, training=False)
    model = models.InpaintingModel(official_cfg).to(device)
    payload = torch.load(checkpoint, map_location=device)
    unwrap(model.generator).load_state_dict(payload["generator"])
    model.generator.eval()
    return model, load_scaler(scaler), int(payload.get("iteration", 0))


def group_coords(coords, shape):
    h, w, bands = shape
    size = int(config.PFGIN_PATCH_SIZE)
    stride = int(config.PFGIN_TILE_STRIDE)
    window = int(config.PFGIN_SPECTRAL_WINDOW)
    bstride = int(config.PFGIN_SPECTRAL_STRIDE)
    groups = {}
    for idx, (y, x, b) in enumerate(coords):
        y0 = tile_origin(int(y), h, size, stride)
        x0 = tile_origin(int(x), w, size, stride)
        b0 = spectral_origin(int(b), bands, window, bstride)
        groups.setdefault((y0, x0, b0), []).append(idx)
    return groups


@torch.no_grad()
def reconstruct_mask(model, device, test, mask, coords, lo, hi):
    predictions = np.full(len(coords), np.nan, dtype=np.float32)
    groups = sorted(group_coords(coords, test.shape).items())
    batch_size = int(config.PFGIN_TEST_BATCH_SIZE)
    size = int(config.PFGIN_PATCH_SIZE)
    window = int(config.PFGIN_SPECTRAL_WINDOW)

    for start in tqdm(range(0, len(groups), batch_size), desc="PFGIN tiles", leave=False):
        current = groups[start:start + batch_size]
        image_batch = []
        mask_batch = []
        meta = []

        for (y0, x0, b0), indices in current:
            tile = extract_spatial(test[:, :, b0:b0 + window], y0, x0, size)
            tile_mask = extract_spatial(mask[:, :, b0:b0 + window], y0, x0, size)
            invalid = ~np.isfinite(tile)
            norm = normalize_cube_patch(tile, lo[b0:b0 + window], hi[b0:b0 + window])
            effective_mask = np.asarray(tile_mask, dtype=bool) | invalid
            image_batch.append(np.transpose(norm, (2, 0, 1)))
            mask_batch.append(np.transpose(effective_mask.astype(np.float32), (2, 0, 1)))
            meta.append((y0, x0, b0, indices))

        images = torch.from_numpy(np.stack(image_batch).astype(np.float32)).to(device)
        masks = torch.from_numpy(np.stack(mask_batch).astype(np.float32)).to(device)
        outputs = model(images, masks).detach().cpu().numpy()

        for bi, (y0, x0, b0, indices) in enumerate(meta):
            out = outputs[bi]
            for idx in indices:
                y, x, b = map(int, coords[idx])
                yy, xx, bb = y - y0, x - x0, b - b0
                if 0 <= yy < size and 0 <= xx < size and 0 <= bb < window:
                    norm_value = float(out[bb, yy, xx])
                    predictions[idx] = norm_value * float(hi[b] - lo[b]) + float(lo[b])

    return predictions


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("PFGIN testing is configured for CUDA. No CUDA GPU was found.")

    gpu_ids = list(config.PFGIN_GPU_IDS)
    device = torch.device(f"cuda:{gpu_ids[0]}")
    torch.cuda.set_device(gpu_ids[0])
    model, (lo, hi), trained_iter = load_model(device)
    if len(gpu_ids) > 1:
        # DataParallel is useful for batched inference. Only generator is needed.
        model.generator = torch.nn.DataParallel(model.generator, device_ids=gpu_ids)

    if bool(config.PFGIN_SMOKE_TEST):
        print("WARNING: PFGIN_SMOKE_TEST=True. The checkpoint is not suitable for paper results.")
    print(f"Loaded PFGIN checkpoint from iteration {trained_iter}.")

    cube, _, split, _ = load_dataset()
    _, _, test = split_blocks(cube, split)

    mask_files = list_mask_files()
    # In a multi-acquisition experiment, list_mask_files() already applies the
    # shared GLOBAL budget.  Keep PFGIN_MAX_MASKS only as a legacy single-run
    # debug cap when no global experiment budget is active.
    if os.environ.get("HSI_TOTAL_TEST_MASKS") is None and config.PFGIN_MAX_MASKS is not None:
        mask_files = mask_files[: int(config.PFGIN_MAX_MASKS)]

    rows = []
    for mask_index, mask_file in enumerate(tqdm(mask_files, desc="PFGIN masks"), start=1):
        mask, meta = load_test_mask(mask_file)
        global_index, global_total = experiment_progress(mask_index, len(mask_files))
        print(
            f"[PFGIN] Experiment Mask {global_index}/{global_total} "
            f"({100.0*global_index/max(global_total,1):5.1f}%) | "
            f"local {mask_index}/{len(mask_files)} | {meta['scenario']} "
            f"S{meta['severity']} R{meta['repeat']}",
            flush=True,
        )
        stable_case_index = benchmark_case_index(mask_file)
        coords = select_eval_coords(mask, config.SEED + stable_case_index)
        if len(coords) == 0:
            continue

        pred = reconstruct_mask(model, device, test, mask, coords, lo, hi)
        truth = np.asarray(test)[coords[:, 0], coords[:, 1], coords[:, 2]]
        reg = regression_metrics(truth, pred)
        sam = spectral_angle_metrics(test, coords, pred)
        rows.append({
            "scenario": meta["scenario"],
            "severity": meta["severity"],
            "repeat": meta["repeat"],
            "method": "pfgin",
            **reg,
            **sam,
        })
        save_method_metrics("pfgin", rows)

    output = save_method_metrics("pfgin", rows)
    print("\nCompleted.")
    print(output)
    if os.environ.get("HSI_TOTAL_TEST_MASKS") is None and config.PFGIN_MAX_MASKS is not None:
        print("WARNING: PFGIN_MAX_MASKS is applying an additional legacy debug cap.")


if __name__ == "__main__":
    main()
