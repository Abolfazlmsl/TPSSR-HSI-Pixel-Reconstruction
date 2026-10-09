from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve()
PROJECT_ROOT = next(parent for parent in HERE.parents if (parent / "config.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.io import loadmat, savemat
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


def require_official_repo():
    repo = config.DDS2M_REPO_DIR
    required = [
        repo / "main_completion.py",
        repo / "runners" / "diffusion.py",
        repo / "runners" / "VS2M.py",
        repo / "functions" / "denoising.py",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Official DDS2M was not found. Run this first:\n"
            "python reconstruction_methods/dds2m/setup_official.py\n\n"
            + "\n".join(str(p) for p in missing)
        )
    return repo


def settings_signature():
    payload = "|".join(
        str(v)
        for v in [
            config.DDS2M_TILE_SIZE,
            config.DDS2M_TILE_STRIDE,
            config.DDS2M_TIMESTEPS,
            config.DDS2M_START_POINT,
            config.DDS2M_RANK,
            config.DDS2M_SIGMA_0,
            config.DDS2M_ETA,
            config.DDS2M_ETA_B,
            config.DDS2M_BETA,
            config.DDS2M_INNER_ITERATIONS,
            config.DDS2M_LEARNING_RATE,
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def tile_origin(y, x, height, width):
    size = int(config.DDS2M_TILE_SIZE)
    stride = int(config.DDS2M_TILE_STRIDE)
    if size < 32:
        raise ValueError("DDS2M_TILE_SIZE is unrealistically small.")
    if stride <= 0 or stride > size:
        raise ValueError("DDS2M_TILE_STRIDE must be in [1, DDS2M_TILE_SIZE].")

    def one(coord, length):
        if length <= size:
            return 0
        ideal = coord - size // 2
        origin = int(round(ideal / stride) * stride)
        return max(0, min(origin, length - size))

    return one(int(y), height), one(int(x), width)


def extract_square(array, y0, x0, size, fill_value=0):
    h, w = array.shape[:2]
    y1 = min(h, y0 + size)
    x1 = min(w, x0 + size)
    piece = np.asarray(array[y0:y1, x0:x1, ...]).copy()

    pad_y = size - piece.shape[0]
    pad_x = size - piece.shape[1]
    if pad_y <= 0 and pad_x <= 0:
        return piece

    pad_spec = [(0, max(0, pad_y)), (0, max(0, pad_x))]
    pad_spec.extend([(0, 0)] * (piece.ndim - 2))

    if piece.dtype == bool:
        return np.pad(piece, pad_spec, mode="constant", constant_values=False)
    if piece.shape[0] > 1 and piece.shape[1] > 1:
        return np.pad(piece, pad_spec, mode="reflect")
    return np.pad(piece, pad_spec, mode="constant", constant_values=fill_value)


def normalize_observed(tile, missing):
    tile = np.asarray(tile, dtype=np.float32)
    known = (~missing) & np.isfinite(tile)
    values = tile[known]

    if values.size == 0:
        return np.zeros_like(tile, dtype=np.float32), 0.0, 1.0

    lo = float(np.quantile(values, 0.001))
    hi = float(np.quantile(values, 0.999))
    if not np.isfinite(lo):
        lo = float(np.nanmin(values))
    if not np.isfinite(hi):
        hi = float(np.nanmax(values))
    if hi - lo < 1e-6:
        hi = lo + 1.0

    normalized = np.clip((tile - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    normalized[missing] = 0.0
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized, lo, hi


def write_dds2m_config(path, data_root, filename, image_size, channels):
    text = (
        'data:\n'
        '  dataset: "msi"\n'
        '  training_dataset: "CelebA_HQ"\n'
        f'  root: "{data_root.as_posix()}"\n'
        f'  filename: "{filename}"\n'
        '  category: ""\n'
        f'  image_size: {int(image_size)}\n'
        f'  channels: {int(channels)}\n'
        '  logit_transform: false\n'
        '  uniform_dequantization: false\n'
        '  gaussian_dequantization: false\n'
        '  random_flip: false\n'
        '  rescaled: true\n'
        '  num_workers: 1\n'
        '  out_of_dist: true\n\n'
        'model:\n'
        f'  iter_number: {int(config.DDS2M_INNER_ITERATIONS)}\n'
        '  reinit: false\n'
        f'  lr: {float(config.DDS2M_LEARNING_RATE)}\n'
        '  point: 0\n'
        '  in_channels: 3\n'
        '  out_ch: 3\n'
        '  ch: 128\n'
        '  ch_mult: [1, 1, 2, 2, 4, 4]\n'
        '  num_res_blocks: 2\n'
        '  attn_resolutions: [16]\n'
        '  dropout: 0.0\n'
        '  var_type: fixedsmall\n'
        '  ema_rate: 0.999\n'
        '  ema: true\n'
        '  resamp_with_conv: true\n\n'
        'diffusion:\n'
        '  beta_schedule: linear\n'
        '  beta_start: 0.0001\n'
        '  beta_end: 0.005\n'
        '  num_diffusion_timesteps: 3000\n\n'
        'sampling:\n'
        '  batch_size: 1\n'
        '  last_only: true\n'
    )
    path.write_text(text, encoding="utf-8")


def locate_latest_output(repo, filename, started_at):
    candidates = []
    results_root = repo / "results"
    if results_root.exists():
        for path in results_root.glob("*/latest.mat"):
            try:
                if path.stat().st_mtime >= started_at - 2 and Path(filename).stem in str(path.parent):
                    candidates.append(path)
            except FileNotFoundError:
                pass
    if not candidates:
        raise FileNotFoundError(
            "DDS2M finished without producing latest.mat. Check the per-tile log."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_official_case(repo, case_dir, case_name, observed, known_mask, gpu_id):
    data_dir = case_dir / "input"
    data_dir.mkdir(parents=True, exist_ok=True)
    mat_name = case_name + ".mat"

    savemat(
        data_dir / mat_name,
        {
            "img_clean": observed.astype(np.float32),
            "mask_10": known_mask.astype(np.float32),
            "mask_20": known_mask.astype(np.float32),
            "mask_30": known_mask.astype(np.float32),
        },
        do_compression=True,
    )

    config_name = f"hsi_project_{case_name}.yml"
    config_path = repo / "configs" / config_name
    write_dds2m_config(config_path, data_dir.resolve(), mat_name, observed.shape[0], observed.shape[2])

    command = [
        sys.executable,
        "main_completion.py",
        "--config", config_name,
        "--seed", str(int(config.SEED)),
        "--timesteps", str(int(config.DDS2M_TIMESTEPS)),
        "--start_point", str(float(config.DDS2M_START_POINT)),
        "--deg", "completion30",
        "--sigma_0", str(float(config.DDS2M_SIGMA_0)),
        "--eta", str(float(config.DDS2M_ETA)),
        "--etaB", str(float(config.DDS2M_ETA_B)),
        "--beta", str(float(config.DDS2M_BETA)),
        "--rank", str(int(config.DDS2M_RANK)),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    log_path = case_dir / "official_dds2m.log"
    started_at = time.time()

    if config.DDS2M_VERBOSE_SUBPROCESS:
        subprocess.run(command, cwd=repo, env=env, check=True)
    else:
        with log_path.open("w", encoding="utf-8") as log_handle:
            subprocess.run(
                command,
                cwd=repo,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=True,
            )

    output_path = locate_latest_output(repo, mat_name, started_at)
    output = loadmat(output_path)
    if "x_recon" not in output:
        raise KeyError(f"x_recon not found in {output_path}")
    recon = np.asarray(output["x_recon"], dtype=np.float32)

    # Do not use x_best: the upstream reproduction code chooses it by clean PSNR.
    if not config.DDS2M_KEEP_OFFICIAL_OUTPUTS:
        shutil.rmtree(output_path.parent, ignore_errors=True)
    try:
        config_path.unlink()
    except FileNotFoundError:
        pass
    return recon


def reconstruct_mask(repo, test, mask, coords, mask_tag, cache_root):
    predictions = np.full(len(coords), np.nan, dtype=np.float32)
    h, w, _bands = test.shape
    size = int(config.DDS2M_TILE_SIZE)

    groups = {}
    for index, (y, x, _band) in enumerate(coords):
        groups.setdefault(tile_origin(int(y), int(x), h, w), []).append(index)

    signature = settings_signature()
    tile_jobs = list(enumerate(sorted(groups.items()), start=1))

    gpu_ids = list(getattr(config, "DDS2M_GPU_IDS", [config.DDS2M_GPU_ID]))
    gpu_ids = [int(g) for g in gpu_ids]
    if not gpu_ids:
        raise ValueError("DDS2M_GPU_IDS must contain at least one GPU id.")

    # One sequential worker is assigned to each GPU. This guarantees that no
    # GPU receives two DDS2M subprocesses simultaneously, while independent
    # tiles can run in parallel across multiple GPUs.
    jobs_by_gpu = {gpu_id: [] for gpu_id in gpu_ids}
    for job_index, job in enumerate(tile_jobs):
        gpu_id = gpu_ids[job_index % len(gpu_ids)]
        jobs_by_gpu[gpu_id].append(job)

    progress_lock = threading.Lock()

    def process_one_tile(gpu_id, tile_number, group_item):
        (y0, x0), indices = group_item
        cache_path = cache_root / f"{mask_tag}__y{y0}_x{x0}__{signature}.npz"

        if config.DDS2M_RESUME and cache_path.exists():
            cached = np.load(cache_path)
            return (
                cached["indices"].astype(np.int64),
                cached["predictions"].astype(np.float32),
            )

        tile = extract_square(test, y0, x0, size)
        tile_mask = extract_square(mask, y0, x0, size)
        missing = tile_mask | ~np.isfinite(tile)
        known_mask = (~missing).astype(np.float32)
        observed, lo, hi = normalize_observed(tile, missing)

        case_name = f"{mask_tag}__tile{tile_number:03d}__y{y0}_x{x0}"
        case_dir = cache_root / "work" / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        recon_norm = run_official_case(
            repo, case_dir, case_name, observed, known_mask, gpu_id
        )
        if recon_norm.shape != observed.shape:
            raise RuntimeError(
                f"Unexpected DDS2M output shape {recon_norm.shape}; expected {observed.shape}."
            )
        recon = recon_norm * (hi - lo) + lo

        tile_indices = []
        tile_predictions = []
        for index in indices:
            y, x, b = map(int, coords[index])
            yy, xx = y - y0, x - x0
            if 0 <= yy < recon.shape[0] and 0 <= xx < recon.shape[1]:
                value = float(recon[yy, xx, b])
                tile_indices.append(index)
                tile_predictions.append(value)

        tile_indices = np.asarray(tile_indices, dtype=np.int64)
        tile_predictions = np.asarray(tile_predictions, dtype=np.float32)
        np.savez_compressed(
            cache_path,
            indices=tile_indices,
            predictions=tile_predictions,
        )
        if not config.DDS2M_KEEP_OFFICIAL_OUTPUTS:
            shutil.rmtree(case_dir, ignore_errors=True)

        return tile_indices, tile_predictions

    def gpu_worker(gpu_id, jobs, pbar):
        outputs = []
        for tile_number, group_item in jobs:
            result = process_one_tile(gpu_id, tile_number, group_item)
            outputs.append(result)
            with progress_lock:
                pbar.update(1)
                pbar.set_postfix_str(f"GPU {gpu_id}")
        return outputs

    print(f"DDS2M GPUs: {gpu_ids} | tiles: {len(tile_jobs)}")
    with tqdm(total=len(tile_jobs), desc=f"DDS2M tiles {mask_tag}", leave=False) as pbar:
        if len(gpu_ids) == 1:
            worker_outputs = [gpu_worker(gpu_ids[0], jobs_by_gpu[gpu_ids[0]], pbar)]
        else:
            with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
                futures = [
                    executor.submit(gpu_worker, gpu_id, jobs_by_gpu[gpu_id], pbar)
                    for gpu_id in gpu_ids
                    if jobs_by_gpu[gpu_id]
                ]
                worker_outputs = [future.result() for future in futures]

    for outputs in worker_outputs:
        for tile_indices, tile_predictions in outputs:
            predictions[tile_indices] = tile_predictions

    return predictions


def main():
    repo = require_official_repo()
    cube, _, split, _ = load_dataset()
    _, _, test = split_blocks(cube, split)

    mask_files = list_mask_files()
    # list_mask_files() enforces the shared GLOBAL experiment budget.  The old
    # DDS2M_MAX_MASKS knob is used only for legacy/debug single runs.
    if os.environ.get("HSI_TOTAL_TEST_MASKS") is None and config.DDS2M_MAX_MASKS is not None:
        mask_files = mask_files[: int(config.DDS2M_MAX_MASKS)]

    method_name = getattr(config, "DDS2M_METHOD_NAME", "dds2m")
    output_root = config.RESULTS_DIR / method_name
    cache_root = output_root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for mask_index, mask_file in enumerate(tqdm(mask_files, desc="DDS2M masks"), start=1):
        mask, meta = load_test_mask(mask_file)
        global_index, global_total = experiment_progress(mask_index, len(mask_files))
        print(
            f"[DDS2M] Experiment Mask {global_index}/{global_total} "
            f"({100.0*global_index/max(global_total,1):5.1f}%) | "
            f"local {mask_index}/{len(mask_files)} | {meta['scenario']} "
            f"S{meta['severity']} R{meta['repeat']}",
            flush=True,
        )
        stable_case_index = benchmark_case_index(mask_file)
        coords = select_eval_coords(mask, config.SEED + stable_case_index)
        if len(coords) == 0:
            continue

        mask_tag = f"{meta['scenario']}__s{meta['severity']}__r{meta['repeat']}"
        pred = reconstruct_mask(repo, test, mask, coords, mask_tag, cache_root)
        truth = np.asarray(test)[coords[:, 0], coords[:, 1], coords[:, 2]]
        reg = regression_metrics(truth, pred)
        sam = spectral_angle_metrics(test, coords, pred)

        rows.append({
            "scenario": meta["scenario"],
            "severity": meta["severity"],
            "repeat": meta["repeat"],
            "method": method_name,
            **reg,
            **sam,
        })
        save_method_metrics(method_name, rows)

    output = save_method_metrics(method_name, rows)
    print("\nCompleted.")
    print(output)
    if os.environ.get("HSI_TOTAL_TEST_MASKS") is None and config.DDS2M_MAX_MASKS is not None:
        print("WARNING: DDS2M_MAX_MASKS is applying an additional legacy debug cap.")


if __name__ == "__main__":
    main()
