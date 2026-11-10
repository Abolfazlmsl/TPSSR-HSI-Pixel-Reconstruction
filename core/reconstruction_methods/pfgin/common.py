from pathlib import Path
from types import SimpleNamespace
import importlib
import importlib.util
import sys

import numpy as np
import torch


def require_official_repo(config):
    repo = Path(config.PFGIN_REPO_DIR)
    required = [
        repo / "src" / "models.py",
        repo / "src" / "networks.py",
        repo / "src" / "loss.py",
        repo / "kpn" / "network.py",
        repo / "kpn" / "utils.py",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "PFGIN is not installed under external/PFGIN. "
            "Run: python tpssr.py setup-external\n\n"
            + "\n".join(str(p) for p in missing)
        )
    return repo


def import_official_models(repo):
    """Import PFGIN's src package under an alias to avoid collision with this project's src."""
    repo = Path(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    package_name = "pfgin_official_src"
    if package_name not in sys.modules:
        init_path = repo / "src" / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            package_name,
            init_path,
            submodule_search_locations=[str(repo / "src")],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)

    return importlib.import_module(f"{package_name}.models")


def make_official_config(config, model_dir, device, training=True):
    bands = int(config.PFGIN_SPECTRAL_WINDOW)
    return SimpleNamespace(
        mode=str(config.PFGIN_MODE),
        FILTER_TYPE="ss",
        kernel_size=list(config.PFGIN_KERNEL_SIZE),
        in_chs=2 * bands,
        GAN_LOSS=str(config.PFGIN_GAN_LOSS),
        LR=float(config.PFGIN_LEARNING_RATE),
        D2G_LR=float(config.PFGIN_D2G_LR),
        BETA1=float(config.PFGIN_BETA1),
        BETA2=float(config.PFGIN_BETA2),
        L1_LOSS_WEIGHT=float(config.PFGIN_L1_LOSS_WEIGHT),
        FM_LOSS_WEIGHT=float(config.PFGIN_FM_LOSS_WEIGHT),
        STYLE_LOSS_WEIGHT=float(config.PFGIN_STYLE_LOSS_WEIGHT),
        CONTENT_LOSS_WEIGHT=float(config.PFGIN_CONTENT_LOSS_WEIGHT),
        INPAINT_ADV_LOSS_WEIGHT=float(config.PFGIN_ADV_LOSS_WEIGHT),
        modelpath=str(Path(model_dir) / "official_checkpoints"),
        GPU=list(config.PFGIN_GPU_IDS),
        MODE=1 if training else 2,
        DEVICE=device,
    )


def unwrap(module):
    return module.module if isinstance(module, torch.nn.DataParallel) else module


def fit_scaler(train_cubes, config, output_path):
    rng = np.random.default_rng(int(config.SEED) + 9101)
    total = int(config.PFGIN_SCALER_SPECTRA)
    per_cube = max(1, total // max(1, len(train_cubes)))
    chunks = []

    for cube in train_cubes:
        h, w, b = cube.shape
        n = min(per_cube, h * w)
        flat_idx = rng.choice(h * w, size=n, replace=False)
        ys = flat_idx // w
        xs = flat_idx % w
        spectra = np.asarray(cube[ys, xs, :], dtype=np.float32)
        chunks.append(spectra)

    samples = np.concatenate(chunks, axis=0)
    lo = np.nanquantile(samples, float(config.PFGIN_SCALER_LOW_Q), axis=0).astype(np.float32)
    hi = np.nanquantile(samples, float(config.PFGIN_SCALER_HIGH_Q), axis=0).astype(np.float32)
    bad = ~np.isfinite(lo) | ~np.isfinite(hi) | ((hi - lo) < 1e-6)
    lo[bad] = 0.0
    hi[bad] = 1.0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, lo=lo, hi=hi)
    return lo, hi


def load_scaler(path):
    data = np.load(path)
    return data["lo"].astype(np.float32), data["hi"].astype(np.float32)


def normalize_cube_patch(patch, lo, hi):
    patch = np.asarray(patch, dtype=np.float32)
    out = (patch - lo.reshape(1, 1, -1)) / (hi - lo).reshape(1, 1, -1)
    out = np.clip(out, 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def denormalize_values(values, bands, lo, hi):
    values = np.asarray(values, dtype=np.float32)
    bands = np.asarray(bands, dtype=np.int64)
    return values * (hi[bands] - lo[bands]) + lo[bands]


def tile_origin(coord, length, size, stride):
    if length <= size:
        return 0
    ideal = int(coord) - size // 2
    origin = int(round(ideal / stride) * stride)
    return max(0, min(origin, length - size))


def spectral_origin(band, bands, window, stride):
    if bands <= window:
        return 0
    ideal = int(band) - window // 2
    origin = int(round(ideal / stride) * stride)
    return max(0, min(origin, bands - window))


def extract_spatial(array, y0, x0, size, fill=0.0):
    y1 = min(array.shape[0], y0 + size)
    x1 = min(array.shape[1], x0 + size)
    piece = np.asarray(array[y0:y1, x0:x1, ...]).copy()
    py = size - piece.shape[0]
    px = size - piece.shape[1]
    if py <= 0 and px <= 0:
        return piece
    pads = [(0, max(0, py)), (0, max(0, px))] + [(0, 0)] * (piece.ndim - 2)
    if piece.dtype == bool:
        return np.pad(piece, pads, mode="constant", constant_values=True)
    if piece.shape[0] > 1 and piece.shape[1] > 1:
        return np.pad(piece, pads, mode="reflect")
    return np.pad(piece, pads, mode="constant", constant_values=fill)
