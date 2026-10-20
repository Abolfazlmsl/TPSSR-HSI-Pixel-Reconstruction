"""Geometry-guided reconstruction of missing hyperspectral measurements.

The geometry model provides structural features. The topology network
produces reconstructed reflectance without accessing masked target values.
"""
from __future__ import annotations

from pathlib import Path
import importlib.util
import math
import sys
import time
from typing import Iterable

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

HERE = Path(__file__).resolve()
CORE = next(p for p in HERE.parents if (p / "config.py").exists())
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
METHOD_DIR = CORE / "reconstruction_methods" / "tpssr"
# Some cleaned archives retain the same code under its legacy directory.
# This fallback changes the architecture-import location only, not checkpoint paths.
if not (METHOD_DIR / "train_geometry.py").is_file():
    legacy_dir = CORE / "reconstruction_methods" / "texture_restoration_router_rf_v2"
    if (legacy_dir / "train_geometry.py").is_file():
        METHOD_DIR = legacy_dir
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

import config
from src.evaluation_utils import regression_metrics, spectral_angle_metrics
from src.topology_completion import TopologyCompletionNetwork


METHOD = "tpssr"


def _import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


GEOMETRY_ARCH = _import_file(
    "final_tpssr_geometry_arch",
    METHOD_DIR / "train_geometry.py",
)


def torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _dataset_tag() -> str:
    return str(config.TEST_DATASET_FOLDER).lower()


def topology_checkpoint_path() -> Path:
    return Path(config.MODELS_DIR) / METHOD / f"topology_{_dataset_tag()}" / "best_model.pt"


def _candidate_geometry_paths(topology_ck: dict | None = None) -> list[Path]:
    root = Path(config.MODELS_DIR) / METHOD
    tag = _dataset_tag()
    candidates: list[Path] = []

    if topology_ck is not None:
        src = topology_ck.get("source_geometry_checkpoint")
        if src:
            p = Path(str(src))
            candidates.append(p)
            # Also support a moved project/model root by reusing the old parent folder.
            if p.parent.name:
                candidates.append(root / p.parent.name / p.name)

    candidates += [
        root / f"geometry_{tag}" / "best_model.pt",
    ]

    unique = []
    seen = set()
    for p in candidates:
        s = str(p)
        if s not in seen:
            seen.add(s)
            unique.append(p)
    return unique


def load_models(device: torch.device):
    """Load the exact final architecture.

    Returns: topology_model, topology_checkpoint, geometry_model, geometry_checkpoint,
    topology_path, geometry_path.
    """
    tf_path = topology_checkpoint_path()
    if not tf_path.exists():
        raise FileNotFoundError(
            f"Topology completion checkpoint not found: {tf_path}\n"
            "Run train.py before inference, or restore the trained checkpoints."
        )
    tf_ck = torch_load(tf_path, "cpu")
    if bool(tf_ck.get("uses_geometry_final_reconstruction", False)):
        raise RuntimeError("Final TPSSR requires Geometry Stage-A guidance only, not Geometry final reconstruction.")

    s = tf_ck["settings"]
    tf_model = TopologyCompletionNetwork(
        n_bands=int(tf_ck["n_bands"]),
        structure_feature_channels=int(s["structure_feature_channels"]),
        spectral_width=int(s["spectral_width"]),
        base_width=int(s["base_width"]),
        transport_embed_ch=int(s["transport_embed_ch"]),
        transport_heads=int(s["transport_heads"]),
        transport_topk=int(s["transport_topk"]),
        transport_radius=int(s["transport_radius"]),
        transport_temperature=float(s["transport_temperature"]),
        radiometry_delta_scale=float(s["radiometry_delta_scale"]),
        detail_scale=float(s["detail_scale"]),
    ).to(device)
    tf_model.load_state_dict(tf_ck["model_state_dict"], strict=True)
    tf_model.eval()

    geometry_path = next((p for p in _candidate_geometry_paths(tf_ck) if p.exists()), None)
    if geometry_path is None:
        tried = "\n".join(str(p) for p in _candidate_geometry_paths(tf_ck))
        raise FileNotFoundError(f"Geometry Stage-A checkpoint not found. Tried:\n{tried}")
    geometry_ck = torch_load(geometry_path, "cpu")
    vs = geometry_ck["settings"]
    geometry_model = GEOMETRY_ARCH.GeometryGuidanceModel(
        n_bands=int(geometry_ck["n_bands"]),
        spectral_width=int(vs["spectral_width"]),
        base_width=int(vs["base_width"]),
        attention_heads=int(vs["attention_heads"]),
    ).to(device)
    geometry_model.load_state_dict(geometry_ck["model_state_dict"], strict=True)
    geometry_model.eval()

    if int(geometry_ck["n_bands"]) != int(tf_ck["n_bands"]):
        raise RuntimeError("Geometry and TopologyFirst checkpoints have different spectral dimensions.")

    return tf_model, tf_ck, geometry_model, geometry_ck, tf_path, geometry_path


def comps4(mask2d: np.ndarray) -> list[np.ndarray]:
    m = np.asarray(mask2d, bool)
    seen = np.zeros_like(m, bool)
    out = []
    for sy, sx in np.argwhere(m):
        sy, sx = int(sy), int(sx)
        if seen[sy, sx]:
            continue
        stack = [(sy, sx)]
        seen[sy, sx] = True
        pts = []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < m.shape[0] and 0 <= xx < m.shape[1] and m[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        out.append(np.asarray(pts, np.int64))
    return out


def _bounds(cy: int, cx: int, ps: int, h: int, w: int):
    r = ps // 2
    sy0, sy1 = max(0, cy-r), min(h, cy+r+1)
    sx0, sx1 = max(0, cx-r), min(w, cx+r+1)
    dy0, dx0 = r-(cy-sy0), r-(cx-sx0)
    return sy0, sy1, sx0, sx1, dy0, dy0+sy1-sy0, dx0, dx0+sx1-sx0


def prepare_patch(test: np.ndarray, full_mask: np.ndarray, band: int, comp: np.ndarray, ck: dict, center_override=None):
    """Build the exact same model inputs as the original evaluator, locally.

    SAFE PERFORMANCE FIX
    --------------------
    The previous implementation normalized/copied the *entire HSI test cube*
    for every connected component of every affected band, then immediately
    sliced a 65x65 patch from that huge temporary array.  This function performs
    the identical pointwise normalization/masking only on the spatial slice that
    can actually enter the model.  No model weights, masks, patch geometry,
    normalization values, autocast settings, or metrics are changed.

    For the returned patch, the numerical arrays are element-for-element equal
    to the corresponding slice produced by the old full-cube implementation.
    """
    h, w, b = test.shape
    ps = int(ck.get("patch_size", 65))
    mean = np.asarray(ck["band_mean"], np.float32)
    std = np.maximum(np.asarray(ck["band_std"], np.float32), 1e-6)

    ys, xs = comp[:, 0], comp[:, 1]
    if center_override is None:
        cy = int(round((ys.min() + ys.max()) / 2))
        cx = int(round((xs.min() + xs.max()) / 2))
    else:
        cy, cx = map(int, center_override)
    sy0, sy1, sx0, sx1, dy0, dy1, dx0, dx1 = _bounds(cy, cx, ps, h, w)

    # IMPORTANT: only the local slice can reach the network.  Applying the same
    # pointwise operations here is mathematically identical to processing the
    # whole cube first and slicing afterwards, while avoiding enormous repeated
    # allocations/copies for large SWIR/VNIR cubes.
    local_test = np.asarray(test[sy0:sy1, sx0:sx1, :], dtype=np.float32)
    local_mask = np.asarray(full_mask[sy0:sy1, sx0:sx1, :], dtype=bool)
    finite = np.isfinite(local_test)
    safe = np.where(finite, local_test, mean[None, None, :])
    norm = (safe - mean[None, None, :]) / std[None, None, :]
    corrupted = norm.copy()
    corrupted[local_mask] = 0.0
    valid_fraction_local = np.mean(
        (finite & (~local_mask)).astype(np.float32), axis=2
    )

    cube = np.zeros((ps, ps, b), np.float32)
    cube[dy0:dy1, dx0:dx1] = corrupted
    target = np.zeros((ps, ps), np.float32)
    target[dy0:dy1, dx0:dx1] = norm[:, :, int(band)]

    m = np.zeros((ps, ps), np.float32)
    for y, x in comp:
        py, px = int(y)-sy0+dy0, int(x)-sx0+dx0
        if 0 <= py < ps and 0 <= px < ps:
            m[py, px] = 1.0

    obs = target.copy()
    obs[m > 0] = 0.0
    vfp = np.zeros((ps, ps), np.float32)
    vfp[dy0:dy1, dx0:dx1] = valid_fraction_local
    sv = ((m < 0.5) & (vfp > 0.95)).astype(np.float32)
    dist = distance_transform_edt(m > 0).astype(np.float32)
    dist /= max(float(dist.max()), 1.0)
    bp = 2.0 * band / max(b-1, 1) - 1.0

    tensors = dict(
        corrupted_cube=torch.from_numpy(np.moveaxis(cube, -1, 0)[None]),
        observed_target=torch.from_numpy(obs[None, None]),
        mask=torch.from_numpy(m[None, None]),
        spatial_valid=torch.from_numpy(sv[None, None]),
        valid_fraction=torch.from_numpy(vfp[None, None]),
        distance=torch.from_numpy(dist[None, None]),
        band_position=torch.tensor([bp], dtype=torch.float32),
    )
    meta = (sy0, sx0, dy0, dx0, ps, mean, std)
    return tensors, meta


@torch.inference_mode()
def _run_task_batch(
    tf_model,
    geometry_model,
    tasks,
    device: torch.device,
    verify_state: dict | None = None,
):
    """Run independent band/tile reconstruction tasks in one GPU batch.

    The first real batch of each mask is checked against batch-size=1.  If a
    numerical mismatch is too large, the caller automatically halves the batch
    until the check passes (or reaches exact batch-size=1).  This preserves the
    fast path where it is numerically safe without silently accepting the large
    FP16 batch deviations observed in some masks.
    """
    if not tasks:
        return []

    keys = tuple(tasks[0][3].keys())
    batch = {
        k: torch.cat([task[3][k] for task in tasks], dim=0).to(device, non_blocking=True)
        for k in keys
    }

    try:
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            structure = geometry_model.structure_net(**batch)
            out = tf_model(structure=structure, **batch)
        pred = out["prediction"][:, 0].float().cpu().numpy()

        if verify_state is not None and not verify_state.get("done", False):
            if len(tasks) == 1:
                verify_state["done"] = True
                verify_state["batch_cap"] = 1
                print("[FAST EXACT] exact fallback active: batch_size=1", flush=True)
            else:
                single = {k: tasks[0][3][k].to(device, non_blocking=True) for k in keys}
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    s1 = geometry_model.structure_net(**single)
                    o1 = tf_model(structure=s1, **single)
                ref = o1["prediction"][0, 0].float().cpu().numpy()
                diff = float(np.max(np.abs(ref - pred[0])))
                verify_state["last_diff"] = diff
                tol = float(verify_state.get("tol", 5e-4))
                if not np.isfinite(diff) or diff > tol:
                    raise RuntimeError(
                        f"BATCH_EQ_MISMATCH max_abs_diff={diff:.6g}"
                    )
                verify_state["done"] = True
                verify_state["batch_cap"] = min(
                    int(verify_state.get("batch_cap", len(tasks))), len(tasks)
                )
                print(
                    f"[FAST EXACT] batch equivalence check passed: "
                    f"batch={len(tasks)} max_abs_diff={diff:.3g}",
                    flush=True,
                )

        return [pred[i] for i in range(len(tasks))]
    finally:
        del batch


def _flush_tasks_adaptive(
    tf_model,
    geometry_model,
    tasks,
    device: torch.device,
    verify_state: dict | None,
):
    """Run a task batch, halving automatically on OOM or batch mismatch."""
    if not tasks:
        return []

    cap = int(verify_state.get("batch_cap", len(tasks))) if verify_state is not None else len(tasks)
    if len(tasks) > cap:
        out = []
        for i in range(0, len(tasks), cap):
            out.extend(
                _flush_tasks_adaptive(
                    tf_model, geometry_model, tasks[i:i+cap], device, verify_state
                )
            )
        return out

    try:
        return _run_task_batch(tf_model, geometry_model, tasks, device, verify_state)
    except RuntimeError as exc:
        msg = str(exc)
        is_oom = "out of memory" in msg.lower() and device.type == "cuda"
        is_mismatch = "BATCH_EQ_MISMATCH" in msg
        if not (is_oom or is_mismatch) or len(tasks) <= 1:
            raise

        if device.type == "cuda":
            torch.cuda.empty_cache()

        new_cap = max(1, len(tasks) // 2)
        if verify_state is not None:
            verify_state["batch_cap"] = min(
                int(verify_state.get("batch_cap", len(tasks))), new_cap
            )
            verify_state["done"] = False
        why = "OOM" if is_oom else "numerical mismatch"
        print(
            f"[FAST EXACT] {why} at batch={len(tasks)} -> retry batch={new_cap}",
            flush=True,
        )
        out = []
        for i in range(0, len(tasks), new_cap):
            out.extend(
                _flush_tasks_adaptive(
                    tf_model, geometry_model, tasks[i:i+new_cap], device, verify_state
                )
            )
        return out



def _component_from_seed(mask2d: np.ndarray, sy: int, sx: int) -> np.ndarray:
    """Return the 4-connected component containing one selected benchmark pixel.

    Unlike comps4(), this does not scan every masked voxel in the full band.  It
    only explores components that contain at least one selected evaluation
    coordinate.  This matters for sparse/isolated masks where the full mask may
    contain ~100k disconnected voxels but the historical benchmark evaluates at
    most a sampled subset of spatial pixels.
    """
    m = np.asarray(mask2d, dtype=bool)
    h, w = m.shape
    sy, sx = int(sy), int(sx)
    if not (0 <= sy < h and 0 <= sx < w and m[sy, sx]):
        return np.empty((0, 2), dtype=np.int64)

    stack = [(sy, sx)]
    seen = {(sy, sx)}
    pts = []
    while stack:
        y, x = stack.pop()
        pts.append((y, x))
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and m[yy, xx] and (yy, xx) not in seen:
                seen.add((yy, xx))
                stack.append((yy, xx))
    return np.asarray(pts, dtype=np.int64)


def _tile_selected_points(selected_coords: np.ndarray, comp: np.ndarray, patch_size: int):
    """Cover selected benchmark coordinates with deterministic interior tiles.

    Small components keep the historical single component-centred patch exactly.
    Large/long components (detector lines, band-wide masks, etc.) are tiled only
    as much as needed to cover selected benchmark points.  Every selected point
    is kept away from the patch edge when possible.
    """
    selected = np.asarray(selected_coords, dtype=np.int64)
    comp = np.asarray(comp, dtype=np.int64)
    if len(selected) == 0:
        return []

    ys, xs = comp[:, 0], comp[:, 1]
    hspan = int(ys.max() - ys.min() + 1)
    wspan = int(xs.max() - xs.min() + 1)
    ps = int(patch_size)

    if hspan <= ps and wspan <= ps:
        cy = int(round((int(ys.min()) + int(ys.max())) / 2.0))
        cx = int(round((int(xs.min()) + int(xs.max())) / 2.0))
        return [(cy, cx, selected)]

    # A modest interior margin avoids evaluating selected pixels directly on a
    # patch edge, while still covering long lines/large regions efficiently.
    radius = ps // 2
    safe_r = max(2, radius - 6)

    # Deterministic row-major greedy cover.
    order = np.lexsort((selected[:, 1], selected[:, 0]))
    remaining = selected[order].copy()
    tiles = []
    while len(remaining):
        sy, sx = map(int, remaining[0])
        near = (
            (np.abs(remaining[:, 0] - sy) <= safe_r)
            & (np.abs(remaining[:, 1] - sx) <= safe_r)
        )
        group0 = remaining[near]
        cy = int(round(float(np.median(group0[:, 0]))))
        cx = int(round(float(np.median(group0[:, 1]))))
        cover = (
            (np.abs(remaining[:, 0] - cy) <= safe_r)
            & (np.abs(remaining[:, 1] - cx) <= safe_r)
        )
        if not np.any(cover):
            cover[0] = True
            cy, cx = sy, sx
        group = remaining[cover]
        tiles.append((cy, cx, group))
        remaining = remaining[~cover]
    return tiles


def _selected_component_specs(mask3d: np.ndarray, eval_coords: np.ndarray, patch_size: int = 65):
    """Build selected band/component/tile tasks with full benchmark coverage.

    Small components reproduce the historical component-centred input exactly.
    Large components are deterministically tiled so long detector lines and
    band-wide regions no longer lose selected pixels outside one 65x65 patch.
    The complete corruption mask still enters ``corrupted_cube`` as context.
    """
    mask3d = np.asarray(mask3d, dtype=bool)
    coords = np.asarray(eval_coords, dtype=np.int64)
    if coords.size == 0:
        return []

    specs = []
    for band in np.unique(coords[:, 2]):
        band = int(band)
        band_sel = coords[coords[:, 2] == band, :2]
        order = np.lexsort((band_sel[:, 1], band_sel[:, 0]))
        band_sel = band_sel[order]
        selected_keys = {(int(y), int(x)) for y, x in band_sel}
        covered = set()
        target_mask = np.asarray(mask3d[:, :, band], dtype=bool)

        for y, x in band_sel:
            key = (int(y), int(x))
            if key in covered:
                continue
            comp = _component_from_seed(target_mask, key[0], key[1])
            if len(comp) == 0:
                continue
            comp_keys = {(int(cy), int(cx)) for cy, cx in comp}
            selected_here = sorted(selected_keys.intersection(comp_keys))
            if not selected_here:
                continue
            covered.update(selected_here)
            selected_arr = np.asarray(selected_here, dtype=np.int64)
            for cy, cx, tile_selected in _tile_selected_points(
                selected_arr, comp, int(patch_size)
            ):
                specs.append(
                    (
                        band,
                        comp,
                        np.asarray(tile_selected, dtype=np.int64),
                        (int(cy), int(cx)),
                    )
                )
    return specs


@torch.inference_mode()
def reconstruct_mask(
    tf_model,
    tf_ck: dict,
    geometry_model,
    test: np.ndarray,
    mask3d: np.ndarray,
    device: torch.device,
    bands: Iterable[int] | None = None,
    eval_coords: np.ndarray | None = None,
    batch_size: int = 8,
    verify_batch: bool = True,
    progress_desc: str | None = None,
    progress_position: int = 0,
):
    """Sparse, batched final reconstruction.

    Independent band/component tasks are batched. If eval_coords is supplied,
    only connected components that contain benchmark-selected coordinates are
    executed; the full corruption mask is still supplied to the model, and each
    selected component is reconstructed with exactly the same inputs as before.
    This matches the historical TPSSR benchmark sampling protocol while avoiding
    tens of thousands of unscored forwards for sparse masks.

    Returns
    -------
    band_predictions:
        {band: (coords_yx[N,2], predicted_values[N])}
    affected:
        affected band indices.
    """
    mask3d = np.asarray(mask3d, bool)

    # Historical benchmark protocol: score only select_eval_coords(), but keep
    # the COMPLETE corruption mask as model context.  This is the key speed fix
    # for sparse scenarios such as isolated_voxels: unscored disconnected holes
    # no longer trigger expensive Geometry+Topology forwards.
    if eval_coords is not None:
        eval_coords = np.asarray(eval_coords, dtype=np.int64)
        if eval_coords.ndim != 2 or (eval_coords.size and eval_coords.shape[1] != 3):
            raise ValueError("eval_coords must have shape [N,3].")
        if bands is not None and len(eval_coords):
            allowed = np.asarray(sorted(set(int(b) for b in bands)), dtype=np.int64)
            eval_coords = eval_coords[np.isin(eval_coords[:, 2], allowed)]
        affected = np.unique(eval_coords[:, 2]).astype(int) if len(eval_coords) else np.empty(0, dtype=int)
        task_specs = _selected_component_specs(mask3d, eval_coords, int(tf_ck.get("patch_size", 65)))
    else:
        affected = np.flatnonzero(np.any(mask3d, axis=(0, 1)))
        if bands is not None:
            allowed = set(int(b) for b in bands)
            affected = np.asarray([b for b in affected if int(b) in allowed], dtype=int)
        task_specs = []
        for band in affected:
            band = int(band)
            target_mask = np.asarray(mask3d[:, :, band], bool)
            for comp in comps4(target_mask):
                # Full-mask mode scores the complete component.
                ys, xs = comp[:, 0], comp[:, 1]
                cy = int(round((int(ys.min()) + int(ys.max())) / 2.0))
                cx = int(round((int(xs.min()) + int(xs.max())) / 2.0))
                task_specs.append((band, comp, comp, (cy, cx)))

    batch_size = max(1, int(batch_size))
    pending = []
    per_band_coords: dict[int, list[np.ndarray]] = {int(b): [] for b in affected}
    per_band_values: dict[int, list[np.ndarray]] = {int(b): [] for b in affected}
    verify_state = {"done": not bool(verify_batch), "batch_cap": batch_size, "tol": 5e-4}

    def consume(tasks, outputs):
        for task, fin in zip(tasks, outputs):
            band, _comp, selected_coords, _tensors, meta = task
            sy0, sx0, dy0, dx0, _ps, mean_b, std_b = meta
            coords = []
            values = []
            for y, x in selected_coords:
                py, px = int(y) - sy0 + dy0, int(x) - sx0 + dx0
                if 0 <= py < fin.shape[0] and 0 <= px < fin.shape[1]:
                    coords.append((int(y), int(x)))
                    values.append(float(fin[py, px] * std_b + mean_b))
            if coords:
                per_band_coords[int(band)].append(np.asarray(coords, dtype=np.int64))
                per_band_values[int(band)].append(np.asarray(values, dtype=np.float32))

    inner_bar = None
    if progress_desc is not None:
        inner_bar = tqdm(
            total=len(task_specs),
            desc=str(progress_desc),
            unit="task",
            dynamic_ncols=True,
            position=int(progress_position),
            leave=False,
            mininterval=0.5,
        )

    try:
        for band, comp, selected_coords, center_override in task_specs:
            tensors, meta0 = prepare_patch(
                test, mask3d, band, comp, tf_ck, center_override=center_override
            )
            sy0, sx0, dy0, dx0, ps, mean, std = meta0
            meta = (sy0, sx0, dy0, dx0, ps, float(mean[band]), float(std[band]))
            pending.append((band, comp, selected_coords, tensors, meta))
            if len(pending) >= batch_size:
                n_done = len(pending)
                outputs = _flush_tasks_adaptive(tf_model, geometry_model, pending, device, verify_state)
                consume(pending, outputs)
                pending.clear()
                if inner_bar is not None:
                    inner_bar.update(n_done)

        if pending:
            n_done = len(pending)
            outputs = _flush_tasks_adaptive(tf_model, geometry_model, pending, device, verify_state)
            consume(pending, outputs)
            pending.clear()
            if inner_bar is not None:
                inner_bar.update(n_done)
    finally:
        if inner_bar is not None:
            inner_bar.close()

    band_predictions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for band in affected:
        band = int(band)
        if not per_band_coords[band]:
            continue
        coords = np.concatenate(per_band_coords[band], axis=0)
        values = np.concatenate(per_band_values[band], axis=0)
        # Match the old np.where(valid) row-major ordering. If overlapping
        # components ever occur, keep the last computed value for that pixel.
        order = np.lexsort((coords[:, 1], coords[:, 0]))
        coords = coords[order]
        values = values[order]
        band_predictions[band] = (coords, values)

    return band_predictions, affected


def masked_ssim3_sparse(
    gt: np.ndarray,
    coords_yx: np.ndarray,
    pred_values: np.ndarray,
    target_mask2d: np.ndarray | None = None,
) -> float:
    """Exact sparse equivalent of the old masked_ssim3.

    The legacy prediction image started as GT and replaced only reconstructed
    pixels. We recreate each 3x3 window on demand instead of copying the entire
    2-D band, so the metric definition is unchanged.
    """
    gt = np.asarray(gt, float)
    coords_yx = np.asarray(coords_yx, dtype=np.int64)
    pred_values = np.asarray(pred_values, float)
    if len(coords_yx) == 0:
        return float("nan")

    finite = np.isfinite(pred_values)
    finite &= np.isfinite(gt[coords_yx[:, 0], coords_yx[:, 1]])
    coords_yx = coords_yx[finite]
    pred_values = pred_values[finite]
    if len(coords_yx) == 0:
        return float("nan")

    vals_gt = gt[coords_yx[:, 0], coords_yx[:, 1]]
    dr = max(float(np.nanpercentile(vals_gt, 99) - np.nanpercentile(vals_gt, 1)), 1e-6)
    c1, c2 = (.01 * dr) ** 2, (.03 * dr) ** 2
    lookup = {(int(y), int(x)): float(v) for (y, x), v in zip(coords_yx, pred_values)}
    target_mask2d = None if target_mask2d is None else np.asarray(target_mask2d, dtype=bool)
    vals = []
    h, w = gt.shape
    for y, x in coords_yx:
        y, x = int(y), int(x)
        if y < 1 or x < 1 or y >= h - 1 or x >= w - 1:
            continue
        a = gt[y-1:y+2, x-1:x+2].astype(float, copy=False).ravel()
        b = a.copy()
        k = 0
        window_valid = True
        for yy in range(y - 1, y + 2):
            for xx in range(x - 1, x + 2):
                pv = lookup.get((yy, xx))
                if pv is not None:
                    b[k] = pv
                elif target_mask2d is not None and bool(target_mask2d[yy, xx]):
                    # Historical sampled evaluator never substitutes hidden GT
                    # for a masked neighbour that was not selected/predicted.
                    window_valid = False
                    break
                k += 1
            if not window_valid:
                break
        if not window_valid or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        ma, mb = a.mean(), b.mean()
        va, vb = a.var(), b.var()
        cov = np.mean((a-ma) * (b-mb))
        den = (ma*ma + mb*mb + c1) * (va + vb + c2)
        if den > 1e-20:
            vals.append(((2*ma*mb+c1) * (2*cov+c2)) / den)
    return float(np.mean(vals)) if vals else float("nan")


def evaluate_reconstruction(
    gt_cube: np.ndarray,
    band_predictions: dict[int, tuple[np.ndarray, np.ndarray]],
    mask3d: np.ndarray,
    eval_coords: np.ndarray | None = None,
):
    target = np.asarray(mask3d, bool)
    truth_parts = []
    pred_parts = []
    coord_parts = []
    ssim_vals = []
    ssim_weights = []
    filled_voxels = 0

    for band in sorted(band_predictions):
        band = int(band)
        coords_yx, values = band_predictions[band]
        coords_yx = np.asarray(coords_yx, dtype=np.int64)
        values = np.asarray(values, dtype=np.float32)
        if len(coords_yx) == 0:
            continue
        yy, xx = coords_yx[:, 0], coords_yx[:, 1]
        gt_vals = np.asarray(gt_cube[yy, xx, band])
        valid = target[yy, xx, band] & np.isfinite(gt_vals) & np.isfinite(values)
        if not np.any(valid):
            continue
        c = coords_yx[valid]
        pv = values[valid]
        tv = gt_vals[valid]
        n = int(len(c))
        filled_voxels += n
        truth_parts.append(tv)
        pred_parts.append(pv)
        coord_parts.append(np.column_stack([c[:, 0], c[:, 1], np.full(n, band, dtype=np.int64)]))

        s = masked_ssim3_sparse(
            np.asarray(gt_cube[:, :, band]), c, pv, target_mask2d=target[:, :, band]
        )
        if np.isfinite(s):
            ssim_vals.append(float(s))
            ssim_weights.append(n)

    if truth_parts:
        truth = np.concatenate(truth_parts)
        pred = np.concatenate(pred_parts)
        coords = np.concatenate(coord_parts, axis=0)
    else:
        truth = np.empty(0, dtype=np.float32)
        pred = np.empty(0, dtype=np.float32)
        coords = np.empty((0, 3), dtype=np.int64)

    reg = regression_metrics(truth, pred)
    sam = spectral_angle_metrics(gt_cube, coords, pred)
    if ssim_vals:
        ssim_mean = float(np.mean(ssim_vals))
        ssim_weighted = float(np.average(ssim_vals, weights=ssim_weights))
    else:
        ssim_mean = float("nan")
        ssim_weighted = float("nan")

    full_mask_voxels = int(np.count_nonzero(target))
    full_spatial = np.any(target, axis=2)
    if eval_coords is None:
        target_voxels = full_mask_voxels
        eval_spatial_pixels = int(np.count_nonzero(full_spatial))
        eval_affected_bands = int(np.count_nonzero(np.any(target, axis=(0, 1))))
    else:
        ec = np.asarray(eval_coords, dtype=np.int64)
        target_voxels = int(len(ec))
        if len(ec):
            eval_spatial_pixels = int(len(np.unique(ec[:, :2], axis=0)))
            eval_affected_bands = int(len(np.unique(ec[:, 2])))
        else:
            eval_spatial_pixels = 0
            eval_affected_bands = 0
    return {
        "target_voxels": target_voxels,
        "full_mask_voxels": full_mask_voxels,
        "filled_voxels": int(filled_voxels),
        "coverage": float(filled_voxels / max(target_voxels, 1)),
        "masked_spatial_pixels": eval_spatial_pixels,
        "full_mask_spatial_pixels": int(np.count_nonzero(full_spatial)),
        "affected_bands": eval_affected_bands,
        "mae": float(reg["mae"]),
        "rmse": float(reg["rmse"]),
        "r2": float(reg["r2"]),
        "sam_deg": float(sam["sam_deg"]),
        "sam_pixel_coverage": float(sam["sam_pixel_coverage"]),
        "ssim": ssim_weighted,
        "ssim_mean_band": ssim_mean,
    }

