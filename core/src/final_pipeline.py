"""Final TPSSR end-to-end inference pipeline.

Architecture:
  Stage 1: Spectral vs Deep-v3.1
  Stage 2: learned hard-case gate
           -> Frozen V27 Stage-A -> V13 TopologyFirst V1 when needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

import config
from src.component_routing_utils import (
    connected_components_4,
    component_mask,
    representative_pixels,
)
from src.evaluation_utils import regression_metrics
from src.final_inference_utils import (
    torch_load,
    load_deep_v31,
    predict_pixel_previews,
    component_router_context,
)
from src.final_tpssr_runtime import (
    load_models as load_final_region_models,
    reconstruct_mask as reconstruct_final_region,
)
from src.hierarchical_component_router import HierarchicalComponentRouter


@dataclass
class FinalPipeline:
    device: torch.device
    deep_model: torch.nn.Module
    deep_checkpoint: dict
    deep_path: Path
    router_model: torch.nn.Module
    router_checkpoint: dict
    router_path: Path
    gate_threshold: float
    topology_model: torch.nn.Module
    topology_checkpoint: dict
    topology_path: Path
    v27_model: torch.nn.Module
    v27_path: Path
    deep_batch: int = 16
    region_batch: int = 32
    verify_region_batch: bool = True


def load_hierarchical_router(device):
    tag = str(config.TEST_DATASET_FOLDER).lower()
    path = (
        Path(config.MODELS_DIR)
        / "texture_restoration_router_rf_v2"
        / f"hierarchical_router_{tag}_dev_2gpu"
        / "best_model.pt"
    )
    if not path.exists():
        raise FileNotFoundError(
            "Final hierarchical-router checkpoint not found:\n"
            f"{path}\nRun train.py first."
        )
    ck = torch_load(path, "cpu")
    if ck.get("architecture") != HierarchicalComponentRouter.ARCHITECTURE_VERSION:
        raise RuntimeError("Hierarchical-router checkpoint architecture mismatch.")
    st = ck.get("settings", {})
    model = HierarchicalComponentRouter(
        n_bands=int(ck["n_bands"]),
        wavelengths_nm=np.asarray(ck["wavelengths_nm"], dtype=np.float32),
        d_model=int(st.get("d_model", 192)),
        n_heads=int(st.get("n_heads", 8)),
        n_layers=int(st.get("n_layers", 6)),
        ff_mult=int(st.get("ff_mult", 4)),
        dropout=float(st.get("dropout", 0.05)),
    )
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, ck, path, float(ck.get("gate_threshold", 0.5))


def load_pipeline(
    device,
    deep_batch=16,
    region_batch=32,
    verify_region_batch=True,
):
    deep_model, deep_ck, deep_path = load_deep_v31(device)
    router_model, router_ck, router_path, threshold = load_hierarchical_router(device)
    tf_model, tf_ck, v27_model, _v27_ck, tf_path, v27_path = (
        load_final_region_models(device)
    )
    return FinalPipeline(
        device=device,
        deep_model=deep_model,
        deep_checkpoint=deep_ck,
        deep_path=Path(deep_path),
        router_model=router_model,
        router_checkpoint=router_ck,
        router_path=Path(router_path),
        gate_threshold=float(threshold),
        topology_model=tf_model,
        topology_checkpoint=tf_ck,
        topology_path=Path(tf_path),
        v27_model=v27_model,
        v27_path=Path(v27_path),
        deep_batch=int(deep_batch),
        region_batch=int(region_batch),
        verify_region_batch=bool(verify_region_batch),
    )


@torch.inference_mode()
def route_component_hierarchical(
    pipeline,
    test_cube,
    mask,
    component,
    preview,
    pixel_lookup,
):
    (
        features,
        spectral,
        spectral_availability,
        spectral_confidence,
        deep,
        band_fraction,
        band_count,
        counts,
        component_feat,
    ) = component_router_context(
        test_cube,
        mask,
        component,
        preview,
        pixel_lookup,
        pipeline.deep_checkpoint,
        pipeline.router_checkpoint,
    )

    def tensor(x):
        return torch.from_numpy(np.asarray(x, dtype=np.float32)[None]).to(
            pipeline.device, non_blocking=True
        )

    aux = pipeline.router_model(
        tensor(features["input"]),
        tensor(spectral),
        tensor(spectral_availability),
        tensor(spectral_confidence),
        tensor(features["geometry"]),
        tensor(deep),
        tensor(band_fraction),
        tensor(band_count),
        tensor(component_feat),
    )

    predicted_error = aux["predicted_candidate_error"][0].float().cpu().numpy()
    stage1_probability = aux["stage1_probabilities"][0].float().cpu().numpy()
    gate_probability = aux["region_gate_probability"][0].float().cpu().numpy()

    active = counts > 0
    if not np.any(active):
        raise RuntimeError("Empty corruption component reached hierarchical router.")
    weights = counts[active].astype(np.float64)

    aggregate_error = np.average(predicted_error[active, :], axis=0, weights=weights)
    if np.any(spectral_availability[active] < 0.999):
        aggregate_error[0] = np.inf

    stage1 = int(np.argmin(aggregate_error[:2]))
    aggregate_stage1_probability = np.average(
        stage1_probability[active, :], axis=0, weights=weights
    )
    aggregate_gate_probability = float(
        np.average(gate_probability[active], weights=weights)
    )
    final_route = (
        2
        if aggregate_gate_probability >= float(pipeline.gate_threshold)
        else stage1
    )

    return {
        "stage1": stage1,
        "final": final_route,
        "gate": int(final_route == 2),
        "gate_probability": aggregate_gate_probability,
        "predicted_error": aggregate_error.astype(np.float32),
        "stage1_probability": aggregate_stage1_probability.astype(np.float32),
    }


def sparse_band_predictions(coords, pred):
    coords = np.asarray(coords, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.float32)
    out = {}
    for band in np.unique(coords[:, 2]):
        band = int(band)
        use = coords[:, 2] == band
        out[band] = (coords[use, :2].copy(), pred[use].copy())
    return out


def subset_rmse(truth, pred, use):
    use = np.asarray(use, dtype=bool)
    if not np.any(use):
        return float("nan")
    return float(
        regression_metrics(np.asarray(truth)[use], np.asarray(pred)[use])["rmse"]
    )


@torch.inference_mode()
def predict_mask(
    pipeline,
    test_cube,
    mask,
    eval_coords,
    preview_progress=None,
    region_progress=None,
):
    coords = np.asarray(eval_coords, dtype=np.int64)
    n = len(coords)
    if n == 0:
        raise RuntimeError("No evaluation coordinates for mask.")

    band_mean = np.asarray(pipeline.deep_checkpoint["band_mean"], dtype=np.float32)
    band_std = np.maximum(
        np.asarray(pipeline.deep_checkpoint["band_std"], dtype=np.float32), 1e-6
    )

    spatial_union = np.any(np.asarray(mask, dtype=bool), axis=2)
    raw_components = connected_components_4(spatial_union)
    components = []
    pixel_to_component = {}
    for cid, comp_coords in enumerate(raw_components):
        cmask = component_mask(spatial_union.shape, comp_coords)
        components.append(cmask)
        for y, x in comp_coords:
            pixel_to_component[(int(y), int(x))] = int(cid)

    needed_components = sorted(
        {pixel_to_component[(int(y), int(x))] for y, x, _b in coords}
    )
    eval_by_component = {cid: [] for cid in needed_components}
    for i, (y, x, _b) in enumerate(coords):
        eval_by_component[pixel_to_component[(int(y), int(x))]].append(int(i))

    pixel_set = {(int(y), int(x)) for y, x, _b in coords}
    for cid in needed_components:
        for y, x in representative_pixels(
            components[cid],
            int(getattr(config, "V31_COMPONENT_ROUTER_PREVIEW_PIXELS", 32)),
        ):
            pixel_set.add((int(y), int(x)))
    pixels = np.asarray(sorted(pixel_set), dtype=np.int64)
    preview = predict_pixel_previews(
        pipeline.deep_model,
        pipeline.deep_checkpoint,
        pipeline.router_checkpoint,
        test_cube,
        mask,
        pixels,
        pipeline.device,
        pipeline.deep_batch,
        progress_desc=preview_progress,
    )
    pixel_lookup = {tuple(map(int, p)): i for i, p in enumerate(pixels)}

    spectral_pred = np.full(n, np.nan, dtype=np.float32)
    deep_pred = np.full(n, np.nan, dtype=np.float32)
    spectral_available = np.zeros(n, dtype=bool)
    for i, (y, x, b) in enumerate(coords):
        row = pixel_lookup[(int(y), int(x))]
        b = int(b)
        spectral_pred[i] = (
            preview["spectral_norm"][row, b] * band_std[b] + band_mean[b]
        )
        deep_pred[i] = preview["deep_norm"][row, b] * band_std[b] + band_mean[b]
        spectral_available[i] = bool(preview["spectral_available"][row, b])

    stage1_route = np.full(n, -1, dtype=np.int64)
    final_route = np.full(n, -1, dtype=np.int64)
    gate_probability = np.full(n, np.nan, dtype=np.float32)
    predicted_error = np.full((n, 3), np.nan, dtype=np.float32)
    component_records = []
    spectral_fallback_components = 0

    for cid in needed_components:
        idx = np.asarray(eval_by_component[cid], dtype=np.int64)
        rr = route_component_hierarchical(
            pipeline,
            test_cube,
            mask,
            components[cid],
            preview,
            pixel_lookup,
        )
        stage1 = int(rr["stage1"])
        final = int(rr["final"])
        if final != 2 and stage1 == 0 and not bool(np.all(spectral_available[idx])):
            stage1 = 1
            final = 1
            spectral_fallback_components += 1

        stage1_route[idx] = stage1
        final_route[idx] = final
        gate_probability[idx] = float(rr["gate_probability"])
        predicted_error[idx, :] = rr["predicted_error"][None, :]
        component_records.append(
            {
                "component": int(cid),
                "area": int(np.count_nonzero(components[cid])),
                "eval_voxels": int(len(idx)),
                "stage1": int(stage1),
                "final": int(final),
                "gate_probability": float(rr["gate_probability"]),
            }
        )

    if np.any(stage1_route < 0) or np.any(final_route < 0):
        raise RuntimeError("Hierarchical routing left evaluation voxels unrouted.")

    final_pred = np.full(n, np.nan, dtype=np.float32)
    spec_idx = np.flatnonzero(final_route == 0)
    deep_idx = np.flatnonzero(final_route == 1)
    region_idx = np.flatnonzero(final_route == 2)
    if len(spec_idx):
        final_pred[spec_idx] = spectral_pred[spec_idx]
    if len(deep_idx):
        final_pred[deep_idx] = deep_pred[deep_idx]

    if len(region_idx):
        region_coords = coords[region_idx]
        region_band_preds, _ = reconstruct_final_region(
            pipeline.topology_model,
            pipeline.topology_checkpoint,
            pipeline.v27_model,
            test_cube,
            mask,
            pipeline.device,
            eval_coords=region_coords,
            batch_size=pipeline.region_batch,
            verify_batch=pipeline.verify_region_batch,
            progress_desc=region_progress or "Final Region",
        )
        region_lookup = {}
        for band, (yx, values) in region_band_preds.items():
            for (y, x), value in zip(yx, values):
                region_lookup[(int(y), int(x), int(band))] = float(value)
        for i in region_idx:
            key = tuple(map(int, coords[int(i)]))
            if key in region_lookup:
                final_pred[int(i)] = region_lookup[key]

    if np.any(~np.isfinite(final_pred)):
        bad = int(np.sum(~np.isfinite(final_pred)))
        raise RuntimeError(
            f"Final TPSSR end-to-end left {bad} requested voxels unfilled."
        )

    return {
        "prediction": final_pred,
        "spectral_prediction": spectral_pred,
        "deep_prediction": deep_pred,
        "spectral_available": spectral_available,
        "stage1_route": stage1_route,
        "final_route": final_route,
        "gate_probability": gate_probability,
        "predicted_error": predicted_error,
        "component_records": component_records,
        "spectral_fallback_components": int(spectral_fallback_components),
    }
