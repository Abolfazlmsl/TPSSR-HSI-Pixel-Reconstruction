"""TPSSR reconstruction pipeline with spectral and spatial experts.

The confidence gate reconstructs spectrally supported pixels. A fixed,
observed-only routing policy directs spatially difficult cases to the
geometry-guided topology model. An optional trained three-expert router can
select between interpolation, learned fusion and Region.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import config
from src.component_routing_utils import connected_components_4
from src.evaluation_utils import regression_metrics
from src.spatial_inference import (
    load_models as load_final_region_models,
    reconstruct_mask as reconstruct_final_region,
    torch_load,
)

@dataclass
class ModelBundle:
    """Inference models and measured-spectra normalization parameters."""

    device: torch.device
    topology_model: torch.nn.Module
    topology_checkpoint: dict
    topology_path: Path
    geometry_model: torch.nn.Module
    geometry_path: Path
    hybrid_model: torch.nn.Module
    hybrid_mean: np.ndarray
    hybrid_std: np.ndarray
    hybrid_wave_norm: np.ndarray
    hybrid_gate_checkpoint_path: Path
    hybrid_backbone_checkpoint_path: Path
    region_batch: int = 32
    verify_region_batch: bool = True
    hybrid_batch: int = 32
    routing_mode: str = "tpssr"
    fusion_model: torch.nn.Module | None = None
    fusion_checkpoint_path: Path | None = None
    router_checkpoint: dict | None = None
    router3_checkpoint: dict | None = None


def _load_accepted_spectral_hybrid_v2(device):
    """Load the trained band-wise confidence gate and its pinned backbone."""
    # These are the same classes that produced the benchmarked 200-mask V2
    # ablation; no reimplementation or silent substitution of weights.
    from training.spectral_gate import FrozenHybridGated, TRAIN_CORES, VALID_CORES, sha

    parent = Path(config.MODELS_DIR) / 'tpssr'
    from training.spectral_reference import (
        select_spectral_source_checkpoint, ACCEPTED_GATE, FrozenInterpolation)
    sensor = str(config.TEST_DATASET_FOLDER).upper().rstrip('0123456789')
    gate_path, mode = select_spectral_source_checkpoint(parent, sensor)
    gate = None
    if mode == ACCEPTED_GATE:
        gate = torch_load(gate_path, 'cpu')
        if gate.get('architecture') != 'spectral_only_hybrid_confidence_gate_v2' or gate.get('gate_variant') != 'band':
            raise RuntimeError('Expected accepted BAND-wise Hybrid V2 checkpoint.')
        if tuple(gate['run_config']['train_cores']) != tuple(TRAIN_CORES) or tuple(gate['run_config']['val_cores']) != tuple(VALID_CORES):
            raise RuntimeError('Hybrid checkpoint train/validation split mismatch.')
        base_guess = parent / 'spectral_backbone' / 'hybrid' / 'diagnostic_best_val.pt'
        candidates = [Path(gate.get('backbone_checkpoint') or base_guess), base_guess,
                      base_guess.with_name('best_model.pt')]
        backbone_path = next((p for p in candidates if p.is_file() and sha(p) == gate['backbone_sha256']), None)
        if backbone_path is None:
            raise FileNotFoundError('Frozen V1 backbone checkpoint missing or SHA256 mismatch: ' + str(base_guess))
    else:
        # The diagnostic checkpoint is used for normalization ONLY. Its
        # rejected neural prediction weights are never activated.
        backbone_path = gate_path
    source = torch_load(backbone_path, 'cpu')
    if source.get('architecture') != 'spectral_only_hybrid_residual_v1':
        raise RuntimeError('Wrong Hybrid V1 backbone architecture.')
    if tuple(source['run_config']['train_cores']) != tuple(TRAIN_CORES) or tuple(source['run_config']['val_cores']) != tuple(VALID_CORES):
        raise RuntimeError('Hybrid backbone split mismatch.')
    for core in (str(config.TEST_DATASET_FOLDER),):
        if core in set(TRAIN_CORES) | set(VALID_CORES):
            raise RuntimeError('Test core was used in training/validation of hybrid.')
    wavelength_path = Path(config.PROCESSED_DATA_ROOT) / str(config.TEST_DATASET_FOLDER) / 'wavelengths_nm.npy'
    waves = np.asarray(np.load(wavelength_path), np.float32)
    if len(waves) != int(source['model_params']['bands']):
        raise RuntimeError('Hybrid backbone spectral band count differs from test cube.')
    for ck in ((source, gate) if gate is not None else (source,)):
        if not np.allclose(np.asarray(ck['wavelengths_nm'],np.float32), waves, rtol=0, atol=.5):
            raise RuntimeError('Hybrid backbone/gate wavelength grid differs from test cube.')
    avg = np.asarray(source['band_mean'],np.float32)
    sd = np.asarray(source['band_std'],np.float32)
    if avg.shape != waves.shape or sd.shape != waves.shape or not np.isfinite(avg).all() or not np.isfinite(sd).all() or (sd<=0).any():
        raise RuntimeError('Invalid Hybrid backbone normalization statistics.')
    if mode == ACCEPTED_GATE:
        prm=source['model_params']
        model=FrozenHybridGated(len(waves),int(prm['width']),int(prm['heads']),int(prm['layers']),variant='band')
        model.base.load_state_dict(source['model_state_dict'],strict=True)
        model.gate.load_state_dict(gate['gate_state_dict'],strict=True)
    else:
        model=FrozenInterpolation()
        print('[TPSSR] VNIR spectral source: observed-band INTERPOLATION (gate rejected)', flush=True)
    model=model.to(device).eval()
    wn=(waves-waves[0])/max(float(waves[-1]-waves[0]),1e-6)
    return model,avg,sd,wn,gate_path,backbone_path

@torch.inference_mode()
def _hybrid_replace_spectral_only(pipeline, test_cube, mask, spectral_coords):
    """Predict only spectral route==Spectral voxels; NEVER use masked truths."""
    coords = np.asarray(spectral_coords,dtype=np.int64)
    n = len(coords)
    if n==0:
        return np.empty(0,np.float32),np.empty(0,bool)
    pixels,inv=np.unique(coords[:,:2],axis=0,return_inverse=True)
    b=int(test_cube.shape[-1]); xaxis=np.arange(b,dtype=np.float64)
    batch=int(pipeline.hybrid_batch)
    if batch<1:
        raise ValueError('Hybrid batch must be positive.')
    pixel_pred=np.full((len(pixels),b),np.nan,dtype=np.float32)
    for lo in range(0,len(pixels),batch):
        hi=min(lo+batch,len(pixels))
        chunk=pixels[lo:hi]
        features=np.zeros((len(chunk),b,4),dtype=np.float32)
        features[:,:,3]=pipeline.hybrid_wave_norm
        eligible=np.zeros(len(chunk),dtype=bool)
        for r,(y,x) in enumerate(chunk):
            spectrum=np.asarray(test_cube[int(y),int(x),:],np.float32)
            missing=np.asarray(mask[int(y),int(x),:],bool) | ~np.isfinite(spectrum)
            observed=np.flatnonzero(~missing)
            features[r,:,2]=missing.astype(np.float32)
            if len(observed)<2:
                continue
            anchor=np.interp(xaxis,observed,spectrum[observed]).astype(np.float32)
            features[r,observed,0]=(spectrum[observed]-pipeline.hybrid_mean[observed])/pipeline.hybrid_std[observed]
            features[r,:,1]=(anchor-pipeline.hybrid_mean)/pipeline.hybrid_std
            eligible[r]=True
        if np.any(eligible):
            inp=torch.as_tensor(features[eligible],device=pipeline.device)
            result=pipeline.hybrid_model(inp)
            result=result.detach().float().cpu().numpy()*pipeline.hybrid_std+pipeline.hybrid_mean
            pixel_pred[lo:hi][eligible]=result
    values=pixel_pred[inv,coords[:,2]]
    return values,np.isfinite(values)

def load_pipeline(device, region_batch=32, verify_region_batch=True, learned_fusion=False, learned_router=False, learned_router3=False, learned_router3_rmse=False, learned_router3_aligned=False, learned_router3_lineaware=False, learned_router3_lineaware_mse=False, learned_router3_lineaware_rankcv=False, learned_router3_lineaware_regretcv=False, learned_router2_lineaware=False):
    """Load the spectral and topology-guided spatial experts."""
    hybrid, mean, std, wave, gate_path, backbone_path = _load_accepted_spectral_hybrid_v2(device)
    topology, checkpoint, geometry, _geometry_checkpoint, topology_path, geometry_path = (
        load_final_region_models(device)
    )
    fusion_model, fusion_path = None, None
    if (learned_router or learned_router3 or learned_router3_rmse or learned_router3_aligned or learned_router3_lineaware or learned_router3_lineaware_mse or learned_router3_lineaware_rankcv or learned_router3_lineaware_regretcv or learned_router2_lineaware) and not learned_fusion:
        raise ValueError('Learned Router requires learned fusion')
    if learned_fusion:
        from training.neighbor_fusion import checkpoint_path, load_fusion_checkpoint
        from training.spectral_gate import TRAIN_CORES, VALID_CORES
        fusion_path = checkpoint_path(config.MODELS_DIR)
        if not fusion_path.is_file():
            raise FileNotFoundError('Learned fusion checkpoint missing: ' + str(fusion_path)
              + '. Run python train.py --stage fusion --gpus 2 first.')
        fusion_model, saved = load_fusion_checkpoint(
            fusion_path, device, gate_path,
            np.asarray(np.load(Path(config.PROCESSED_DATA_ROOT) /
                               str(config.TEST_DATASET_FOLDER) / 'wavelengths_nm.npy'),np.float32),
            mean, std)
        if (tuple(saved['training_cores']) != tuple(TRAIN_CORES) or
            tuple(saved['validation_cores']) != tuple(VALID_CORES)):
            raise RuntimeError('Learned fusion checkpoint source split mismatch')
        from training.spectral_gate import sha
        if saved.get('backbone_sha256') != sha(backbone_path):
            raise RuntimeError('Learned fusion backbone checkpoint mismatch')
    router_payload = None
    if learned_router:
        from training.learned_router import load_router_checkpoint, router_file
        from training.spectral_gate import VALID_CORES
        router_path = router_file(config.MODELS_DIR)
        if not router_path.is_file():
            raise FileNotFoundError('Learned Router not calibrated: ' + str(router_path))
        router_payload = load_router_checkpoint(router_path, gate_path, fusion_path,
                                                 tuple(VALID_CORES))
    router3_payload = None
    if learned_router3 or learned_router3_rmse or learned_router3_aligned or learned_router3_lineaware or learned_router3_lineaware_mse or learned_router3_lineaware_rankcv or learned_router3_lineaware_regretcv or learned_router2_lineaware:
        from training.three_expert_router import load_checkpoint, path_for
        from training.spectral_gate import VALID_CORES
        objective = ('regret_cv' if learned_router3_lineaware_regretcv else
                     'rank_cv' if learned_router3_lineaware_rankcv else
                     'mse' if (learned_router3_rmse or learned_router3_lineaware_mse) else 'log_rmse')
        region_expert = ('lineaware' if (learned_router3_lineaware or learned_router3_lineaware_mse or learned_router3_lineaware_rankcv or learned_router3_lineaware_regretcv) else
                         'refined' if learned_router3_aligned else 'raw')
        if learned_router2_lineaware:
            from training.three_expert_router import binary_path, load_binary_checkpoint
            path = binary_path(config.MODELS_DIR)
        else:
            path = path_for(config.MODELS_DIR, objective=objective, region_expert=region_expert)
        if not path.is_file():
            raise FileNotFoundError('Three-expert router checkpoint missing: ' + str(path))
        if learned_router3_lineaware or learned_router3_lineaware_mse or learned_router3_lineaware_rankcv or learned_router3_lineaware_regretcv or learned_router2_lineaware:
            from src.lineaware_region import load_lineaware_region
            waves = np.asarray(np.load(Path(config.PROCESSED_DATA_ROOT) /
                str(config.TEST_DATASET_FOLDER) / 'wavelengths_nm.npy'), dtype=np.float32)
            del topology, geometry
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            topology, checkpoint, geometry, _geom_ck, topology_path, geometry_path = (
                load_lineaware_region(device, waves, full_split=True))
        if learned_router2_lineaware:
            router3_payload = load_binary_checkpoint(path, gate_path, fusion_path,
                geometry_path, topology_path, tuple(VALID_CORES))
        else:
            router3_payload = load_checkpoint(path, gate_path, fusion_path,
                geometry_path, topology_path, tuple(VALID_CORES), objective=objective, region_expert=region_expert)
    # Accepted SWIR 8x2 inference uses the line-aware Region trained on the
    # same eight sources. The router checkpoint is validated against the
    # original Region first; no spectral or routing weights are changed.
    swir8_sources = ('SWIR1','SWIR2','SWIR3','SWIR4','SWIR6','SWIR7','SWIR8','SWIR9')
    use_a2b_trial = (
        os.environ.get('TPSSR_A2B_LINEAWARE_TEST') == '1' and
        tuple(config.TRAIN_DATASET_FOLDERS) == ('SWIR1','SWIR2','SWIR3','SWIR4','SWIR5') and
        tuple(config.TEST_DATASET_FOLDERS) == ('SWIR6','SWIR7','SWIR8','SWIR9','SWIR10') and
        str(config.TEST_DATASET_FOLDER).upper() == 'SWIR6'
    )
    use_b2a_trial = (
        os.environ.get('TPSSR_B2A_LINEAWARE_TEST') == '1' and
        tuple(config.TRAIN_DATASET_FOLDERS) == ('SWIR6','SWIR7','SWIR8','SWIR9','SWIR10') and
        tuple(config.TEST_DATASET_FOLDERS) == ('SWIR1','SWIR2','SWIR3','SWIR4','SWIR5') and
        str(config.TEST_DATASET_FOLDER).upper() == 'SWIR1'
    )
    use_lineaware = learned_router3_aligned and (
        (tuple(config.TRAIN_DATASET_FOLDERS) == swir8_sources and
         tuple(config.TEST_DATASET_FOLDERS) == ('SWIR5','SWIR10') and
         str(config.TEST_DATASET_FOLDER).upper() == 'SWIR5') or
        use_a2b_trial or use_b2a_trial
    )
    if use_lineaware:
        from src.lineaware_region import load_lineaware_region
        waves = np.asarray(np.load(Path(config.PROCESSED_DATA_ROOT) /
            str(config.TEST_DATASET_FOLDER) / 'wavelengths_nm.npy'), dtype=np.float32)
        del topology, geometry
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        topology, checkpoint, geometry, _geom_ck, topology_path, geometry_path = (
            load_lineaware_region(device, waves)
        )
    return ModelBundle(
        device=device,
        topology_model=topology,
        topology_checkpoint=checkpoint,
        topology_path=Path(topology_path),
        geometry_model=geometry,
        geometry_path=Path(geometry_path),
        hybrid_model=hybrid,
        hybrid_mean=mean,
        hybrid_std=std,
        hybrid_wave_norm=wave,
        hybrid_gate_checkpoint_path=gate_path,
        hybrid_backbone_checkpoint_path=backbone_path,
        fusion_model=fusion_model,
        fusion_checkpoint_path=fusion_path,
        router_checkpoint=router_payload,
        router3_checkpoint=router3_payload,
        region_batch=int(region_batch),
        verify_region_batch=bool(verify_region_batch),
    )


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
def _predict_base_routes(
    pipeline, test_cube, mask, eval_coords, region_progress=None
):
    """A *mask-feasibility* ablation with only frozen Hybrid and frozen Region.

    Same-pixel Hybrid needs >=2 finite observed bands.  Pixels that cannot
    satisfy this requirement instead use Region.  To respect exclusive routing
    per component without discarding valid spectra elsewhere in a mixed large
    spatial hole, split the masked 2D pixels into separate 4-connected
    *eligibility components* BEFORE routing.  Each such component is entirely
    Hybrid (0) or entirely Region (2); no blends and no Deep inference.

    Decisions use only the provided mask and observed data (never target truth,
    scenario or test-acquisition identifiers).  A learned two-expert Router is
    NOT trained by this experiment.
    """
    mask = np.asarray(mask, dtype=bool)
    coords = np.asarray(eval_coords, dtype=np.int64)
    n = int(len(coords))
    if n == 0:
        raise ValueError("No evaluation coordinates provided.")
    if test_cube.shape != mask.shape or coords.shape[1] != 3:
        raise ValueError("Mask/test cube/coordinate shape mismatch.")
    if pipeline.hybrid_model is None:
        raise RuntimeError("Accepted Hybrid V2 BAND checkpoint was not loaded.")
    if np.any(coords < 0) or np.any(coords >= np.array(mask.shape)[None, :]):
        raise ValueError("Evaluation coordinates outside test cube.")
    if not np.all(mask[coords[:, 0], coords[:, 1], coords[:, 2]]):
        raise ValueError("Evaluation coordinates include unmasked values.")

    # v2.0 remains untouched; v2.1 uses a frozen mask-geometry policy.
    # The new policy never accesses truth inside the mask or outcome labels.
    support = pipeline.routing_mode == "reference"
    guard = pipeline.routing_mode == "base"
    gap_policy = support or guard
    spatial_mask = np.any(mask, axis=2)
    observed_counts = np.sum((~mask) & np.isfinite(test_cube), axis=2, dtype=np.int32)
    if gap_policy:
        if guard:
            from src.routing_guard import expert_eligibility_maps
        else:
            from src.routing_support import expert_eligibility_maps
        can_hybrid, no_anchor_region, gap_region, policy_audit = (
            expert_eligibility_maps(mask, test_cube)
        )
        needs_region = no_anchor_region | gap_region
        eligibility_classes = (
            (0, can_hybrid, "hybrid_spectral_information_sufficient"),
            (2, no_anchor_region, "region_lt_2_observed_bands"),
            (2, gap_region, "region_long_gap_with_local_support_and_true_mask_purity" if guard else "region_long_gap_with_local_observed_neighbors"),
        )
    else:
        can_hybrid = spatial_mask & (observed_counts >= 2)
        needs_region = spatial_mask & ~can_hybrid
        policy_audit = dict(
            masked_pixels=int(spatial_mask.sum()),
            region_no_anchor_pixels=int(needs_region.sum()),
            long_gap_candidate_pixels=0,
            region_long_gap_pixels=0,
            hybrid_pixels=int(can_hybrid.sum()),
        )
        eligibility_classes = (
            (0, can_hybrid, "hybrid_ge_2_observed_bands"),
            (2, needs_region, "region_lt_2_observed_bands"),
        )

    final_route = np.full(n, -1, dtype=np.int64)
    final_route[can_hybrid[coords[:, 0], coords[:, 1]]] = 0
    final_route[needs_region[coords[:, 0], coords[:, 1]]] = 2
    if np.any(final_route < 0):
        raise RuntimeError("Two-Expert selection left evaluation voxels unrouted.")

    label_map = np.full(spatial_mask.shape, -1, dtype=np.int32)
    meta = []
    for route_id, eligible_map, description in eligibility_classes:
        for pixels in connected_components_4(eligible_map):
            pixels = np.asarray(pixels, dtype=np.int64)
            if not len(pixels):
                continue
            cid = len(meta)
            label_map[pixels[:, 0], pixels[:, 1]] = cid
            counts = observed_counts[pixels[:, 0], pixels[:, 1]]
            meta.append((route_id, description, len(pixels),
                         int(counts.min()), int(counts.max())))
    component_ids = label_map[coords[:, 0], coords[:, 1]]
    if np.any(component_ids < 0):
        raise RuntimeError("Masked evaluation coordinate was not assigned a component.")
    n_eval_by_component = np.bincount(component_ids, minlength=len(meta))
    records = []
    for cid, count in enumerate(n_eval_by_component):
        if count == 0:
            continue
        route_id, description, area, min_obs, max_obs = meta[cid]
        records.append({
            "component": int(cid),
            "area": int(area),
            "eval_voxels": int(count),
            "stage1": int(route_id),
            "final": int(route_id),
            "gate_probability": float(route_id == 2),
            "baseline_stage1": int(route_id),
            "baseline_final": int(route_id),
            "spectral_safe_override": 0,
            "stage1_spectral_override": 0,
            "stage1_decision_reason": "guard_true_mask_gap_purity_guard" if guard else "support_spectral_gap_spatial_support" if support else "observed_band_availability_only",
            "region_decision_reason": description,
            "guard_protected": 0,
            "guard_reason": "not_used_mask_feasibility_ablation",
            "router_predicted_spectral_error": float("nan"),
            "router_predicted_deep_error": float("nan"),
            "router_predicted_region_error": float("nan"),
            "effective_component_type": "hybrid" if route_id == 0 else "region",
            "min_observed_bands": min_obs,
            "max_observed_bands": max_obs,
        })

    prediction = np.full(n, np.nan, dtype=np.float32)
    hybrid_idx = np.flatnonzero(final_route == 0)
    region_idx = np.flatnonzero(final_route == 2)
    if len(hybrid_idx):
        values, eligible = _hybrid_replace_spectral_only(
            pipeline, test_cube, mask, coords[hybrid_idx]
        )
        if not np.all(eligible) or not np.isfinite(values).all():
            # Never substitute the old Spectral expert or a Deep checkpoint.
            raise RuntimeError(
                "Hybrid unexpectedly failed on a >=2-observed-band pixel."
            )
        prediction[hybrid_idx] = values
    if len(region_idx):
        reconstructed, _ = reconstruct_final_region(
            pipeline.topology_model,
            pipeline.topology_checkpoint,
            pipeline.geometry_model,
            test_cube,
            mask,
            pipeline.device,
            eval_coords=coords[region_idx],
            batch_size=pipeline.region_batch,
            verify_batch=pipeline.verify_region_batch,
            progress_desc=region_progress or "Hybrid V2 + Region",
        )
        # Avoid allocating Region predictions for unrequested bands.
        lookup = {}
        for band, (yx, values) in reconstructed.items():
            for (y, x), val in zip(yx, values):
                lookup[(int(y), int(x), int(band))] = float(val)
        for i in region_idx:
            prediction[i] = lookup.get(tuple(map(int, coords[i])), np.nan)

    if not np.isfinite(prediction).all():
        raise RuntimeError(
            f"Two-Expert reconstruction left {np.count_nonzero(~np.isfinite(prediction))} "
            "evaluation voxels unfilled."
        )
    if np.any(final_route == 1):
        raise AssertionError("disabled expert was selected in Hybrid/Region ablation.")

    return {
        "prediction": prediction,
        "spectral_prediction": np.where(final_route == 0, prediction, np.nan),  # legacy route-0 key
        "deep_prediction": np.full(n, np.nan, dtype=np.float32),
        "spectral_available": final_route == 0,  # route 0 = Hybrid, not raw Spectral
        "stage1_route": final_route.copy(),
        "final_route": final_route,
        "gate_probability": (final_route == 2).astype(np.float32),
        "predicted_error": np.full((n, 3), np.nan, dtype=np.float32),
        "component_records": records,
        "spectral_fallback_components": 0,
        "hybrid_changed_voxels": int(len(hybrid_idx)),
        "hybrid_fallback_voxels": 0,
        "v6_fixed_reference_prediction": None,
        "support_policy_audit": policy_audit,
        "support_region_gap_eval_voxels": int(np.sum(gap_region[coords[:, 0], coords[:, 1]])) if gap_policy else 0,
        "guard_policy_audit": policy_audit if guard else {},
        "guard_region_gap_eval_voxels": int(np.sum(gap_region[coords[:, 0], coords[:, 1]])) if guard else 0,
    }


def _policy_component_rows(mask,cube,coords,extra):
    """Regenerate consistent component audit after V2.3 extra-region routes."""
    from src.routing_guard import expert_eligibility_maps
    hybrid,_,region,_=expert_eligibility_maps(mask,cube)
    new_hybrid=hybrid & (~extra)
    new_region=(region | extra) | (np.any(mask,axis=2) & (~hybrid) & (~region))
    observed=np.sum((~mask) & np.isfinite(cube),axis=2,dtype=np.int32)
    rows=[];cid=0
    for final,eligible,reason in ((0,new_hybrid,'policy_hybrid_observed_spectrum'),
                                  (2,new_region,'policy_region_gap_or_no_anchors')):
        for pixels in connected_components_4(eligible):
            pixels=np.asarray(pixels,dtype=np.int64)
            if not len(pixels):continue
            yy=pixels[:,0];xx=pixels[:,1]
            # Component count only for entries present among official evaluation coordinates.
            mask_comp=np.zeros(eligible.shape,bool);mask_comp[yy,xx]=True
            n_eval=int(np.sum(mask_comp[coords[:,0],coords[:,1]]))
            if not n_eval:continue
            counts=observed[yy,xx]
            rows.append(dict(component=cid,area=int(len(pixels)),eval_voxels=n_eval,
                stage1=final,final=final,gate_probability=float(final==2),
                baseline_stage1=final,baseline_final=final,spectral_safe_override=0,
                stage1_spectral_override=0,stage1_decision_reason='policy_mask_geometry',
                region_decision_reason=reason,guard_protected=0,
                guard_reason='not_used',router_predicted_spectral_error=float('nan'),
                router_predicted_deep_error=float('nan'),
                router_predicted_region_error=float('nan'),
                effective_component_type='hybrid' if final==0 else 'region',
                min_observed_bands=int(counts.min()),max_observed_bands=int(counts.max())))
            cid+=1
    return rows

@torch.inference_mode()
def predict_mask(pipeline, test_cube, mask, eval_coords,
                 preview_progress=None, region_progress=None):
    from src.routing_policy import route_extra_region,local_neighbor_median
    from src.routing_guard import expert_eligibility_maps
    coords=np.asarray(eval_coords,np.int64)
    mask=np.asarray(mask,bool)
    # Compute the observed-only base routing before applying the local policy.
    original_mode = pipeline.routing_mode
    pipeline.routing_mode = 'base'
    try:
        out = _predict_base_routes(
            pipeline, test_cube, mask, coords, region_progress=region_progress
        )
    finally:
        pipeline.routing_mode = original_mode
    base=np.asarray(out['prediction'],np.float32)
    old_route=np.asarray(out['final_route'],np.int64)
    eligible_hybrid,_,_,_=expert_eligibility_maps(mask,test_cube)
    extra,stats=route_extra_region(test_cube,mask,eligible_hybrid,coords)
    add=(old_route==0) & extra[coords[:,0],coords[:,1]]
    new=base.copy();new_route=old_route.copy()
    if np.any(add):
        rebuilt,_=reconstruct_final_region(
            pipeline.topology_model,pipeline.topology_checkpoint,
            pipeline.geometry_model,test_cube,mask,pipeline.device,
            eval_coords=coords[add],batch_size=pipeline.region_batch,
            verify_batch=pipeline.verify_region_batch,
            progress_desc=region_progress or 'V2.3 extra Region')
        lookup={}
        for band,(yx,values) in rebuilt.items():
            for (y,x),val in zip(yx,values):
                key=(int(y),int(x),int(band))
                if key in lookup:raise RuntimeError('Duplicate V2.3 Region voxel')
                lookup[key]=float(val)
        for i in np.flatnonzero(add):
            new[i]=lookup.get(tuple(map(int,coords[i])),np.nan)
        new_route[add]=2
    med,median_stats=local_neighbor_median(test_cube,mask,coords,new_route)
    use=(new_route==2)&np.isfinite(med)
    new[use]=med[use]
    if not np.isfinite(new).all():
        raise RuntimeError('Incomplete V2.3 reconstruction')
    if not np.array_equal(new[(old_route==0)&~add],base[(old_route==0)&~add]):
        raise AssertionError('V2.3 changed an untouched Hybrid prediction')
    if not np.array_equal(new[(old_route==2)&~use],base[(old_route==2)&~use]):
        raise AssertionError('V2.3 changed an untouched Region prediction')
    out['prediction']=new
    out['spectral_prediction']=np.where(new_route==0,new,np.nan)
    out['final_route']=new_route
    out['stage1_route']=new_route.copy()
    out['spectral_available']=new_route==0
    out['gate_probability']=(new_route==2).astype(np.float32)
    out['component_records']=_policy_component_rows(mask,test_cube,coords,extra)
    out['routing_audit']=dict(**stats,**median_stats,
               additional_region_voxels=int(add.sum()),
               local_median_voxels=int(use.sum()))
    out['policy_baseline_guard_prediction']=base
    return out


# Candidate policy for development-only comparison against frozen V2.3.
# The original predict_mask() above remains byte-for-byte unchanged.
CANDIDATE_NEIGHBOR_MIN_DONORS = 5
CANDIDATE_NEIGHBOR_MAX_DONORS = 8
CANDIDATE_SHORT_GAP_BANDS = 10
CANDIDATE_NEIGHBOR_CORRECTION = 0.5


def _candidate_longest_gap(missing):
    edges = np.flatnonzero(np.diff(np.pad(np.asarray(missing, bool).astype(np.int8), (1, 1))))
    return int(np.max(edges[1::2] - edges[::2])) if len(edges) else 0


def _candidate_neighbors(cube, mask, y, x, hole, observed, anchor):
    """Observed-only transfer of spectral curvature from similar intact neighbors.

    This deliberately matches the non-learned neighbor-trend probe: donor
    interpolation excludes the same hole, and similarity is evaluated ONLY at
    wavelengths observed at the target. No masked target value is read.
    """
    target = np.asarray(cube[y, x, :], np.float32)
    reference = target[observed].astype(np.float64)
    if reference.size < 30:
        return None, 0
    scale = max(float(np.std(reference)), 0.01)
    n_target = int(observed.sum())
    h, w, bands = cube.shape
    xaxis = np.arange(bands, dtype=np.float64)
    candidates = []
    last_radius = 0
    for radius in (4, 8, 14):
        for yy in range(max(0, y - radius), min(h, y + radius + 1)):
            for xx in range(max(0, x - radius), min(w, x + radius + 1)):
                dist = max(abs(yy-y), abs(xx-x))
                if dist <= last_radius or (yy == y and xx == x):
                    continue
                # Quick reject using only the donor's measured mask and values.
                donor = np.asarray(cube[yy, xx, :], np.float32)
                donor_valid = (~mask[yy, xx, :]) & np.isfinite(donor)
                if not np.all(donor_valid[hole]):
                    continue
                joint = donor_valid & observed
                if int(joint.sum()) < max(30, int(0.7 * n_target)):
                    continue
                nx = donor[joint].astype(np.float64)
                ty = target[joint].astype(np.float64)
                mx, my = float(nx.mean()), float(ty.mean())
                var = float(np.mean((nx - mx) ** 2))
                if var < 1e-9:
                    continue
                slope = float(np.mean((nx - mx) * (ty - my)) / var)
                if not 0.5 <= slope <= 1.5:
                    continue
                intercept = my - slope * mx
                dissimilarity = float(np.sqrt(np.mean((slope*nx + intercept - ty)**2))) / scale
                if not np.isfinite(dissimilarity) or dissimilarity > 0.75:
                    continue
                support = np.flatnonzero(donor_valid & ~hole)
                if len(support) < 2:
                    continue
                donor_interp = np.interp(xaxis, support, donor[support])
                residual = slope * (donor[hole].astype(np.float64) - donor_interp[hole])
                if not np.isfinite(residual).all():
                    continue
                score = dissimilarity + 0.012 * float(np.hypot(yy-y, xx-x))
                candidates.append((score, residual))
        if len(candidates) >= CANDIDATE_NEIGHBOR_MIN_DONORS:
            break
        last_radius = radius
    if len(candidates) < CANDIDATE_NEIGHBOR_MIN_DONORS:
        return None, len(candidates)
    candidates.sort(key=lambda entry: entry[0])
    residuals = np.stack([r for _, r in candidates[:CANDIDATE_NEIGHBOR_MAX_DONORS]])
    correction = np.median(residuals, axis=0)
    values = anchor[hole].astype(np.float64) + CANDIDATE_NEIGHBOR_CORRECTION*correction
    if not np.isfinite(values).all():
        return None, len(candidates)
    return values.astype(np.float32), len(candidates)


@torch.inference_mode()
def predict_mask_candidate(pipeline, test_cube, mask, eval_coords,
                           preview_progress=None, region_progress=None):
    """Development candidate, without altering V2.3 or checkpoint weights.

    Region is mandatory only if fewer than two finite observed bands remain.
    Otherwise, prefer observed-neighbor spectral curvature (>=5 donors), then
    the frozen Hybrid on long gaps, or interpolation on short ones. This is a
    *fixed hypothesis* selected after inspecting source diagnostics; its final
    performance is unknown and it is NOT a clean held-out model selection.
    """
    from src.routing_policy import local_neighbor_median, route_extra_region
    cube = np.asarray(test_cube)
    mask = np.asarray(mask, bool)
    coords = np.asarray(eval_coords, np.int64)
    if cube.shape != mask.shape or coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('Invalid cube, mask or evaluation coordinates')
    # Feasibility-only mode invokes Region only for pixels with fewer than
    # two observed bands and frozen Hybrid for all other masked pixels.
    previous_mode = pipeline.routing_mode
    pipeline.routing_mode = 'candidate_feasibility_only'
    try:
        out = _predict_base_routes(pipeline, cube, mask, coords,
                                   region_progress=region_progress)
    finally:
        pipeline.routing_mode = previous_mode
    prediction = np.asarray(out['prediction'], np.float32).copy()
    route = np.asarray(out['final_route'], np.int64).copy()
    method = np.full(len(coords), 'region', dtype='<U12')
    spectral = np.flatnonzero(route == 0)
    method[spectral] = 'hybrid'
    counters = dict(neighbor_pixels=0, hybrid_pixels=0,
                    interpolation_pixels=0, no_donor_pixels=0,
                    region_pixels=0, region_median_voxels=0)
    if len(spectral):
        # A wavelength masked everywhere cannot have an observed donor.
        globally_masked = np.all(mask, axis=(0, 1))
        pixels, inv = np.unique(coords[spectral, :2], axis=0, return_inverse=True)
        ordered = np.argsort(inv, kind='stable')
        first = np.r_[0, np.flatnonzero(np.diff(inv[ordered])) + 1]
        last = np.r_[first[1:], len(ordered)]
        bands = cube.shape[2]
        axis = np.arange(bands, dtype=np.float64)
        for i, (yy, xx) in enumerate(pixels):
            yy, xx = int(yy), int(xx)
            where = spectral[ordered[first[i]:last[i]]]
            target = np.asarray(cube[yy, xx, :], np.float32)
            hole = np.asarray(mask[yy, xx, :], bool)
            observed = ~hole & np.isfinite(target)
            observed_idx = np.flatnonzero(observed)
            if len(observed_idx) < 2:
                raise RuntimeError('Candidate spectral pixel lacks anchors')
            anchor = np.interp(axis, observed_idx, target[observed_idx]).astype(np.float32)
            if np.any(globally_masked & hole):
                neighbor, donors = None, 0
            else:
                neighbor, donors = _candidate_neighbors(
                    cube, mask, yy, xx, hole, observed, anchor)
            if neighbor is not None:
                # The return order is the order of True entries in hole.
                predicted_bands = np.flatnonzero(hole)
                values = neighbor[np.searchsorted(predicted_bands, coords[where, 2])]
                prediction[where] = values
                method[where] = 'neighbor'
                counters['neighbor_pixels'] += 1
            elif _candidate_longest_gap(~observed) < CANDIDATE_SHORT_GAP_BANDS:
                prediction[where] = anchor[coords[where, 2]]
                method[where] = 'interp'
                counters['interpolation_pixels'] += 1
                counters['no_donor_pixels'] += 1
            else:
                # Preserve the original frozen Hybrid prediction.
                method[where] = 'hybrid'
                counters['hybrid_pixels'] += 1
                counters['no_donor_pixels'] += 1
    # Preserve existing measured-neighbor replacement for fully-masked pixels.
    med, median_stats = local_neighbor_median(cube, mask, coords, route)
    use = (route == 2) & np.isfinite(med)
    prediction[use] = med[use]
    method[use] = 'local_median'
    counters['region_pixels'] = int(np.unique(coords[route == 2, :2], axis=0).shape[0])
    counters['region_median_voxels'] = int(use.sum())
    if not np.isfinite(prediction).all():
        raise RuntimeError('Candidate reconstruction left nonfinite values')
    # Diagnostic only: count how many V2.3 optional long-gap routes are bypassed.
    from src.routing_guard import expert_eligibility_maps
    can_hybrid, _, _, _ = expert_eligibility_maps(mask, cube)
    extra, old_policy = route_extra_region(cube, mask, can_hybrid, coords)
    old_extra_voxels = int(np.sum((route == 0) & extra[coords[:, 0], coords[:, 1]]))
    for record in out['component_records']:
        if int(record.get('final', -1)) == 0:
            record['effective_component_type'] = 'spectral_candidate_selector'
    out['prediction'] = prediction
    out['spectral_prediction'] = np.where(route == 0, prediction, np.nan)
    out['final_route'] = route
    out['stage1_route'] = route.copy()
    out['candidate_method'] = method
    out['candidate_method_voxels'] = {name: int(np.sum(method == name))
                                       for name in ('interp','hybrid','neighbor','region','local_median')}
    out['candidate_policy_audit'] = dict(**counters,
        old_extra_region_voxels_bypassed=old_extra_voxels,
        old_extra_region_gap_candidates=int(old_policy['gap_candidates']),
        **median_stats)
    out['routing_audit'] = dict(**median_stats, additional_region_voxels=0,
        local_median_voxels=int(use.sum()),
        old_extra_region_voxels_bypassed=old_extra_voxels)
    out['hybrid_changed_voxels'] = int(np.sum(method == 'hybrid'))
    out['hybrid_fallback_voxels'] = int(np.sum(method == 'interp'))
    out['policy_baseline_guard_prediction'] = None
    return out


# Candidate V2 builds on V1 without retraining or modifying the accepted models.
# Parameters are fixed engineering hypotheses, not fitted on test-mask errors.
CANDIDATE_V2_SPARSE_MISSING_FRACTION = 0.85
CANDIDATE_V2_SPARSE_LONGEST_FRACTION = 0.60
CANDIDATE_V2_MIN_SPATIAL_SUPPORT = 5
CANDIDATE_V2_SPATIAL_OBSERVED_FRACTION = 0.95
CANDIDATE_V2_GLOBAL_BAND_FRACTION = 0.995
CANDIDATE_V2_GLOBAL_HYBRID_WEIGHT = 0.50


def _candidate_v2_sparse_region_pixels(cube, mask, coords, route):
    """Flag extreme spectral losses only when nearby measured pixels support the hole.

    Nothing in this function reads a target value at a masked wavelength.
    Spatial support is counted on the full true mask, not the sampled evaluation
    coordinates. A severe gap alone is insufficient to activate Region.
    """
    h, w, bands = cube.shape
    selected = np.zeros((h, w), dtype=bool)
    spectral_pixels = np.unique(coords[route == 0, :2], axis=0)
    for y, x in spectral_pixels:
        y, x = int(y), int(x)
        observed = ~mask[y, x] & np.isfinite(cube[y, x])
        n_observed = int(observed.sum())
        hole = ~observed
        if n_observed < 2 or float(hole.mean()) < CANDIDATE_V2_SPARSE_MISSING_FRACTION:
            continue
        if _candidate_longest_gap(hole) < CANDIDATE_V2_SPARSE_LONGEST_FRACTION * bands:
            continue
        # The neighboring measurements must actually cover the missing gap.
        enough = 0
        for yy in range(max(0, y-1), min(h, y+2)):
            for xx in range(max(0, x-1), min(w, x+2)):
                if yy == y and xx == x:
                    continue
                donor_ok = ~mask[yy, xx] & np.isfinite(cube[yy, xx])
                if float(np.mean(donor_ok[hole])) >= CANDIDATE_V2_SPATIAL_OBSERVED_FRACTION:
                    enough += 1
        if enough >= CANDIDATE_V2_MIN_SPATIAL_SUPPORT:
            selected[y, x] = True
    return selected


def _candidate_v2_component_rows(mask, cube, coords, region_extra):
    """Rebuild component accounting only if V2 introduces an extra Region route."""
    observed_counts = np.sum((~mask) & np.isfinite(cube), axis=2, dtype=np.int32)
    spatial_mask = np.any(mask, axis=2)
    region = spatial_mask & ((observed_counts < 2) | region_extra)
    spectral = spatial_mask & ~region
    index = np.full(mask.shape[:2], -1, np.int32)
    components = []
    for route_id, active, reason in ((0, spectral, 'v2_spectral_feasible'),
                                    (2, region, 'v2_no_anchors_or_supported_sparse_gap')):
        for pixels in connected_components_4(active):
            pixels = np.asarray(pixels, np.int64)
            if len(pixels) == 0:
                continue
            cid = len(components)
            index[pixels[:, 0], pixels[:, 1]] = cid
            obs = observed_counts[pixels[:, 0], pixels[:, 1]]
            components.append((route_id, reason, int(len(pixels)),
                               int(obs.min()), int(obs.max())))
    counts = np.bincount(index[coords[:, 0], coords[:, 1]], minlength=len(components))
    rows = []
    for cid, n in enumerate(counts):
        if not n:
            continue
        route_id, reason, area, low, high = components[cid]
        rows.append(dict(component=cid, area=area, eval_voxels=int(n),
                         stage1=route_id, final=route_id,
                         gate_probability=float(route_id == 2),
                         baseline_stage1=route_id, baseline_final=route_id,
                         spectral_safe_override=0, stage1_spectral_override=0,
                         stage1_decision_reason='observed_only_candidate_v2',
                         region_decision_reason=reason,
                         guard_protected=0, guard_reason='not_used',
                         router_predicted_spectral_error=float('nan'),
                         router_predicted_deep_error=float('nan'),
                         router_predicted_region_error=float('nan'),
                         effective_component_type='hybrid' if route_id == 0 else 'region',
                         min_observed_bands=low, max_observed_bands=high))
    return rows


@torch.inference_mode()
def predict_mask_candidate_v2(pipeline, test_cube, mask, eval_coords,
                              preview_progress=None, region_progress=None):
    """Conservative V2 candidate: preserve V1 and modify only two edge cases.

    For image-wide missing bands, shrink the unsupported frozen-Hybrid correction
    halfway toward same-pixel interpolation. For extremely sparse continuous
    spectra, try Region only when surrounding measured data cover the hole.
    All choices use the mask and observed values; there is no test-truth oracle.
    """
    from src.routing_policy import local_neighbor_median
    cube = np.asarray(test_cube)
    mask = np.asarray(mask, bool)
    coords = np.asarray(eval_coords, np.int64)
    out = predict_mask_candidate(pipeline, cube, mask, coords,
                                 preview_progress=preview_progress,
                                 region_progress=region_progress)
    prediction = np.asarray(out['prediction'], np.float32).copy()
    route = np.asarray(out['final_route'], np.int64).copy()
    method = np.asarray(out['candidate_method'], dtype='<U20').copy()

    # Do not mistake a few damaged rows for a missing wavelength everywhere.
    global_bands = mask.mean(axis=(0, 1)) >= CANDIDATE_V2_GLOBAL_BAND_FRACTION
    global_hybrid = np.flatnonzero((route == 0) & (method == 'hybrid') & global_bands[coords[:, 2]])
    if global_hybrid.size:
        band_axis = np.arange(cube.shape[2], dtype=np.float64)
        pixels, inverse = np.unique(coords[global_hybrid, :2], axis=0, return_inverse=True)
        for j, (yy, xx) in enumerate(pixels):
            target = np.asarray(cube[int(yy), int(xx), :], np.float32)
            measured = (~mask[int(yy), int(xx)]) & np.isfinite(target)
            support = np.flatnonzero(measured)
            if support.size < 2:
                continue
            anchor = np.interp(band_axis, support, target[support])
            positions = global_hybrid[inverse == j]
            baseline = anchor[coords[positions, 2]]
            prediction[positions] = (baseline + CANDIDATE_V2_GLOBAL_HYBRID_WEIGHT *
                                      (prediction[positions] - baseline)).astype(np.float32)
            method[positions] = 'band_guard'

    # Unlike V2.3, a moderate-length gap is never routed to Region on length alone.
    extra_pixels = _candidate_v2_sparse_region_pixels(cube, mask, coords, route)
    extra = (route == 0) & extra_pixels[coords[:, 0], coords[:, 1]]
    extra_local_median_voxels = 0
    if np.any(extra):
        rebuilt, _ = reconstruct_final_region(
            pipeline.topology_model, pipeline.topology_checkpoint,
            pipeline.geometry_model, cube, mask, pipeline.device,
            eval_coords=coords[extra], batch_size=pipeline.region_batch,
            verify_batch=pipeline.verify_region_batch,
            progress_desc=region_progress or 'Candidate V2 supported extreme Region')
        lookup = {}
        for band, (yx, values) in rebuilt.items():
            for (y, x), value in zip(yx, values):
                key = (int(y), int(x), int(band))
                if key in lookup:
                    raise RuntimeError('Duplicate candidate V2 Region output')
                lookup[key] = float(value)
        for i in np.flatnonzero(extra):
            prediction[i] = lookup.get(tuple(map(int, coords[i])), np.nan)
        route[extra] = 2
        method[extra] = 'region_extreme'
        # Keep the same observed-neighbor safety replacement as V1 for Region.
        local, _ = local_neighbor_median(cube, mask, coords, route)
        accepted = extra & np.isfinite(local)
        prediction[accepted] = local[accepted]
        method[accepted] = 'local_median'
        extra_local_median_voxels = int(np.sum(accepted))
        out['component_records'] = _candidate_v2_component_rows(mask, cube, coords, extra_pixels)

    if not np.isfinite(prediction).all():
        raise RuntimeError('Candidate V2 left nonfinite values')
    out['prediction'] = prediction
    out['final_route'] = route
    out['stage1_route'] = route.copy()
    out['spectral_prediction'] = np.where(route == 0, prediction, np.nan)
    out['spectral_available'] = route == 0
    out['gate_probability'] = (route == 2).astype(np.float32)
    out['candidate_method'] = method
    names = ('interp', 'hybrid', 'neighbor', 'region', 'local_median',
             'band_guard', 'region_extreme')
    out['candidate_method_voxels'] = {name: int(np.sum(method == name)) for name in names}
    out['candidate_policy_audit'].update(
        v2_global_missing_bands=int(global_bands.sum()),
        v2_band_guard_voxels=int(np.sum(method == 'band_guard')),
        v2_region_extreme_voxels=int(np.sum(extra)),
        v2_region_extreme_pixels=int(np.sum(extra_pixels)),
        v2_policy='band_guard_0.5_and_supported_sparse_region')
    out['routing_audit']['additional_region_voxels'] = int(np.sum(extra))
    out['routing_audit']['local_median_voxels'] = (
        int(out['routing_audit'].get('local_median_voxels', 0)) + extra_local_median_voxels)
    out['hybrid_changed_voxels'] = int(np.sum(method == 'hybrid'))
    out['hybrid_fallback_voxels'] = int(np.sum(method == 'interp'))
    return out


@torch.inference_mode()
def predict_mask_learned_fusion(pipeline, test_cube, mask, eval_coords,
                                preview_progress=None, region_progress=None,
                                preserve_v2_band_guard=False):
    """Replace V2 spectral estimates with learned fusion, preserving Region.

    This evaluation route is intentionally separate from neighbor_v2. The
    accepted V2 route and all existing checkpoints remain unchanged.
    """
    if pipeline.fusion_model is None:
        raise RuntimeError('Learned fusion model was not loaded')
    from training.neighbor_fusion import crop_at, make_features
    cube = np.asarray(test_cube)
    damaged = np.asarray(mask, bool)
    coords = np.asarray(eval_coords, np.int64)
    out = predict_mask_candidate_v2(pipeline, cube, damaged, coords,
                                    preview_progress=preview_progress,
                                    region_progress=region_progress)
    route = np.asarray(out['final_route'], np.int64)
    method = np.asarray(out['candidate_method'], dtype='<U20').copy()
    protected = ((route == 0) & (method == 'band_guard')) if preserve_v2_band_guard else np.zeros(len(route), bool)
    indices = np.flatnonzero((route == 0) & ~protected)
    if not len(indices):
        out['candidate_policy_audit']['learned_fusion_voxels'] = 0
        out['candidate_policy_audit']['fusion_v2_band_guard_preserved_voxels'] = int(protected.sum())
        return out
    pixels, inv = np.unique(coords[indices,:2],axis=0,return_inverse=True)
    pred = np.asarray(out['prediction'],np.float32).copy()
    batch_size = max(1,int(pipeline.hybrid_batch))
    b = cube.shape[2]
    collect_router = bool(getattr(pipeline, 'router_checkpoint', None)
                          and pipeline.router_checkpoint.get('enabled', False))
    router_features, router_interpolations = [], []
    if collect_router:
        from training.learned_router import feature_row
    wave = np.asarray(np.load(Path(config.PROCESSED_DATA_ROOT) /
                              str(config.TEST_DATASET_FOLDER) / 'wavelengths_nm.npy'),np.float32)
    # Avoid materializing whole HSI cubes or reading a hidden target value.
    for lo in range(0,len(pixels),batch_size):
        hi = min(lo+batch_size,len(pixels))
        local = []
        for y,x in pixels[lo:hi]:
            crop, bad, cy,cx = crop_at(cube, damaged, int(y),int(x))
            features=make_features(crop,bad,cy,cx,
                                   pipeline.hybrid_mean,pipeline.hybrid_std,wave)
            if features is None:
                raise RuntimeError('Spectral fusion route lacks observed anchors')
            local.append(features)
        tensors=[torch.as_tensor(np.stack([sample[k] for sample in local]),
                                 device=pipeline.device) for k in range(5)]
        hybrid=pipeline.hybrid_model(tensors[1]).detach()
        out_norm=pipeline.fusion_model(tensors[0],hybrid,tensors[2],tensors[3],tensors[4])
        normalized = out_norm.cpu().numpy()
        raw = normalized * pipeline.hybrid_std + pipeline.hybrid_mean
        if collect_router:
            for sample, fusion_norm in zip(local, normalized):
                router_features.append(feature_row(sample[0], sample[2], sample[3],
                                                   sample[4], fusion_norm))
                router_interpolations.append(sample[0][:, 1] * pipeline.hybrid_std
                                             + pipeline.hybrid_mean)
        group = (inv >= lo) & (inv < hi)
        positions = indices[group]
        values = raw[inv[group]-lo,coords[positions,2]]
        pred[positions]=values.astype(np.float32)
        method[positions]='learned_fusion'
    if not np.isfinite(pred).all():
        raise RuntimeError('Learned fusion produced nonfinite masked predictions')
    out['prediction']=pred
    out['spectral_prediction']=np.where(route==0,pred,np.nan)
    out['candidate_method']=method
    out['candidate_method_voxels']={name:int(np.sum(method==name))
        for name in ('interp','hybrid','neighbor','region','local_median',
                     'band_guard','region_extreme','learned_fusion')}
    out['candidate_policy_audit']['learned_fusion_voxels']=int(indices.size)
    out['candidate_policy_audit']['fusion_v2_band_guard_preserved_voxels']=int(protected.sum())
    out['candidate_policy_audit']['fusion_training']='frozen_hybrid_plus_source_trained_local_donor_attention'
    if collect_router:
        out['_router_pixels'] = pixels
        out['_router_features'] = np.stack(router_features)
        out['_router_interpolations'] = np.stack(router_interpolations)
    return out


@torch.inference_mode()
def predict_mask_learned_fusion_guarded(pipeline, test_cube, mask, eval_coords,
                                        preview_progress=None, region_progress=None):
    """Learned fusion except where V2's global-band guard already protects Hybrid."""
    return predict_mask_learned_fusion(
        pipeline, test_cube, mask, eval_coords,
        preview_progress=preview_progress, region_progress=region_progress,
        preserve_v2_band_guard=True,
    )


@torch.inference_mode()
def predict_mask_learned_router(pipeline, test_cube, mask, eval_coords,
                                preview_progress=None, region_progress=None):
    """Learned interpolation/Fusion choice with V2 Region and band guard intact.

    No Region error label is estimated from in-sample Region training data.
    Later cross-fitted Region calibration can expand this to three Experts.
    """
    if pipeline.router_checkpoint is None:
        raise RuntimeError('Learned Router checkpoint was not loaded')
    from training.learned_router import choice
    coords = np.asarray(eval_coords, np.int64)
    out = predict_mask_learned_fusion_guarded(
        pipeline, test_cube, mask, coords, preview_progress, region_progress)
    method = np.asarray(out['candidate_method'], dtype='<U20').copy()
    eligible = np.flatnonzero(method == 'learned_fusion')
    audit = out['candidate_policy_audit']
    audit['learned_router_enabled'] = int(pipeline.router_checkpoint['enabled'])
    audit['learned_router_candidates'] = int(len(eligible))
    audit['learned_router_interp_voxels'] = 0
    if not len(eligible) or not pipeline.router_checkpoint['enabled']:
        return out

    pixels, reverse = np.unique(coords[eligible, :2], axis=0, return_inverse=True)
    cached_pixels = out.pop('_router_pixels')
    if not np.array_equal(cached_pixels, pixels):
        raise RuntimeError('Learned Router feature cache is misaligned')
    features = out.pop('_router_features')
    interpolations = out.pop('_router_interpolations')
    prediction = np.asarray(out['prediction'], np.float32).copy()
    choose_interp = choice(pipeline.router_checkpoint['model'], features,
                           pipeline.router_checkpoint['enabled'])
    for i in np.flatnonzero(choose_interp):
        selected = eligible[reverse == i]
        prediction[selected] = interpolations[i][coords[selected, 2]]
        method[selected] = 'router_interp'
    out['prediction'] = prediction
    out['spectral_prediction'] = np.where(np.asarray(out['final_route']) == 0, prediction, np.nan)
    out['candidate_method'] = method
    out['candidate_method_voxels'] = {name: int(np.sum(method == name))
        for name in ('interp', 'hybrid', 'neighbor', 'region', 'local_median',
                     'band_guard', 'region_extreme', 'learned_fusion', 'router_interp')}
    audit['learned_router_interp_voxels'] = int(np.sum(method == 'router_interp'))
    audit['learned_router_interp_pixels'] = int(choose_interp.sum())
    if not np.isfinite(prediction).all():
        raise RuntimeError('Learned Router produced a nonfinite reconstruction')
    return out


def spatial_consistent_router_choices(pixels, risks, strength, passes=5):
    """Minimize learned unary risk plus a 4-connected Potts prior.

    The regularization weight must be calibrated on source-core data. This
    function never uses the masked reference, damage scenario, or severity.
    Zero strength is the original trained router, bit-for-bit.
    """
    pixels = np.asarray(pixels, dtype=np.int64)
    risks = np.asarray(risks, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or risks.shape != (len(pixels), 3):
        raise ValueError('Spatial router input shape mismatch')
    if not np.isfinite(strength) or strength < 0 or passes < 1:
        raise ValueError('Invalid spatial router calibration')
    if len(pixels) != len(np.unique(pixels, axis=0)):
        raise ValueError('Spatial router requires unique pixels')
    if not np.isfinite(np.min(risks, axis=1)).all():
        raise ValueError('Spatial router has no available expert for a pixel')
    original = np.argmin(risks, axis=1)
    if strength == 0 or len(pixels) < 2:
        return original

    # Log relative risk has a common dimensionless scale for the three models.
    # Invalid experts remain forbidden, regardless of neighborhood labels.
    minrisk = np.min(risks, axis=1)
    unary = np.log(np.maximum(risks, 1e-12)) - np.log(np.maximum(minrisk[:, None], 1e-12))
    unary[~np.isfinite(risks)] = np.inf
    lookup = {tuple(row): i for i, row in enumerate(pixels)}
    neighbors = [[] for _ in range(len(pixels))]
    for i, (y, x) in enumerate(pixels):
        for q in ((int(y)-1, int(x)), (int(y)+1, int(x)),
                  (int(y), int(x)-1), (int(y), int(x)+1)):
            j = lookup.get(q)
            if j is not None:
                neighbors[i].append(j)
    labels = original.copy()
    # Checkerboard coordinate updates make the output independent of the
    # order in which eval_coords happened to be provided.
    for _ in range(passes):
        changed = 0
        for color in (0, 1):
            for i, (y, x) in enumerate(pixels):
                if (int(y) + int(x)) % 2 != color or not neighbors[i]:
                    continue
                local = unary[i].copy()
                nearby = labels[neighbors[i]]
                for k in range(3):
                    local[k] += strength * int(np.count_nonzero(nearby != k))
                new = int(np.argmin(local))
                if new != labels[i]:
                    labels[i] = new
                    changed += 1
        if not changed:
            break
    return labels


def load_fusion_guard_calibration(pipeline):
    """Load locked SOURCE-only calibration and verify trained model identities.

    No target-core measurements are used to configure this correction.
    """
    import json
    import hashlib
    report_path = (Path(config.MODELS_DIR) / "tpssr" /
                   "learned_router3_lineaware_rankcv" /
                   "fusion_guard_calibration" / "FUSION_GUARD_CALIBRATION.json")
    if not report_path.is_file():
        raise FileNotFoundError(f"Fusion Guard calibration missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("architecture") != "source_cv_fusion_artifact_guard_candidate_v1"
            or not report.get("activation_recommended")
            or report.get("preset") != str(config.EXPERIMENT_NAME)
            or set(report.get("source_only_cores", [])) != {"SWIR4", "SWIR9"}
            or set(report.get("held_out_test_cores", [])) != {"SWIR5", "SWIR10"}):
        raise RuntimeError("Fusion Guard calibration does not belong to the current SWIR 8x2 split")
    def sha(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1048576), b""):
                h.update(chunk)
        return h.hexdigest()
    for key, path in (("fusion_sha256", pipeline.fusion_checkpoint_path),
                      ("gate_sha256", pipeline.hybrid_gate_checkpoint_path)):
        if path is None or not Path(path).is_file() or sha(path) != report.get(key):
            raise RuntimeError(f"Fusion Guard {key} differs from the source-calibrated checkpoint")
    params = report.get("selected_parameters", {})
    if not (float(params.get("contrast_factor", 0)) > 0 and
            float(params.get("minimum_contrast", 0)) > 0 and
            float(params.get("side_agreement_factor", 0)) > 0 and
            0 <= float(params.get("blend", -1)) <= 1):
        raise RuntimeError("Invalid Fusion Guard parameters")
    return dict(params={k: float(params[k]) for k in
                        ("contrast_factor", "minimum_contrast", "side_agreement_factor", "blend")},
                report=str(report_path), report_sha256=sha(report_path))


def _apply_sparse_fusion_guard(pipeline, cube, mask, coords, pixels, reverse,
                               decisions, fusion_all, prediction, calibration, wave):
    """Apply a locked Fusion-only stripe guard on sampled target voxels.

    In contrast to ROI evaluation, the official mask test samples target pixels.
    Independently predict horizontally adjacent masked Fusion *support* spectra
    when they were not sampled, so source-only calibration and sparse inference
    use equivalent local evidence.  Support predictions are NEVER scored.
    """
    from training.neighbor_fusion import crop_at
    from training.three_expert_router import features_for_pixel
    pars = calibration["params"]
    h, w, bands = cube.shape
    # The complete spectral estimates already evaluated by the three-expert Router.
    support = {(int(y), int(x)): fusion_all[i]
               for i, (y, x) in enumerate(pixels) if fusion_all[i] is not None}
    # Index selected target voxels, avoiding a full H x W x B prediction buffer.
    groups = [[] for _ in range(len(pixels))]
    for i, pidx in enumerate(reverse):
        groups[int(pidx)].append(i)
    changed_pixels = 0
    changed_voxels = 0
    original_prediction = prediction.copy()
    for pidx, (py, px) in enumerate(pixels):
        if int(decisions[pidx]) != 1 or not groups[pidx]:
            continue
        y, x = int(py), int(px)
        if x <= 0 or x >= w-1:
            continue
        indices = np.asarray(groups[pidx], np.int64)
        target_bands = coords[indices, 2]
        # The guard is calibrated only for bands masked at all three pixels.
        if len(target_bands) < 20 or not (np.all(mask[y, x-1, target_bands]) and
                                             np.all(mask[y, x+1, target_bands])):
            continue
        anchor = (~mask[y, x] & ~mask[y, x-1] & ~mask[y, x+1] &
                  np.isfinite(cube[y, x]) & np.isfinite(cube[y, x-1]) &
                  np.isfinite(cube[y, x+1]))
        if int(anchor.sum()) < 30:
            continue
        observed_contrast = float(np.median(np.abs(
            cube[y, x, anchor] - (cube[y, x-1, anchor] + cube[y, x+1, anchor]) * .5)))
        neighbors = []
        for nx in (x-1, x+1):
            key = (y, nx)
            if key not in support:
                patch, damaged, cy, cx = crop_at(cube, mask, y, nx, radius=35)
                _, _, fusion = features_for_pixel(
                    patch, damaged, cy, cx, pipeline.hybrid_mean, pipeline.hybrid_std,
                    wave, pipeline.fusion_model, pipeline.hybrid_model, pipeline.device)
                support[key] = fusion
            neighbors.append(support[key])
        if any(v is None for v in neighbors):
            continue
        left, right = neighbors
        c = original_prediction[indices]
        l = np.asarray(left)[target_bands]
        r = np.asarray(right)[target_bands]
        if not (np.isfinite(c).all() and np.isfinite(l).all() and np.isfinite(r).all()):
            continue
        estimate = .5 * (l + r)
        pred_contrast = float(np.median(np.abs(c - estimate)))
        spread = float(np.median(np.abs(l-r)))
        if (pred_contrast > max(pars["minimum_contrast"],
                                pars["contrast_factor"] * observed_contrast)
                and pred_contrast > pars["side_agreement_factor"] * max(spread, .0004)):
            prediction[indices] = ((1-pars["blend"]) * c +
                                   pars["blend"] * estimate).astype(np.float32)
            changed_pixels += 1
            changed_voxels += len(indices)
    return dict(fusion_guard_enabled=1, fusion_guard_changed_pixels=changed_pixels,
                fusion_guard_changed_voxels=changed_voxels,
                fusion_guard_calibration_sha256=calibration["report_sha256"])


def rankcv_global_minority_choices(pixels, raw_choices, eligible, threshold=0.10,
                                   counts_override=None, policy='threshold',
                                   three_expert_min_gap=0.10):
    """Apply an observed-only, whole-mask routing policy once.

    threshold: replace every active expert whose share is <= threshold.
    hybrid: remove a unique least-used expert whenever three are active.
    hybrid_gap: with three active experts, remove the unique least-used only
    if its share is at least three_expert_min_gap below the SECOND least-used.
    Both hybrid policies use threshold only when two experts are active.
    One selected expert is never changed.
    A tie for least-used expert is left unchanged to avoid arbitrary removal.
    Replacement favors the globally most common surviving, eligible expert.
    Counts concern damaged spatial pixels, never spectral voxels or GT values.
    """
    pixels = np.asarray(pixels, np.int64)
    original = np.asarray(raw_choices, np.int64)
    eligible = np.asarray(eligible, bool)
    n = len(original)
    if n == 0 or pixels.shape != (n, 2) or eligible.shape != (n, 3):
        raise ValueError('RankCV minority input dimensions are inconsistent')
    if not np.all((original >= 0) & (original < 3)):
        raise ValueError('RankCV minority received an unknown expert choice')
    if not (0.0 < float(threshold) < 0.5):
        raise ValueError('Global minority threshold must be between 0 and 0.5')
    if not (0.0 <= float(three_expert_min_gap) < 1.0):
        raise ValueError('Three-expert minimum gap must be in [0, 1)')
    if not np.all(eligible[np.arange(n), original]):
        raise RuntimeError('RankCV selected an unavailable expert')
    if counts_override is None:
        counts = np.bincount(original, minlength=3)
    else:
        counts = np.asarray(counts_override, dtype=np.int64)
        if counts.shape != (3,) or np.any(counts < 0) or int(counts.sum()) < n:
            raise ValueError('Invalid global RankCV expert counts')
    total = int(counts.sum())
    active = [i for i in range(3) if counts[i] > 0]
    if policy == 'threshold':
        minority = {i for i in active if counts[i] / total <= threshold}
    elif policy in ('hybrid', 'hybrid_gap'):
        minority = set()
        if len(active) == 3:
            ordered = sorted(active, key=lambda i: (int(counts[i]), i))
            smallest, second = ordered[:2]
            unique_least = int(counts[smallest]) < int(counts[second])
            sufficient_gap = ((int(counts[second]) - int(counts[smallest])) / total
                              >= float(three_expert_min_gap))
            if unique_least and (policy == 'hybrid' or sufficient_gap):
                minority.add(smallest)
        elif len(active) == 2:
            less = min(active, key=lambda i: int(counts[i]))
            if int(counts[less]) / total < threshold:
                minority.add(less)
    else:
        raise ValueError(f'Unknown Global Minority policy: {policy}')
    allowed = [i for i in active if i not in minority]
    if not minority or not allowed:
        return original.copy()
    global_order = sorted(allowed, key=lambda k: (-int(counts[k]), k))
    selected = original.copy()
    for i in range(n):
        if original[i] in minority:
            for candidate in global_order:
                if eligible[i, candidate]:
                    selected[i] = candidate
                    break
    return selected


@torch.inference_mode()
def predict_mask_learned_router3(pipeline, test_cube, mask, eval_coords,
                                 preview_progress=None, region_progress=None,
                                 region_full_eval=False, spatial_consistency=None,
                                 fusion_guard=None, global_minority_threshold=None,
                                 global_minority_counts=None, routing_only=False,
                                 global_minority_policy='threshold',
                                 three_expert_min_gap=0.10, routing_cache=None):
    """Three-way learned routing; no V2 component routing or band-guard policy."""
    from training.neighbor_fusion import crop_at
    from training.three_expert_router import features_for_pixel, predict_risks

    if pipeline.fusion_model is None or pipeline.router3_checkpoint is None:
        raise RuntimeError('Three-expert checkpoints are required')
    cube = np.asarray(test_cube)
    mask = np.asarray(mask, bool)
    coords = np.asarray(eval_coords, np.int64)
    if cube.shape != mask.shape or coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('Router input shapes are inconsistent')
    if len(coords) == 0 or not np.all(mask[coords[:,0],coords[:,1],coords[:,2]]):
        raise ValueError('Requested coordinates must be masked')
    if np.any(coords < 0) or np.any(coords >= np.asarray(mask.shape)):
        raise ValueError('Coordinates out of bounds')

    wave = np.asarray(np.load(Path(config.PROCESSED_DATA_ROOT) /
                       str(config.TEST_DATASET_FOLDER) / 'wavelengths_nm.npy'), np.float32)
    pixels, reverse = np.unique(coords[:,:2], axis=0, return_inverse=True)
    base_cache = (routing_cache.load_features(pixels, cube.shape[-1])
                  if routing_cache is not None and not routing_only else None)
    if base_cache is not None:
        eligible = base_cache['eligible'].astype(bool)
        risks = base_cache['risks']
        independent_decisions = base_cache['raw_choices'].astype(np.int64)
        interp_all = [base_cache['interp'][i] if eligible[i, 0] else None
                      for i in range(len(pixels))]
        fusion_all = [base_cache['fusion'][i] if eligible[i, 1] else None
                      for i in range(len(pixels))]
    else:
        x, interp_all, fusion_all = [], [], []
        # Compute only observed-only features; the reference truth is not queried.
        for y, xx in pixels:
            patch, damaged, cy, cx = crop_at(cube, mask, int(y), int(xx), radius=35)
            feat, interp, fusion = features_for_pixel(
                patch, damaged, cy, cx, pipeline.hybrid_mean, pipeline.hybrid_std,
                wave, pipeline.fusion_model, pipeline.hybrid_model, pipeline.device)
            x.append(feat)
            interp_all.append(interp)
            fusion_all.append(fusion)
        eligible = np.asarray([[a is not None, b is not None, True]
                               for a,b in zip(interp_all,fusion_all)], bool)
        if pipeline.router3_checkpoint.get('architecture') == 'spectral_binary_router_with_region_fallback_v1':
            from training.three_expert_router import predict_binary_risks
            risks = predict_binary_risks(pipeline.router3_checkpoint, np.stack(x), eligible)
        else:
            risks = predict_risks(pipeline.router3_checkpoint, np.stack(x), eligible)
        independent_decisions = np.argmin(risks, axis=1)
        if routing_cache is not None and not routing_only:
            routing_cache.save_features(
                pixels, eligible, risks, independent_decisions,
                interp_all, fusion_all, cube.shape[-1])
    if routing_only:
        # Only observed-band features and learned RankCV risks are evaluated.
        # Region reconstruction is deliberately skipped during the distributed
        # count synchronization prepass.
        return {'raw_route_counts': np.bincount(independent_decisions, minlength=3),
                'evaluated_pixels': int(len(pixels))}
    if global_minority_counts is not None and global_minority_threshold is None:
        raise ValueError('Global expert counts require a minority threshold')
    if spatial_consistency is not None and global_minority_threshold is not None:
        raise ValueError('Spatial Consistency and Global Minority cannot be combined')
    if global_minority_threshold is not None:
        decisions = rankcv_global_minority_choices(
            pixels, independent_decisions, eligible, global_minority_threshold,
            counts_override=global_minority_counts, policy=global_minority_policy,
            three_expert_min_gap=three_expert_min_gap)
    elif spatial_consistency is None:
        decisions = independent_decisions
    else:
        if spatial_consistency.get('architecture') != 'spatial_consistency_source_cv_v1':
            raise RuntimeError('Unknown spatial Router checkpoint architecture')
        decisions = spatial_consistent_router_choices(
            pixels, risks, float(spatial_consistency['strength']))

    n = len(coords)
    selected = decisions[reverse]
    route = np.where(selected == 2, 2, 0).astype(np.int32)
    prediction = np.full(n, np.nan, dtype=np.float32)
    for i in np.flatnonzero(selected != 2):
        p = int(reverse[i]); band = int(coords[i,2])
        source = interp_all[p] if selected[i] == 0 else fusion_all[p]
        prediction[i] = source[band]
    region_indices = np.flatnonzero(selected == 2)
    if region_indices.size:
        # The topology inference tiles large connected defects around the
        # requested coordinates. Reuse a previous Region result ONLY if the
        # whole requested coordinate set (and tiling settings) is identical.
        # Per-pixel Region cache reuse across different sets would be wrong.
        region_input_coords = coords if region_full_eval else coords[region_indices]
        cache_hit = (routing_cache.load_region(
            coords[region_indices], region_input_coords, region_full_eval,
            pipeline.region_batch, pipeline.verify_region_batch)
            if routing_cache is not None else None)
        if cache_hit is not None:
            prediction[region_indices], n_refined = cache_hit
        else:
            reconstructed, _ = reconstruct_final_region(
                pipeline.topology_model, pipeline.topology_checkpoint,
                pipeline.geometry_model, cube, mask, pipeline.device,
                eval_coords=region_input_coords, batch_size=pipeline.region_batch,
                verify_batch=pipeline.verify_region_batch,
                progress_desc=region_progress or 'Learned Router3 Region')
            region_lookup = {(int(y),int(xx),int(b)):float(val)
                             for b,(yx, values) in reconstructed.items()
                             for (y,xx),val in zip(yx,values)}
            for i in region_indices:
                prediction[i] = region_lookup.get(tuple(map(int,coords[i])),np.nan)
            if pipeline.router3_checkpoint.get('region_expert', 'raw') in ('refined', 'lineaware'):
                from training.three_expert_router import refine_region
                prediction[region_indices], n_refined = refine_region(
                    cube, mask, coords[region_indices], prediction[region_indices])
            else:
                n_refined = 0
            if routing_cache is not None and np.isfinite(prediction[region_indices]).all():
                routing_cache.save_region(
                    coords[region_indices], region_input_coords, region_full_eval,
                    pipeline.region_batch, pipeline.verify_region_batch,
                    prediction[region_indices], n_refined)
    else:
        n_refined = 0
    fusion_guard_audit = dict(fusion_guard_enabled=0,
                              fusion_guard_changed_pixels=0,
                              fusion_guard_changed_voxels=0,
                              fusion_guard_calibration_sha256="")
    if fusion_guard is not None:
        fusion_guard_audit = _apply_sparse_fusion_guard(
            pipeline, cube, mask, coords, pixels, reverse, decisions,
            fusion_all, prediction, fusion_guard, wave)

    if not np.isfinite(prediction).all():
        raise RuntimeError('Three-expert learned router failed to cover all evaluation voxels')

    method = np.asarray(['router3_interp','router3_fusion','router3_region'],dtype='<U20')[selected]
    records=[]
    # Records describe connected groups of *evaluated pixels* (not original V2 components).
    for which,name in enumerate(('interp','fusion','region')):
        spatial = np.zeros(mask.shape[:2],bool)
        chosen_pixels = pixels[decisions == which]
        if not len(chosen_pixels):
            continue
        spatial[chosen_pixels[:,0],chosen_pixels[:,1]] = True
        for component in connected_components_4(spatial):
            component = np.asarray(component,np.int64)
            matched = np.zeros(spatial.shape,bool)
            matched[component[:,0],component[:,1]] = True
            cidx = np.flatnonzero((selected == which) & matched[coords[:,0],coords[:,1]])
            if not len(cidx):continue
            obs = np.sum(~mask[component[:,0],component[:,1]] &
                         np.isfinite(cube[component[:,0],component[:,1]]),axis=1)
            records.append(dict(component=len(records),area=len(component),eval_voxels=len(cidx),
                stage1=2 if which==2 else 0,final=2 if which==2 else 0,
                gate_probability=float(which==2),baseline_stage1=2 if which==2 else 0,
                baseline_final=2 if which==2 else 0,spectral_safe_override=0,
                stage1_spectral_override=0,stage1_decision_reason='learned_quality_risk',
                region_decision_reason='learned_quality_risk',guard_protected=0,
                guard_reason='not_applicable',
                router_predicted_spectral_error=float(np.mean(risks[reverse[cidx],0])),
                router_predicted_deep_error=float(np.mean(risks[reverse[cidx],1])),
                router_predicted_region_error=float(np.mean(risks[reverse[cidx],2])),
                effective_component_type=name,min_observed_bands=int(obs.min()),
                max_observed_bands=int(obs.max())))

    return dict(prediction=prediction,
        spectral_prediction=np.where(route==0,prediction,np.nan),
        deep_prediction=np.full(n,np.nan,np.float32),
        spectral_available=route==0,stage1_route=route.copy(),final_route=route,
        gate_probability=(route==2).astype(np.float32),
        predicted_error=risks[reverse].astype(np.float32),component_records=records,
        spectral_fallback_components=0,hybrid_changed_voxels=int(np.sum(selected==1)),
        hybrid_fallback_voxels=0,v6_fixed_reference_prediction=None,
        support_policy_audit={},guard_policy_audit={},
        candidate_method=method,
        candidate_method_voxels={key:int(np.sum(selected==i)) for i,key in enumerate(
            ('router3_interp','router3_fusion','router3_region'))},
        candidate_policy_audit=dict(
            lazy_cache_features_hit=int(bool(routing_cache and routing_cache.feature_hit)),
            lazy_cache_region_hit=int(bool(routing_cache and routing_cache.region_hit)),
            lazy_cache_region_used=int(bool(routing_cache and routing_cache.region_used)),
            learned_router3=1,learned_three_expert_decisions=int(len(pixels)),
            router3_interp_pixels=int(np.sum(decisions==0)),
            router3_fusion_pixels=int(np.sum(decisions==1)),
            router3_region_pixels=int(np.sum(decisions==2)),
            router3_interp_voxels=int(np.sum(selected==0)),
            router3_fusion_voxels=int(np.sum(selected==1)),
            router3_region_voxels=int(np.sum(selected==2)),
            router3_region_refined_voxels=int(n_refined),
            spatial_consistency_enabled=int(spatial_consistency is not None),
            spatial_consistency_strength=(float(spatial_consistency['strength'])
                                          if spatial_consistency is not None else 0.0),
            spatial_consistency_changed_pixels=(int(np.count_nonzero(
                decisions != independent_decisions)) if spatial_consistency is not None else 0),
            global_minority_enabled=int(global_minority_threshold is not None),
            global_minority_policy=(global_minority_policy if global_minority_threshold is not None else 'off'),
            global_minority_threshold_percent=(float(global_minority_threshold)*100.0
                                               if global_minority_threshold is not None else 0.0),
            global_minority_three_expert_gap_percent=(float(three_expert_min_gap)*100.0
                                                     if global_minority_policy == 'hybrid_gap'
                                                     and global_minority_threshold is not None else 0.0),
            global_minority_changed_pixels=int(np.count_nonzero(
                decisions != independent_decisions)) if global_minority_threshold is not None else 0,
            **fusion_guard_audit),
        routing_audit={'additional_region_voxels':0})
