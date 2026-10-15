"""TPSSR final two-Expert runtime: Hybrid V2 + Region, V2.2/V2.3 policy.

No Deep-v3.1 or learned router is imported or instantiated.
Region weights and Hybrid V2 weights are unchanged.
V2.3 is the controlled routing/local-median postprocessing of the V2.2
prediction, using only originally observable data.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import config
from src.component_routing_utils import connected_components_4
from src.evaluation_utils import regression_metrics
from src.final_tpssr_runtime import (
    load_models as load_final_region_models,
    reconstruct_mask as reconstruct_final_region,
    torch_load,
)

@dataclass
class FinalPipeline:
    device: torch.device
    deep_model: object
    deep_checkpoint: dict
    deep_path: Path
    router_model: object
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
    routing_mode: str = "hybrid_v2_region_2expert_v23"
    hybrid_model: torch.nn.Module | None = None
    hybrid_mean: np.ndarray | None = None
    hybrid_std: np.ndarray | None = None
    hybrid_wave_norm: np.ndarray | None = None
    hybrid_gate_checkpoint_path: Path | None = None
    hybrid_backbone_checkpoint_path: Path | None = None
    hybrid_batch: int = 32

def _load_accepted_spectral_hybrid_v2(device):
    """Load exactly the validated BAND gate plus its SHA256-pinned V1 backbone."""
    # These are the same classes that produced the benchmarked 200-mask V2
    # ablation; no reimplementation or silent substitution of weights.
    from train_spectral_hybrid_gate_v2 import FrozenHybridGated, TRAIN_CORES, VALID_CORES, sha

    parent = Path(config.MODELS_DIR) / 'texture_restoration_router_rf_v2'
    gate_path = parent / 'spectral_hybrid_gate_v2' / 'band' / 'best_model.pt'
    if not gate_path.is_file():
        raise FileNotFoundError('Accepted Hybrid V2 band checkpoint missing: ' + str(gate_path))
    gate = torch_load(gate_path, 'cpu')
    if gate.get('architecture') != 'spectral_only_hybrid_confidence_gate_v2' or gate.get('gate_variant') != 'band':
        raise RuntimeError('Expected accepted BAND-wise Hybrid V2 checkpoint.')
    if tuple(gate['run_config']['train_cores']) != tuple(TRAIN_CORES) or tuple(gate['run_config']['val_cores']) != tuple(VALID_CORES):
        raise RuntimeError('Hybrid checkpoint train/validation split mismatch.')
    # Validate both the V1 checkpoint and the resulting model wavelengths.
    base_guess = parent / 'spectral_architecture_compare_v1' / 'hybrid' / 'diagnostic_best_val.pt'
    candidates = [Path(gate.get('backbone_checkpoint') or base_guess), base_guess,
                  base_guess.with_name('best_model.pt')]
    backbone_path = next((p for p in candidates if p.is_file() and sha(p) == gate['backbone_sha256']), None)
    if backbone_path is None:
        raise FileNotFoundError('Frozen V1 backbone checkpoint missing or SHA256 mismatch: ' + str(base_guess))
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
    for ck in (source,gate):
        if not np.allclose(np.asarray(ck['wavelengths_nm'],np.float32), waves, rtol=0, atol=.5):
            raise RuntimeError('Hybrid backbone/gate wavelength grid differs from test cube.')
    avg = np.asarray(source['band_mean'],np.float32)
    sd = np.asarray(source['band_std'],np.float32)
    if avg.shape != waves.shape or sd.shape != waves.shape or not np.isfinite(avg).all() or not np.isfinite(sd).all() or (sd<=0).any():
        raise RuntimeError('Invalid Hybrid backbone normalization statistics.')
    prm=source['model_params']
    model=FrozenHybridGated(len(waves),int(prm['width']),int(prm['heads']),int(prm['layers']),variant='band')
    model.base.load_state_dict(source['model_state_dict'],strict=True)
    model.gate.load_state_dict(gate['gate_state_dict'],strict=True)
    model=model.to(device).eval()
    wn=(waves-waves[0])/max(float(waves[-1]-waves[0]),1e-6)
    return model,avg,sd,wn,gate_path,backbone_path

@torch.inference_mode()
def _hybrid_replace_spectral_only(pipeline, test_cube, mask, spectral_coords):
    """Predict only V6 final route==Spectral voxels; NEVER use masked truths."""
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

def load_pipeline(device, deep_batch=16, region_batch=32,
                  verify_region_batch=True, routing_mode='hybrid_v2_region_2expert_v23',
                  **unused_legacy_options):
    if routing_mode not in ('hybrid_v2_region_2expert_v22',
                            'hybrid_v2_region_2expert_v23'):
        raise ValueError('Clean release supports only final V2.3 or V2.2 reference.')
    hybrid,mean,std,wave,gate_path,backbone_path = _load_accepted_spectral_hybrid_v2(device)
    topology,topology_ck,v27,_v27_ck,topology_path,v27_path = load_final_region_models(device)
    return FinalPipeline(
        device=device, deep_model=None,deep_checkpoint={},
        deep_path=Path('not_used_2expert'),router_model=None,
        router_checkpoint={},router_path=Path('not_used_2expert'),
        gate_threshold=float('nan'),topology_model=topology,
        topology_checkpoint=topology_ck,topology_path=Path(topology_path),
        v27_model=v27,v27_path=Path(v27_path),deep_batch=int(deep_batch),
        region_batch=int(region_batch),verify_region_batch=bool(verify_region_batch),
        routing_mode=routing_mode,hybrid_model=hybrid,hybrid_mean=mean,
        hybrid_std=std,hybrid_wave_norm=wave,
        hybrid_gate_checkpoint_path=gate_path,
        hybrid_backbone_checkpoint_path=backbone_path)

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
def _predict_mask_hybrid_region_2expert(
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
    v21 = pipeline.routing_mode == "hybrid_v2_region_2expert_v21"
    v22 = pipeline.routing_mode == "hybrid_v2_region_2expert_v22"
    gap_policy = v21 or v22
    spatial_mask = np.any(mask, axis=2)
    observed_counts = np.sum((~mask) & np.isfinite(test_cube), axis=2, dtype=np.int32)
    if gap_policy:
        if v22:
            from src.spectral_route_policy_v22 import expert_eligibility_maps
        else:
            from src.spectral_route_policy_v21 import expert_eligibility_maps
        can_hybrid, no_anchor_region, gap_region, policy_audit = (
            expert_eligibility_maps(mask, test_cube)
        )
        needs_region = no_anchor_region | gap_region
        eligibility_classes = (
            (0, can_hybrid, "hybrid_spectral_information_sufficient"),
            (2, no_anchor_region, "region_lt_2_observed_bands"),
            (2, gap_region, "region_long_gap_with_local_support_and_true_mask_purity" if v22 else "region_long_gap_with_local_observed_neighbors"),
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
            "stage1_decision_reason": "v22_true_mask_gap_purity_guard" if v22 else "v21_spectral_gap_spatial_support" if v21 else "observed_band_availability_only",
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
            pipeline.v27_model,
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
        raise AssertionError("Deep-v3.1 was selected in Hybrid/Region ablation.")

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
        "v21_policy_audit": policy_audit,
        "v21_region_gap_eval_voxels": int(np.sum(gap_region[coords[:, 0], coords[:, 1]])) if gap_policy else 0,
        "v22_policy_audit": policy_audit if v22 else {},
        "v22_region_gap_eval_voxels": int(np.sum(gap_region[coords[:, 0], coords[:, 1]])) if v22 else 0,
    }


def _v23_component_rows(mask,cube,coords,extra):
    """Regenerate consistent component audit after V2.3 extra-region routes."""
    from src.spectral_route_policy_v22 import expert_eligibility_maps
    hybrid,_,region,_=expert_eligibility_maps(mask,cube)
    new_hybrid=hybrid & (~extra)
    new_region=(region | extra) | (np.any(mask,axis=2) & (~hybrid) & (~region))
    observed=np.sum((~mask) & np.isfinite(cube),axis=2,dtype=np.int32)
    rows=[];cid=0
    for final,eligible,reason in ((0,new_hybrid,'v23_hybrid_observed_spectrum'),
                                  (2,new_region,'v23_region_gap_or_no_anchors')):
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
                stage1_spectral_override=0,stage1_decision_reason='v23_mask_geometry',
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
    if pipeline.routing_mode == 'hybrid_v2_region_2expert_v22':
        return _predict_mask_hybrid_region_2expert(
            pipeline,test_cube,mask,eval_coords,region_progress=region_progress)
    if pipeline.routing_mode != 'hybrid_v2_region_2expert_v23':
        raise ValueError('Unknown two-Expert mode.')
    from src.spectral_route_policy_v23 import route_extra_region,local_neighbor_median
    from src.spectral_route_policy_v22 import expert_eligibility_maps
    coords=np.asarray(eval_coords,np.int64)
    mask=np.asarray(mask,bool)
    # Original V2.2 is computed with precisely the same frozen weights and
    # original V2.2 decisions. This is required for controlled parity.
    orig_mode=pipeline.routing_mode
    try:
        pipeline.routing_mode='hybrid_v2_region_2expert_v22'
        out=_predict_mask_hybrid_region_2expert(
            pipeline,test_cube,mask,coords,region_progress=region_progress)
    finally:
        pipeline.routing_mode=orig_mode
    base=np.asarray(out['prediction'],np.float32)
    old_route=np.asarray(out['final_route'],np.int64)
    eligible_hybrid,_,_,_=expert_eligibility_maps(mask,test_cube)
    extra,stats=route_extra_region(test_cube,mask,eligible_hybrid,coords)
    add=(old_route==0) & extra[coords[:,0],coords[:,1]]
    new=base.copy();new_route=old_route.copy()
    if np.any(add):
        rebuilt,_=reconstruct_final_region(
            pipeline.topology_model,pipeline.topology_checkpoint,
            pipeline.v27_model,test_cube,mask,pipeline.device,
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
    out['component_records']=_v23_component_rows(mask,test_cube,coords,extra)
    out['v23_policy_audit']=dict(**stats,**median_stats,
               additional_region_voxels=int(add.sum()),
               local_median_voxels=int(use.sum()))
    out['v23_baseline_v22_prediction']=base
    return out
