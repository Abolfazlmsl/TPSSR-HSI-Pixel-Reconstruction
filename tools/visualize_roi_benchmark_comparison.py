#!/usr/bin/env python3
"""Reconstruct one manually selected HSI region with the final TPSSR and benchmarks.

Selection is delegated to visualize_tpssr_roi_with_region_maps.py, so the
128x128 crop and draggable defect window remain identical to that tool.
Every method predicts the same target coordinates. Its full reconstructed cube
is saved before plotting. --render-only never loads models or source datasets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CORE = ROOT / "core"
VISUALIZER = Path(__file__).resolve().parent / "visualize_tpssr_roi_with_region_maps.py"
# Final paper methods: RF uses the trained multiscale spatial-spectral model.
DEFAULT_METHODS = ("tpssr", "pfgin", "random_forest_multiscale", "pca",
                   "spatial", "spatio_spectral", "spectral", "dds2m")
SUPPORTED = ("tpssr", "dds2m", "pfgin", "random_forest_multiscale",
             "pca", "spatial", "spectral", "spatio_spectral")
DISPLAY_NAMES = {
    "tpssr": "TPSSR", "dds2m": "DDS2M", "pfgin": "PFGIN",
    "random_forest_multiscale": "RF", "pca": "PCA",
    "spatial": "Spatial", "spectral": "Spectral",
    "spatio_spectral": "Spatio-Spectral",
}
SCENARIOS = (
    "isolated_voxels", "single_detector_line", "multiple_detector_lines",
    "partial_vertical_lines", "spatial_patch", "consecutive_spectral_bands",
    "spatio_spectral_block", "whole_spectrum_pixels", "band_wide_saturation",
    "mixed_saturation",
)
WHOLE_SPECTRUM_SIZES = (1, 3, 5, 9, 15)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description="Manual ROI reconstruction and reusable paper figures")
    selection = parser.add_argument_group("Select the same ROI/mask as the existing visualizer")
    selection.add_argument("--preset", default="swir_8x2", choices=("swir_8x2", "swir_region_a_to_b", "swir_region_b_to_a", "vnir_8x2"))
    selection.add_argument("--dataset", default=None,
                           help="Held-out acquisition. Defaults to SWIR5/VNIR5/SWIR6/SWIR1 by preset")
    selection.add_argument("--router", choices=("learned_router3_aligned", "learned_router3_lineaware_rankcv",
                                               "learned_router3_lineaware"), default=None,
                           help="Trained Router checkpoint; default is RankCV for SWIR and aligned for VNIR")
    spatial = selection.add_mutually_exclusive_group()
    spatial.add_argument("--spatial-consistency", dest="spatial_consistency", action="store_true",
                         help="Use the source-calibrated RankCV spatial Router for this preset")
    spatial.add_argument("--no-spatial-consistency", dest="spatial_consistency", action="store_false",
                         help="Use the original RankCV Router without spatial consistency")
    selection.set_defaults(spatial_consistency=None)
    selection.add_argument("--global-minority-percent", type=float, default=10.0,
                           help="Final two-expert minority threshold, percent (default: 10)")
    selection.add_argument("--minority-policy", choices=("threshold", "hybrid", "hybrid_gap"), default="hybrid_gap",
                           help="Final hybrid-gap policy; legacy hybrid and threshold remain available")
    selection.add_argument("--three-expert-min-gap-percent", type=float, default=10.0,
                           help="Protect close three-expert choices (percentage-point gap; default 10)")
    selection.add_argument("--fusion-guard", action="store_true",
                           help="Enable locked source-calibrated Fusion correction for TPSSR")
    flip = selection.add_mutually_exclusive_group()
    flip.add_argument("--flip-x", dest="flip_x", action="store_true",
                      help="Mirror figures horizontally; default on for VNIR (display only)")
    flip.add_argument("--no-flip-x", dest="flip_x", action="store_false",
                      help="Keep original horizontal display orientation")
    selection.set_defaults(flip_x=None)
    selection.add_argument("--image-scope", default="full", choices=("auto", "full", "test-block"),
                           help="full (default): show the ENTIRE HSI for ROI selection; test-block: last held-out block; auto: legacy behavior")
    selection.add_argument("--scenario", default="whole_spectrum_pixels", choices=SCENARIOS,
                           help="One scenario (all 10 benchmark scenario names are supported)")
    selection.add_argument("--band-wide-mode", choices=("local", "full-roi"), default="local",
                           help="Band-wide Saturation only: local legacy example or delete full bands "
                                "across the 128x128 input ROI")
    selection.add_argument("--band-wide-eval", choices=("benchmark", "all"), default=None,
                           help="With full-roi band-wide: all valid masked ROI voxels (default), or "
                                "benchmark (up to 5000 spatial pixels). Local mode keeps benchmark semantics")
    selection.add_argument("--band-wide-band-selection", choices=("display", "official", "max-curvature"),
                           default="display", help="Choose missing-band center: display / original benchmark repeat / highest curvature")
    selection.add_argument("--band-wide-repeat", type=int, choices=range(1, 5), default=1,
                           help="Original 200-mask benchmark repetition for band_wide_saturation (r01..r04)")
    selection.add_argument("--list-scenarios", action="store_true",
                           help="Print the supported scenario names and exit")
    selection.add_argument("--severity", type=int, default=5, choices=range(1, 6))
    selection.add_argument("--whole-spectrum-size", type=int, default=None)
    selection.add_argument("--roi-y", type=int, default=None)
    selection.add_argument("--roi-x", type=int, default=None)
    selection.add_argument("--missing-y", type=int, default=None)
    selection.add_argument("--missing-x", type=int, default=None)
    selection.add_argument("--seed", type=int, default=4207)
    selection.add_argument("--preview-max-height", type=int, default=1300)
    selection.add_argument("--preview-max-width", type=int, default=1000)
    compute = parser.add_argument_group("Reconstruction")
    compute.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                         help="Comma-separated methods or all. Available: " + ", ".join(SUPPORTED))
    compute.add_argument("--gpus", type=int, choices=(1, 2), default=2,
                         help="TPSSR GPUs (default 2)")
    compute.add_argument("--dds2m-gpus", default="0,1", help="Physical GPU IDs for DDS2M (default 0,1)")
    compute.add_argument("--pfgin-gpu", type=int, default=0)
    compute.add_argument("--region-batch", type=int, default=16)
    compute.add_argument("--outdir", default=None,
                         help="Output folder; reuse the same folder to resume incomplete methods")
    compute.add_argument("--from-outdir", type=Path, default=None,
                         help="Reuse a saved ROI in a NEW output folder; when upgrading band-wide "
                              "benchmark to all, preserve the mask but recompute ALL method predictions")
    compute.add_argument("--select-only", action="store_true",
                         help="Save the selected ROI/mask for later reconstruction")
    compute.add_argument("--rerun", default="", help="Comma-separated methods to intentionally recompute")
    view = parser.add_argument_group("Display; can be changed without reconstruction")
    view.add_argument("--render-only", type=Path, default=None,
                      help="Regenerate figures from a saved output folder; does not load a model")
    view.add_argument("--band", type=int, default=None, help="Zero-based wavelength band; middle band by default")
    view.add_argument("--gamma", type=float, default=1.0)
    view.add_argument("--stretch", type=float, default=1.0,
                      help="Display-only contrast stretch about midpoint (1.0 = none)")
    view.add_argument("--cmap", default="gray")
    view.add_argument("--dpi", type=int, default=350)
    view.add_argument("--crop", default=None, help="Display crop in ROI pixels: y0:y1,x0:x1")
    view.add_argument("--show-errors", action="store_true", help="Also export comparable absolute-error maps")
    return parser.parse_args(argv)


def print_scenarios():
    print("Available scenarios (S1..S5):")
    for name in SCENARIOS:
        print("  " + name)
    print("\nRun one scenario per command with --scenario and --severity.")
    print("All results are saved under roi_comparison/ by default.")
    print("These are local demonstrative masks, not the standard 200 evaluation masks.")


def parse_methods(raw):
    if raw.strip().lower() == "all":
        return list(SUPPORTED)
    names = [x.strip().lower().replace("-", "_") for x in raw.split(",") if x.strip()]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate method in --methods")
    unknown = set(names) - set(SUPPORTED)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}; supported={SUPPORTED}")
    return names


def resolve_display_options(args):
    if args.dataset is None:
        args.dataset = {"swir_8x2": "SWIR5", "vnir_8x2": "VNIR5",
                        "swir_region_a_to_b": "SWIR6", "swir_region_b_to_a": "SWIR1"}[args.preset]
    # An illustrative full-band ROI should be fully reconstructed and fully
    # scored unless benchmark sub-sampling is explicitly requested.
    if args.band_wide_eval is None:
        args.band_wide_eval = ("all" if args.scenario == "band_wide_saturation"
                               and args.band_wide_mode == "full-roi" else "benchmark")
    if args.scenario != "band_wide_saturation" and args.band_wide_mode != "local":
        raise ValueError("--band-wide-mode full-roi requires --scenario band_wide_saturation")
    if args.scenario != "band_wide_saturation" and args.band_wide_eval != "benchmark":
        raise ValueError("--band-wide-eval requires --scenario band_wide_saturation")
    if args.band_wide_mode == "local" and args.band_wide_eval != "benchmark":
        raise ValueError("--band-wide-eval all requires --band-wide-mode full-roi")
    if args.band_wide_band_selection != "display":
        if args.scenario != "band_wide_saturation" or args.band_wide_mode != "full-roi":
            raise ValueError("Official/max-curvature band selection requires --band-wide-mode full-roi")
        if args.band is not None:
            raise ValueError("Omit --band with difficulty-based selection; the selected mask center is displayed automatically")
    # VNIR presentation is mirrored for visual alignment; data arrays and model
    # pixel coordinates remain in the ORIGINAL sensor coordinate system.
    if args.router is None:
        args.router = "learned_router3_lineaware_rankcv"
    if args.spatial_consistency is None:
        args.spatial_consistency = False
    if not 0 < args.global_minority_percent < 50:
        raise ValueError("--global-minority-percent must be strictly between 0 and 50")
    if not 0 <= args.three_expert_min_gap_percent < 100:
        raise ValueError("--three-expert-min-gap-percent must be between 0 and 100")
    if args.global_minority_percent is not None:
        if args.router != "learned_router3_lineaware_rankcv":
            raise ValueError("Global Minority requires RankCV Line-aware Router")
        if args.spatial_consistency or args.fusion_guard:
            raise ValueError("Global Minority cannot be combined with Spatial Consistency or Fusion Guard")
    if args.spatial_consistency and args.router != "learned_router3_lineaware_rankcv":
        raise ValueError("--spatial-consistency requires the RankCV Router")
    if args.fusion_guard and (args.preset != "swir_8x2"
                              or args.router != "learned_router3_lineaware_rankcv"
                              or not args.spatial_consistency):
        raise ValueError("--fusion-guard requires SWIR 8x2 RankCV with spatial consistency")
    if args.flip_x is None:
        args.flip_x = args.dataset.upper().startswith("VNIR")
    if args.dataset.upper().startswith("VNIR") != args.preset.startswith("vnir_"):
        raise ValueError("Dataset and model preset must use the same sensor (VNIR/SWIR)")


def output_folder(args):
    if args.outdir:
        return Path(args.outdir).expanduser().resolve()
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    bw_tag = (f"_{args.band_wide_mode}_{args.band_wide_eval}_{args.band_wide_band_selection}"
              f"_r{args.band_wide_repeat:02d}" if args.scenario == "band_wide_saturation" else "")
    return (ROOT / "roi_comparison" / args.dataset.upper() /
            f"{args.scenario}_S{args.severity}{bw_tag}__minority{args.minority_policy}{args.global_minority_percent:g}pct"
            f"__gap{args.three_expert_min_gap_percent:g}pp__robustdonor_{tag}").resolve()


def select_roi(args, out):
    """Re-use existing interactive ROI selection and corruption-mask generation."""
    if (out / "selection.json").exists():
        meta = json.loads((out / "selection.json").read_text(encoding="utf-8"))
        if (args.scenario == "band_wide_saturation" and
                (meta.get("band_wide_mode", "local") != args.band_wide_mode or
                 meta.get("band_wide_eval", "benchmark") != args.band_wide_eval or
                 meta.get("band_wide_band_selection", "display") != args.band_wide_band_selection or
                 int(meta.get("band_wide_repeat", 1)) != args.band_wide_repeat)):
            raise ValueError("Saved Band-wide mode/evaluation selection differs. Use another --outdir.")
        if meta.get("scenario") != args.scenario or int(meta.get("severity", -1)) != args.severity:
            raise ValueError(f"Saved selection is {meta.get('scenario')} S{meta.get('severity')}, "
                             f"but requested {args.scenario} S{args.severity}. "
                             "Choose a new --outdir or use the original scenario/severity.")
        if meta.get("dataset", "").upper() != args.dataset.upper() or meta.get("preset") != args.preset:
            raise ValueError("Saved selection dataset/preset do not match requested settings")
        if meta.get("router") != args.router:
            raise ValueError("Saved selection uses a different Router. Use the original --router to resume.")
        if bool(meta.get("spatial_consistency", False)) != args.spatial_consistency:
            raise ValueError("Saved ROI was created with a different spatial Router setting. "
                             "Use a new output folder to prevent mixing predictions.")
        if (meta.get("global_minority_percent") != args.global_minority_percent or
                meta.get("global_minority_policy", "threshold") != args.minority_policy or
                (args.minority_policy == 'hybrid_gap' and
                 float(meta.get('three_expert_min_gap_percent', -1)) != args.three_expert_min_gap_percent)):
            raise ValueError("Saved ROI uses a different Global Minority policy/threshold; "
                             "use --from-outdir with a NEW output folder to reuse its mask.")
        if meta.get("fusion_donor_match") != "robust":
            raise ValueError("Saved ROI was created without the final Robust Donor setting")
        if bool(meta.get("fusion_guard", False)) != bool(args.fusion_guard):
            raise ValueError("Saved ROI has different Fusion Guard setting; use a new output folder")
        if int(meta.get("seed", -1)) != args.seed:
            raise ValueError("Saved selection seed differs; use the original --seed to resume")
        if args.whole_spectrum_size is not None and args.scenario == "whole_spectrum_pixels":
            if int(meta.get("whole_spectrum_size", -1)) != args.whole_spectrum_size:
                raise ValueError("Saved whole-spectrum mask size differs")
        print(f"[RESUME] Using the saved selection in {out}", flush=True)
        return
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Folder exists but no selection.json was found: {out}")
    command = [sys.executable, str(VISUALIZER),
               "--preset", args.preset, "--dataset", args.dataset,
               "--image-scope", args.image_scope, "--scenario", args.scenario,
               "--severity", str(args.severity), "--seed", str(args.seed),
               "--band-wide-mode", args.band_wide_mode,
               "--band-wide-eval", args.band_wide_eval,
               "--band-wide-band-selection", args.band_wide_band_selection,
               "--band-wide-repeat", str(args.band_wide_repeat),
               "--router", args.router,
               *(["--spatial-consistency"] if args.spatial_consistency else []),
               *(["--fusion-guard"] if args.fusion_guard else []),
               "--global-minority-percent", str(args.global_minority_percent),
               "--minority-policy", args.minority_policy,
               "--three-expert-min-gap-percent", str(args.three_expert_min_gap_percent),
               "--flip-x" if args.flip_x else "--no-flip-x",
               "--outdir", str(out), "--no-region-figure", "--select-only",
               "--gpus", str(args.gpus),
               "--preview-max-height", str(args.preview_max_height),
               "--preview-max-width", str(args.preview_max_width)]
    for flag, value in (("--whole-spectrum-size", args.whole_spectrum_size),
                        ("--roi-y", args.roi_y), ("--roi-x", args.roi_x),
                        ("--missing-y", args.missing_y), ("--missing-x", args.missing_x),
                        ("--band", args.band)):
        if value is not None:
            command += [flag, str(value)]
    print(f"[SELECTION] Viewing {args.image_scope} image before selecting the 128x128 ROI using the existing visualizer", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)



def clone_roi_from_previous(source, out, args):
    """Reuse a saved ROI. A sampled-to-full upgrade re-runs every method."""
    import shutil
    source = source.expanduser().resolve()
    if source == out:
        raise ValueError("--from-outdir must differ from the new output folder")
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Destination already contains files: {out}")
    saved = source / "selection.json"
    if not saved.is_file():
        raise FileNotFoundError(f"Old ROI selection was not found: {saved}")
    old_meta = json.loads(saved.read_text(encoding="utf-8"))
    if (str(old_meta.get("dataset", "")).upper() != args.dataset.upper()
            or old_meta.get("preset") != args.preset
            or old_meta.get("scenario") != args.scenario
            or int(old_meta.get("severity", -1)) != args.severity
            or int(old_meta.get("seed", -1)) != args.seed):
        raise ValueError("Old ROI dataset/preset/scenario/severity/seed differ from requested run")

    old_eval = old_meta.get("band_wide_eval", "benchmark")
    old_mode = old_meta.get("band_wide_mode", "local")
    upgrading = (args.scenario == "band_wide_saturation"
                 and old_mode == "full-roi"
                 and old_eval == "benchmark"
                 and args.band_wide_eval == "all")
    if args.scenario == "band_wide_saturation":
        if (old_mode != args.band_wide_mode
                or old_meta.get("band_wide_band_selection", "display") != args.band_wide_band_selection
                or int(old_meta.get("band_wide_repeat", 1)) != args.band_wide_repeat
                or (old_eval != args.band_wide_eval and not upgrading)):
            raise ValueError("Saved band-wide mask selection does not match the requested run")

    required = ("roi_cube.npy", "roi_invalid.npy", "synthetic_mask.npy",
                "model_mask.npy", "coords.npy")
    for name in required:
        if not (source / "_work" / name).is_file():
            raise FileNotFoundError(f"Cannot reuse the old ROI; missing {source / '_work' / name}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "_work").mkdir(exist_ok=True)
    for name in required:
        if upgrading and name == "coords.npy":
            continue
        shutil.copy2(source / "_work" / name, out / "_work" / name)

    if upgrading:
        gt = np.load(out / "_work" / "roi_cube.npy", allow_pickle=False)
        invalid = np.load(out / "_work" / "roi_invalid.npy", allow_pickle=False)
        synthetic = np.load(out / "_work" / "synthetic_mask.npy", allow_pickle=False)
        mask = np.load(out / "_work" / "model_mask.npy", allow_pickle=False)
        if not (gt.shape == invalid.shape == synthetic.shape == mask.shape):
            raise ValueError("Saved ROI arrays have incompatible shapes")
        if not np.array_equal(mask, synthetic | invalid | ~np.isfinite(gt)):
            raise RuntimeError("Saved full-band mask was changed or is inconsistent")
        all_coords = np.argwhere(synthetic & ~invalid & np.isfinite(gt)).astype(np.int32)
        if not len(all_coords):
            raise RuntimeError("The saved ROI contains no valid missing target voxels")
        np.save(out / "_work" / "coords.npy", all_coords)
        old_meta["band_wide_eval"] = "all"
        old_meta["evaluated_target_voxels"] = int(len(all_coords))
        old_meta["evaluated_spatial_pixels"] = int(len(np.unique(all_coords[:, :2], axis=0)))
        old_meta["full_mask_valid_target_voxels"] = int(len(all_coords))
        print(f"[FULL ROI UPGRADE] Preserved all missing bands and ROI; "
              f"now evaluating all {len(all_coords):,} valid masked voxels", flush=True)

    for name in ("roi_ground_truth.npy", "roi_invalid_mask.npy", "roi_synthetic_mask.npy",
                 "roi_model_mask.npy", "roi_corrupted.npy", "wavelengths_nm.npy"):
        if (source / name).is_file():
            shutil.copy2(source / name, out / name)
    if upgrading:
        # Critical: the old predictions covered only sampled coordinates.
        # Reusing them would corrupt full-ROI RMSE and displayed images.
        np.save(out / "roi_coords.npy", all_coords)
        print("[FULL ROI UPGRADE] No old method predictions or metrics were reused", flush=True)
    else:
        if (source / "roi_coords.npy").is_file():
            shutil.copy2(source / "roi_coords.npy", out / "roi_coords.npy")
        trusted_rf = (source / "rf_checkpoint_manifest.json").is_file()
        for method in SUPPORTED:
            if method == "tpssr":
                continue
            if method == "random_forest_multiscale" and not trusted_rf:
                continue
            for name in (f"roi_reconstructed_{method}.npz", f"metrics_{method}.json"):
                if (source / name).is_file():
                    shutil.copy2(source / name, out / name)
        if trusted_rf:
            shutil.copy2(source / "rf_checkpoint_manifest.json", out / "rf_checkpoint_manifest.json")

    old_meta["router"] = args.router
    old_meta["spatial_consistency"] = bool(args.spatial_consistency)
    old_meta["global_minority_percent"] = float(args.global_minority_percent)
    old_meta["global_minority_policy"] = args.minority_policy
    old_meta["three_expert_min_gap_percent"] = args.three_expert_min_gap_percent
    old_meta["fusion_donor_match"] = "robust"
    old_meta["fusion_guard"] = bool(args.fusion_guard)
    old_meta["derived_from_roi"] = str(source)
    (out / "selection.json").write_text(json.dumps(old_meta, indent=2), encoding="utf-8")
    print(f"[REUSE ROI] Selected area and mask copied from {source}", flush=True)

def load_selection(out):
    meta = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    work = out / "_work"
    files = [work / f for f in ("roi_cube.npy", "roi_invalid.npy", "synthetic_mask.npy", "model_mask.npy", "coords.npy")]
    if not all(p.is_file() for p in files):
        raise RuntimeError(f"ROI working arrays missing in {work}. These are required for inference.")
    gt, invalid, synthetic, mask, coords = (np.load(p, allow_pickle=False) for p in files)
    if gt.ndim != 3 or invalid.shape != gt.shape or synthetic.shape != gt.shape or mask.shape != gt.shape:
        raise RuntimeError("Incompatible ROI array shapes")
    if coords.ndim != 2 or coords.shape[1] != 3 or not len(coords) or not np.issubdtype(coords.dtype, np.integer):
        raise RuntimeError("Invalid saved target coordinates")
    if np.any(coords < 0) or np.any(coords >= np.asarray(gt.shape, dtype=coords.dtype)):
        raise RuntimeError("Saved target coordinates are outside the ROI cube")
    # The model must see the COMPLETE synthetic defect, even when only a sampled
    # subset of its voxels is scored. Do not replace the input mask with the
    # evaluation-coordinate mask: that would leak withheld observations.
    expected_mask = synthetic | invalid | ~np.isfinite(gt)
    if not np.array_equal(mask, expected_mask):
        raise RuntimeError("Saved input mask differs from the complete synthetic/invalid mask")
    valid_target = synthetic & ~invalid & np.isfinite(gt)
    flat = np.ravel_multi_index(coords.T, gt.shape)
    if np.unique(flat).size != len(coords):
        raise RuntimeError("Duplicate target coordinates in the saved ROI")
    if not np.all(valid_target[coords[:, 0], coords[:, 1], coords[:, 2]]):
        raise RuntimeError("Saved evaluation coordinates include unmasked or invalid voxels")
    total_valid = int(np.count_nonzero(valid_target))
    sampled_bandwide = (meta.get("scenario") == "band_wide_saturation" and
                        meta.get("band_wide_mode") == "full-roi" and
                        meta.get("band_wide_eval") == "benchmark")
    if sampled_bandwide:
        if len(coords) > total_valid:
            raise RuntimeError("More evaluation coordinates than valid masked voxels")
        print(f"[SELECTION AUDIT] {total_valid:,} valid masked voxels; "
              f"{len(coords):,} sampled evaluation voxels; full defect preserved", flush=True)
    elif len(coords) != total_valid:
        raise RuntimeError("Saved coordinates do not cover all valid target voxels")
    return meta, gt, invalid, synthetic, mask, coords


def persist_reference(out, gt, invalid, synthetic, mask, coords, meta):
    """Keep all arrays required for rendering even when the source data move."""
    for name, array in (("roi_ground_truth.npy", gt), ("roi_invalid_mask.npy", invalid),
                        ("roi_synthetic_mask.npy", synthetic), ("roi_model_mask.npy", mask),
                        ("roi_coords.npy", coords)):
        path = out / name
        if not path.exists():
            np.save(path, array)
    corrupted_file = out / "roi_corrupted.npy"
    if not corrupted_file.exists():
        corrupted = gt.copy()
        corrupted[mask] = np.nan
        np.save(corrupted_file, corrupted)
    if not (out / "wavelengths_nm.npy").exists():
        # Recover real wavelengths using the same dataset/config selection as the model.
        from experiment_config import resolve_experiment
        from _launcher_common import env_for
        import argparse as _argparse
        spec = resolve_experiment(_argparse.Namespace(preset=meta["preset"], dataset=None,
                           active_test=None, train_datasets=None, test_datasets=None,
                           experiment_name=None))
        os.environ.update(env_for(spec, spec.active_test))
        if str(CORE) not in sys.path:
            sys.path.insert(0, str(CORE))
        from src.evaluation_utils import load_dataset
        _, _, _, waves = load_dataset(meta["dataset"])
        np.save(out / "wavelengths_nm.npy", np.asarray(waves, np.float32))


def configure_model_environment(meta):
    from experiment_config import resolve_experiment
    from _launcher_common import env_for
    namespace = argparse.Namespace(preset=meta["preset"], dataset=None, active_test=None,
                                  train_datasets=None, test_datasets=None, experiment_name=None)
    spec = resolve_experiment(namespace)
    os.environ.update(env_for(spec, spec.active_test))
    if str(CORE) not in sys.path:
        sys.path.insert(0, str(CORE))
    import config
    return config



def tpssr_checkpoint_signature(config, args):
    """Identify the trained models actually used, preventing stale ROI comparisons."""
    from training.neighbor_fusion import checkpoint_path
    from training.three_expert_router import path_for
    from training.spectral_gate import sha
    models = Path(config.MODELS_DIR)
    if args.router == "learned_router3_lineaware_rankcv":
        router = path_for(models, objective="rank_cv", region_expert="lineaware")
    elif args.router == "learned_router3_lineaware":
        router = path_for(models, objective="log_rmse", region_expert="lineaware")
    else:
        router = path_for(models, objective="log_rmse", region_expert="refined")
    paths = {"router": router, "fusion": checkpoint_path(models)}
    if args.spatial_consistency:
        paths["spatial_consistency"] = router.parent / "spatial_consistency.joblib"
    if args.fusion_guard:
        paths["fusion_guard_calibration"] = (router.parent / "fusion_guard_calibration" /
                                             "FUSION_GUARD_CALIBRATION.json")
    checksums = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing TPSSR {name} checkpoint: {path}")
        checksums[name] = sha(path)
    return {"preset": args.preset, "router": args.router,
            "fusion_donor_match": os.environ.get("TPSSR_FUSION_DONOR_MATCH", "legacy").strip().lower(),
            "global_minority_policy": args.minority_policy,
            "global_minority_percent": float(args.global_minority_percent),
            "three_expert_min_gap_percent": args.three_expert_min_gap_percent,
            "spatial_consistency": bool(args.spatial_consistency),
            "fusion_guard": bool(args.fusion_guard),
            "checkpoint_sha256": checksums}


def method_prediction(method, args, out, gt, mask, coords, meta, config):
    if method == "tpssr":
        import torch
        detected = torch.cuda.device_count()
        if detected < args.gpus:
            raise RuntimeError(f"TPSSR requested {args.gpus} GPU(s), but {detected} visible")
        worker_cmd = [sys.executable]
        if args.gpus == 2:
            worker_cmd += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
        worker_cmd += [str(VISUALIZER), "--preset", meta["preset"],
                       "--dataset", meta["dataset"], "--scenario", meta["scenario"],
                       "--severity", str(meta["severity"]), "--outdir", str(out),
                       "--router", meta["router"],
                       "--region-batch", str(args.region_batch), "--worker", "--no-region-figure",
                       "--global-minority-percent", str(args.global_minority_percent),
                       "--minority-policy", args.minority_policy,
                       "--three-expert-min-gap-percent", str(args.three_expert_min_gap_percent)]
        if args.spatial_consistency:
            worker_cmd.append("--spatial-consistency")
        if args.fusion_guard:
            worker_cmd.append("--fusion-guard")
        # The visualizer worker uses exactly the trained three-expert pipeline.
        subprocess.run(worker_cmd, cwd=ROOT, check=True, env=os.environ.copy())
        from visualize_tpssr_roi_with_region_maps import merge_outputs
        saved_coords, values, _routes = merge_outputs(out, args.gpus)
        if not np.array_equal(saved_coords, coords):
            raise RuntimeError("TPSSR worker changed the target-coordinate order")
        np.save(out / "roi_expert_routes.npy", np.column_stack([coords, _routes]))
        return values

    if method in ("spatial", "spectral", "spatio_spectral"):
        from src.local_methods import reconstruct
        return reconstruct(gt, mask, coords, method)

    if method == "pca":
        import joblib
        path = config.MODELS_DIR / "pca" / "pca_model.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Trained PCA artifact not found: {path}")
        payload = joblib.load(path)
        from reconstruction_methods.pca.test import reconstruct_pca
        return reconstruct_pca(gt, mask, coords, payload["model"], payload["band_mean"])

    if method == "random_forest_multiscale":
        import joblib
        path = config.MODELS_DIR / "random_forest" / "random_forest_multiscale_model.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Trained multiscale RF checkpoint not found: {path}. "
                                    "Use the approved RF checkpoint for this experiment.")
        item = joblib.load(path)
        from src.enhanced_classical import predict_multiscale_rf, FEATURE_DIM, RF_VARIANT
        if (int(item.get("feature_dim", -1)) != FEATURE_DIM
                or item.get("variant") != RF_VARIANT):
            raise ValueError(f"RF checkpoint is not {RF_VARIANT}. Retrain RF first.")
        return predict_multiscale_rf(item["model"], item["band_mean"], gt, mask, coords)

    if method == "pfgin":
        import torch
        from reconstruction_methods.pfgin.test import load_model, reconstruct_mask
        if not torch.cuda.is_available() or args.pfgin_gpu >= torch.cuda.device_count():
            raise RuntimeError(f"PFGIN CUDA GPU {args.pfgin_gpu} unavailable")
        config.PFGIN_GPU_IDS = [int(args.pfgin_gpu)]
        device = torch.device(f"cuda:{args.pfgin_gpu}")
        torch.cuda.set_device(device)
        model, (lo, hi), _iteration = load_model(device)
        if len(lo) != gt.shape[2]:
            raise RuntimeError("PFGIN scaler bands do not match the ROI")
        try:
            result = reconstruct_mask(model, device, gt, mask, coords, lo, hi)
        finally:
            del model
            torch.cuda.empty_cache()
        return result

    if method == "dds2m":
        from reconstruction_methods.dds2m.test import require_official_repo, reconstruct_mask
        gpu_ids = [int(v.strip()) for v in args.dds2m_gpus.split(",") if v.strip()]
        if not gpu_ids:
            raise ValueError("--dds2m-gpus cannot be empty")
        config.DDS2M_GPU_IDS = gpu_ids
        config.DDS2M_RESUME = True
        # The full input ROI and its mask are part of the cache identity.
        signature = hashlib.sha256()
        signature.update(np.ascontiguousarray(gt).tobytes())
        signature.update(np.ascontiguousarray(mask).tobytes())
        tag = "roi_" + signature.hexdigest()[:16]
        cache_dir = out / "dds2m_tile_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return reconstruct_mask(require_official_repo(), gt, mask, coords, tag, cache_dir)
    raise ValueError(f"Unsupported method: {method}")


def save_prediction(out, method, gt, mask, coords, predictions, duration):
    pred = np.asarray(predictions, np.float32).reshape(-1)
    if pred.shape[0] != len(coords):
        raise RuntimeError(f"{method}: {len(pred)} values for {len(coords)} targets")
    recon = np.asarray(gt, np.float32).copy()
    recon[mask] = np.nan
    recon[coords[:, 0], coords[:, 1], coords[:, 2]] = pred
    # Save reconstruction BEFORE rendering, so plotting cannot lose the output.
    target = gt[coords[:, 0], coords[:, 1], coords[:, 2]]
    from src.evaluation_utils import regression_metrics, spectral_angle_metrics
    reg = regression_metrics(target, pred)
    sam = spectral_angle_metrics(gt, coords, pred)
    n = int(reg["valid_n"])
    rmse = float(reg["rmse"]) if np.isfinite(reg["rmse"]) else None
    mae = float(reg["mae"]) if np.isfinite(reg["mae"]) else None
    r2 = float(reg["r2"]) if np.isfinite(reg["r2"]) else None
    sam_deg = float(sam["sam_deg"]) if np.isfinite(sam["sam_deg"]) else None
    tmp = out / f".roi_reconstructed_{method}.npz.tmp"
    with tmp.open("wb") as f:
        np.savez_compressed(f, reconstruction=recon)
    os.replace(tmp, out / f"roi_reconstructed_{method}.npz")
    info = dict(method=method, total_target_voxels=int(len(coords)),
                predicted_voxels=n, coverage=n / len(coords), rmse=rmse, mae=mae,
                r2=r2, sam_deg=sam_deg, sam_pixel_coverage=float(sam["sam_pixel_coverage"]),
                seconds=float(duration), source="selected_ROI_not_standard_benchmark_mask")
    (out / f"metrics_{method}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def parse_crop(value, shape):
    if not value:
        return slice(0, shape[0]), slice(0, shape[1])
    try:
        aa, bb = value.split(",")
        y0, y1 = (int(t) for t in aa.split(":"))
        x0, x1 = (int(t) for t in bb.split(":"))
    except (ValueError, TypeError) as exc:
        raise ValueError("--crop requires y0:y1,x0:x1 (within the 128x128 ROI)") from exc
    if not (0 <= y0 < y1 <= shape[0] and 0 <= x0 < x1 <= shape[1]):
        raise ValueError(f"Crop {value} outside ROI bounds {shape[:2]}")
    return slice(y0, y1), slice(x0, x1)


def render(out, args):
    """Display-only step: no GPUs, training artifacts, or source images needed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    gt = np.load(out / "roi_ground_truth.npy", allow_pickle=False)
    corrupted = np.load(out / "roi_corrupted.npy", allow_pickle=False)
    synthetic = np.load(out / "roi_synthetic_mask.npy", allow_pickle=False)
    waves = np.load(out / "wavelengths_nm.npy", allow_pickle=False)
    saved_meta = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    flip_x = (bool(saved_meta.get("display_flip_x", saved_meta["dataset"].upper().startswith("VNIR")))
              if args.flip_x is None else args.flip_x)
    band = int(saved_meta["display_band"] if args.band is None else args.band)
    if band < 0 or band >= gt.shape[2]:
        raise ValueError(f"Band must be between 0 and {gt.shape[2]-1}")
    if args.gamma <= 0 or args.stretch <= 0:
        raise ValueError("--gamma and --stretch must be > 0")
    ys, xs = parse_crop(args.crop, gt.shape)
    valid = np.isfinite(gt[:, :, band])
    values = gt[:, :, band][valid]
    lo, hi = map(float, np.percentile(values, [1, 99.5]))
    hi = max(hi, lo + 1e-6)

    def display(array):
        a = np.asarray(array[ys, xs, band], dtype=np.float32)
        scaled = np.clip((a - lo) / (hi - lo), 0, 1)
        scaled = np.clip((scaled - 0.5) * args.stretch + 0.5, 0, 1)
        return np.power(scaled, args.gamma)

    methods = [m for m in SUPPORTED if (out / f"roi_reconstructed_{m}.npz").is_file()]
    names = ["Ground truth", "Corrupted input"] + [DISPLAY_NAMES[m] for m in methods]
    arrays = [gt, corrupted]
    for method in methods:
        with np.load(out / f"roi_reconstructed_{method}.npz", allow_pickle=False) as payload:
            arrays.append(np.asarray(payload["reconstruction"], np.float32))

    columns = min(4, len(names))
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 4.7 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (ax, name, arr) in enumerate(zip(axes.flat, names, arrays)):
        data = display(arr)
        data[~np.isfinite(data)] = 0.0
        ax.imshow(data, cmap=args.cmap, vmin=0, vmax=1, interpolation="nearest")
        if flip_x:
            ax.invert_xaxis()
        ax.set_title(name, fontsize=11)
        if i >= 2 and (out / f"metrics_{methods[i-2]}.json").is_file():
            metric = json.loads((out / f"metrics_{methods[i-2]}.json").read_text(encoding="utf-8"))
            if metric["coverage"] == 0:
                ax.text(.5, .5, "No valid predictions", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11, color="white",
                        bbox=dict(facecolor="black", alpha=.8, edgecolor="none"))
    bw_mode = saved_meta.get("band_wide_mode", "local")
    sampled = int(saved_meta.get("evaluated_target_voxels", 0)) < int(saved_meta.get("full_mask_valid_target_voxels", 0))
    sample_note = " | sampled target pixels only" if bw_mode == "full-roi" and sampled else ""
    fig.suptitle(f"{saved_meta['dataset']} | {saved_meta['scenario']} S{saved_meta['severity']} | "
                 f"band {band} ({waves[band]:.1f} nm){sample_note}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(out / "comparison.png", dpi=args.dpi)
    fig.savefig(out / "comparison.svg")
    plt.close(fig)

    # In addition to the overview, export each method's band as a separate
    # frameless panel; this makes subsequent paper-layout work much easier.
    panels = out / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    for key, arr in zip(["ground_truth", "corrupted"] + methods, arrays):
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.imshow(np.nan_to_num(display(arr), nan=0.0), cmap=args.cmap,
                  vmin=0, vmax=1, interpolation="nearest")
        if flip_x:
            ax.invert_xaxis()
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(panels / f"{key}.png", dpi=args.dpi, pad_inches=0)
        plt.close(fig)

    if args.show_errors:
        fig, axes = plt.subplots(1, max(1, len(methods)), figsize=(4.4 * max(1, len(methods)), 4.5), squeeze=False)
        valid_target_band = synthetic[ys, xs, band] & np.isfinite(gt[ys, xs, band])
        errors = []
        for arr in arrays[2:]:
            err = np.abs(arr[ys, xs, band] - gt[ys, xs, band]).astype(float)
            err[~valid_target_band] = np.nan
            errors.append(err)
        vmax = max([float(np.nanmax(v)) for v in errors if np.isfinite(v).any()] or [1.0])
        for ax, method, err in zip(axes.flat, methods, errors):
            im = ax.imshow(np.ma.masked_invalid(err), cmap="magma", vmin=0,
                           vmax=max(vmax, 1e-9), interpolation="nearest")
            ax.set_title(DISPLAY_NAMES[method], fontsize=11)
            if flip_x:
                ax.invert_xaxis()
            ax.axis("off")
        if errors:
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.7, label="Absolute reflectance error")
        fig.savefig(out / "absolute_errors.png", dpi=args.dpi, bbox_inches="tight")
        fig.savefig(out / "absolute_errors.svg", bbox_inches="tight")
        plt.close(fig)
    print(f"[RENDER] {out / 'comparison.png'} | band {band}, gamma={args.gamma}, stretch={args.stretch}", flush=True)
    return methods


def write_summary(out):
    records = []
    for method in SUPPORTED:
        path = out / f"metrics_{method}.json"
        if path.is_file():
            records.append(json.loads(path.read_text(encoding="utf-8")))
    if records:
        with (out / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ("method", "total_target_voxels", "predicted_voxels", "coverage", "rmse", "mae", "r2", "sam_deg", "sam_pixel_coverage", "seconds", "source")
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)


def run_case(args, out, methods, rerun):
    """Reconstruct all selected methods for a prepared or newly selected case."""
    select_roi(args, out)
    meta, gt, invalid, synthetic, mask, coords = load_selection(out)
    if (args.scenario == "band_wide_saturation" and
            (meta.get("band_wide_mode", "local") != args.band_wide_mode or
             meta.get("band_wide_eval", "benchmark") != args.band_wide_eval or
             meta.get("band_wide_band_selection", "display") != args.band_wide_band_selection or
             int(meta.get("band_wide_repeat", 1)) != args.band_wide_repeat)):
        raise RuntimeError("Saved Band-wide ROI does not match requested mask/evaluation mode")
    if meta.get("router") != args.router:
        raise RuntimeError(f"Requested Router {args.router} differs from saved {meta.get('router')}")
    if bool(meta.get("spatial_consistency", False)) != args.spatial_consistency:
        raise RuntimeError("Saved selection was made with another spatial Router setting")
    if (meta.get("global_minority_percent") != args.global_minority_percent or
            meta.get("global_minority_policy", "threshold") != args.minority_policy or
            (args.minority_policy == 'hybrid_gap' and
             float(meta.get('three_expert_min_gap_percent', -1)) != args.three_expert_min_gap_percent)):
        raise RuntimeError("Saved ROI has a different Global Minority policy/threshold")
    if meta.get("fusion_donor_match") != "robust":
        raise RuntimeError("Saved ROI has a different Fusion Donor matching setting")
    if bool(meta.get("fusion_guard", False)) != bool(args.fusion_guard):
        raise RuntimeError("Saved selection was made with another Fusion Guard setting")
    if args.preset != meta["preset"] or args.dataset.upper() != meta["dataset"].upper():
        raise RuntimeError("Resume arguments must match the saved selection preset and dataset")
    persist_reference(out, gt, invalid, synthetic, mask, coords, meta)
    print(f"[ROI] {out} | {len(coords)} target voxels | "
          f"{int(np.any(synthetic, axis=2).sum())} affected spatial pixels", flush=True)
    if args.select_only:
        print(f"[SELECT ONLY] Saved selection: {out}")
        return
    config = configure_model_environment(meta)
    tp_manifest = out / "tpssr_checkpoint_manifest.json"
    signature = None
    if "tpssr" in methods:
        signature = tpssr_checkpoint_signature(config, args)
        saved = out / "roi_reconstructed_tpssr.npz"
        if saved.exists() and "tpssr" not in rerun:
            if not tp_manifest.is_file() or json.loads(tp_manifest.read_text(encoding="utf-8")) != signature:
                raise RuntimeError("This ROI has an older or unverified TPSSR reconstruction. "
                                   "Use --rerun tpssr to regenerate it without rerunning other methods, "
                                   "or choose a new --outdir.")
    # Avoid silently combining an RF v1 image with the newly trained RF v2.
    rf_manifest = out / "rf_checkpoint_manifest.json"
    rf_signature = None
    if "random_forest_multiscale" in methods:
        rf_path = config.MODELS_DIR / "random_forest" / "random_forest_multiscale_model.joblib"
        if not rf_path.is_file():
            raise FileNotFoundError(f"RF checkpoint missing: {rf_path}")
        rf_signature = dict(path=str(rf_path.resolve()), size=rf_path.stat().st_size,
                            mtime_ns=rf_path.stat().st_mtime_ns)
        rf_saved = out / "roi_reconstructed_random_forest_multiscale.npz"
        if rf_saved.is_file() and "random_forest_multiscale" not in rerun:
            if (not rf_manifest.is_file()
                    or json.loads(rf_manifest.read_text(encoding="utf-8")) != rf_signature):
                raise RuntimeError("Old or unverified RF reconstruction found in this ROI. "
                                   "Use --rerun random_forest_multiscale, or a new --outdir.")
    failed = []
    for method in methods:
        saved = out / f"roi_reconstructed_{method}.npz"
        if saved.is_file() and method not in rerun:
            print(f"[CACHE] {method}: reconstruction saved; skipping inference", flush=True)
            continue
        print(f"[START] {method}", flush=True)
        start = time.monotonic()
        try:
            predictions = method_prediction(method, args, out, gt, mask, coords, meta, config)
            duration = time.monotonic() - start
            metrics = save_prediction(out, method, gt, mask, coords, predictions, duration)
            if method == "tpssr" and signature is not None:
                tp_manifest.write_text(json.dumps(signature, indent=2), encoding="utf-8")
            if method == "random_forest_multiscale" and rf_signature is not None:
                rf_manifest.write_text(json.dumps(rf_signature, indent=2), encoding="utf-8")
            print(f"[SAVED] {method} | RMSE={metrics['rmse']} | "
                  f"coverage={metrics['coverage']:.2%} | seconds={duration:.2f}", flush=True)
        except Exception as exc:
            import traceback
            error_file = out / f"error_{method}.txt"
            error_file.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[ERROR] {method}: {exc}. Other methods will continue. Details: {error_file}", flush=True)
            failed.append(method)
        finally:
            write_summary(out)
    render(out, args)
    if failed:
        raise RuntimeError(f"Methods not completed: {', '.join(failed)}. "
                           f"Rerun with --outdir {out} to resume.")
    print(f"[DONE] Saved reconstructions and visualizations in {out}", flush=True)


def main(argv=None):
    args = arguments(argv)
    if args.list_scenarios:
        print_scenarios()
        return
    if args.render_only is not None:
        out = args.render_only.expanduser().resolve()
        render(out, args)
        return
    resolve_display_options(args)
    os.environ["TPSSR_FUSION_DONOR_MATCH"] = "robust"
    print(f"[CONFIG] {args.dataset.upper()} | {args.preset} | Router={args.router} | "
          f"Global Minority={args.minority_policy} / {args.global_minority_percent:g}%, "
          f"three-expert gap={args.three_expert_min_gap_percent:g} pp | "
          f"Spatial consistency={args.spatial_consistency} | "
          f"flip-x display={args.flip_x} | TPSSR GPUs={args.gpus}", flush=True)
    methods = parse_methods(args.methods)
    rerun = parse_methods(args.rerun)
    if set(rerun) - set(methods):
        raise ValueError("--rerun methods must be included in --methods")
    if not methods and not args.select_only:
        raise ValueError("No methods selected")

    out = output_folder(args)
    if args.from_outdir is not None:
        clone_roi_from_previous(args.from_outdir, out, args)
    run_case(args, out, methods, rerun)


if __name__ == "__main__":
    main()
