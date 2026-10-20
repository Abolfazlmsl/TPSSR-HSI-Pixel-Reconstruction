# TPSSR — Hyperspectral Pixel Reconstruction

TPSSR V2.3 reconstructs missing and corrupted hyperspectral reflectance values using a spectral network and a geometry-guided spatial completion network. This distribution includes the five classical benchmarks and two independently maintained external methods.

## Project layout

```text
train.py                       Four-stage TPSSR training coordinator
training/                     Spectral backbone and confidence-gate training
core/src/pipeline.py           TPSSR inference and observed-only routing
core/src/spatial_inference.py  Geometry-guided spatial reconstruction
core/src/                     Reconstruction networks and evaluation utilities
core/reconstruction_methods/tpssr/   Geometry and topology training
core/reconstruction_methods/baselines/    Spectral, spatial, spatio-spectral
core/reconstruction_methods/pca/          PCA benchmark
core/reconstruction_methods/random_forest/ Random Forest benchmark
core/reconstruction_methods/pfgin/        PFGIN wrapper
core/reconstruction_methods/dds2m/        DDS2M wrapper
external/                      Source implementations for PFGIN and DDS2M
test.py                        Masked-pixel prediction with per-mask progress
evaluate.py                    Summaries and original/corrupted/reconstructed visuals
benchmarks.py                  Five classical/reference benchmarks
external_methods.py            PFGIN and DDS2M entrypoint
plot_method_comparison.py      Scenario/severity plots (DDS2M omitted pending completion)
preflight.py                   Checkpoint and processed-data checks
```

## Reproducible protocol and presets

The released TPSSR V2.3 architecture and fixed mask-only routing policy can be
trained under these four presets. Each experiment has separate folders for
checkpoints, masks and results; the original `swir_8x2` checkpoint layout is
unchanged.

| Preset | Training acquisitions | Held-out test acquisitions | Spectral validation acquisitions |
|---|---|---|---|
| `swir_8x2` | SWIR1–4, SWIR6–9 | SWIR5, SWIR10 | SWIR4, SWIR9 |
| `swir_region_a_to_b` | SWIR1–5 | SWIR6–10 | SWIR4, SWIR5 |
| `swir_region_b_to_a` | SWIR6–10 | SWIR1–5 | SWIR9, SWIR10 |
| `vnir_8x2` | VNIR1–4, VNIR6–9 | VNIR5, VNIR10 | VNIR4, VNIR9 |

Spectral validation acquisitions are **held out from spectral weight fitting**,
although they belong to the source-side partition; the geometry/topology
training uses the configured full source partition with within-acquisition
spatial validation. Therefore A→B and B→A each have three acquisitions for
spectral weight fitting and two for core-level validation, not five spectral
training acquisitions. This choice preserves two-core spectral acceptance
criteria and prevents target-area validation leakage.

The official benchmark uses **200 masks per complete experiment** (100 per test
acquisition with two tests, 40 with five tests). The same deterministic
scenario/severity-stratified mask budgeting is used by TPSSR and the analytical
Spectral baseline. The routing policy was previously developed with insight
from SWIR5/SWIR10 and is frozen for these runs; do not claim those specific
acquisitions are completely unseen in policy development.

## Environment

Recommended: Linux, Python 3.11+, NVIDIA drivers, compatible CUDA-enabled PyTorch, and two GPUs for distributed training. Keep the dependency versions aligned with the environment used for prior experiments. Install core dependencies using:

```bash
python -m pip install -r requirements.txt
```

The `external/` directory retains the separately authored PFGIN and DDS2M source code; consult their documentation for method-specific dependencies. Their original software licenses and scientific attribution must be respected.

## Input data and preparation

Place each acquisition in `input_raw/<acquisition>/` according to the requirements of `core/data_preparation/prepare_dataset.py`. Processed arrays are stored under `processed_data/<acquisition>/` (including `calibrated_reflectance.npy`, `invalid_mask.npy` and `wavelengths_nm.npy`). Prepare only if the existing processed data and masks are absent:

```bash
python prepare_dataset.py --preset swir_8x2
```

Preserve the original experiment's processed arrays, synthetic masks, and trained checkpoints when comparing results. Do not replace those with new random masks for a paired comparison.

## TPSSR training and evaluation (two GPUs)

For **each** new preset, run the following in sequence, replacing `PRESET` with
`swir_region_a_to_b`, `swir_region_b_to_a`, or `vnir_8x2`:

```bash
python prepare_dataset.py --preset PRESET
python preflight.py --preset PRESET --data-only
python train.py --preset PRESET --gpus 2 --dry-run
python train.py --preset PRESET --gpus 2
python preflight.py --preset PRESET
python test.py --preset PRESET --gpus 2 --max-masks 200
python evaluate.py --preset PRESET --visuals 0
python benchmarks.py --preset PRESET --method spectral --stage test --max-masks 200
```

`--data-only` checks existing processed arrays, wavelength alignment and mask
presence without loading the full cubes or using a GPU. Do not start the long
training run if this check fails. For new experiments, the backbone, band gate,
geometry and topology all require new weights; never copy the original
`swir_8x2` weights across sensor or transfer experiments. The checkpoint tag
uses the first held-out acquisition only as a **folder name**, not as training
input. Two GPUs train CNN and Hybrid candidates concurrently, then the two
gate candidates; geometry/topology run DDP on two GPUs. The fixed V2.3 routing
policy is not retrained.

The spectral backbone and confidence gate have validation acceptance checks.
If no band gate meets them within the configured epochs, training stops instead
of silently promoting a diagnostic model to publication status. Investigate
validation results before making a documented new experiment or retrying.

Results are stored under `results/<experiment>/...`, and TPSSR checkpoints
under `trained_models/<experiment>/tpssr/`. The clean code bundle contains
no actual data or trained weights. The smoke and dry-run checks are NOT a
substitute for end-to-end GPU verification on the laboratory server.

## Benchmarks

```bash
python benchmarks.py --preset swir_8x2 --method all --stage both --max-masks 200
python external_methods.py --preset swir_8x2 --method pfgin --stage both --max-masks 200
```

Methods: **Spectral**, **Spatial**, **Spatio-Spectral**, **PCA**, **Random Forest**, **PFGIN**, **DDS2M**. PFGIN can have variable GPU-memory demands; run according to its external documentation. DDS2M is very computationally intensive, has no joint-acquisition training stage, and can be run independently when needed:

```bash
python external_methods.py --preset swir_8x2 --method dds2m --stage test --max-masks 200
```

DDS2M is deliberately **not** included in the method comparison plots until the external run is complete. Avoid rerunning it merely to generate TPSSR plots.

```bash
python plot_method_comparison.py --experiment-dir results/swir__train_1-2-3-4-6-7-8-9__test_5-10
```

The evaluator reports **RMSE, MAE, R², SAM and SSIM** where provided by each method, but SSIM is not used in the final cross-method comparison. Missing benchmark SSIM values are not invented; the plotting script leaves them absent. TPSSR's combined results appear under `results/<experiment>/combined/tpssr/`. Figures show original / masked / reconstructed panels and should be interpreted alongside coverage and comparable sampled-mask identities.

## Safety and reproducibility

- Back up existing results, masks and trained models before replacing a project directory.
- The clean source distribution uses standardized checkpoint folders. Previously trained checkpoints require a one-time file layout migration; keep the original files unchanged until verified using `preflight.py`.
- Checkpoint files are trusted Python/PyTorch artifacts. Only load checkpoints from trusted sources.
- CLI dry-runs and unit tests do **not** establish numerical equivalence to the old GPU implementation or prove full end-to-end convergence. Verify a small set of identical saved masks with the same frozen weights before a paper-scale run.
- PFGIN and DDS2M vendor code has not been rewritten; keep upstream licenses and attribution.

## Development checks

```bash
python -m compileall -q .
python -m pytest -q tests
python train.py --preset swir_8x2 --dry-run
python train.py --preset swir_region_a_to_b --dry-run
```

## Learned Neighbor Fusion (experimental stage 1)

This stage preserves the trained CNN-Transformer, spectral confidence gate,
Geometry/Topology weights, and the benchmarked `neighbor_v2` option. A separate
local residual-attention model learns how to combine interpolation, frozen
Hybrid, and observed donor curvature. Whole-spectrum / no-anchor pixels still
use the original Region path. The learned Router is **not** part of this stage.

Train separately for each transfer direction (two GPUs; no backbone retraining):

```bash
python train.py --preset swir_region_a_to_b --stage fusion --gpus 2
python train.py --preset swir_region_b_to_a --stage fusion --gpus 2
```

Source-only training history and the best model are saved under
`trained_models/<experiment>/tpssr/neighbor_fusion/`. The history compares
Fusion with interpolation, frozen Hybrid and an **approximation** of Neighbor
V2's spectral policy on the same synthetic samples. The authoritative V2
comparison is the identical-mask held-out run below. The learned model is
isolated and cannot overwrite the existing checkpoints.

```bash
python test.py --preset swir_region_a_to_b --router learned_fusion --gpus 2 --max-masks 200
python evaluate.py --preset swir_region_a_to_b --router learned_fusion --visuals 0
python test.py --preset swir_region_b_to_a --router learned_fusion --gpus 2 --max-masks 200
python evaluate.py --preset swir_region_b_to_a --router learned_fusion --visuals 0
```

`--max-masks 200` is the existing **experiment-wide** total; for region
transfer it is distributed across five destination cores, not 200 per core.
For an early integration check, use `--max-masks 10` before the full 200.
Learned Fusion writes to separate `tpssr_learned_fusion` result folders.
Original V2 results in `tpssr_neighbor_v2` remain intact.

The generated training masks are spatial-spectral (with randomized wavelength
positions); masked values are erased before building model inputs. Synthetic
whole-spectrum centers and mixed-saturation cases without spectral anchors
are excluded from Fusion fitting because Region handles them. The current
source validation is developmental, since source validation cores have been
used for prior architecture selection. A clean independent final evaluation
requires an untouched acquisition/protocol.
