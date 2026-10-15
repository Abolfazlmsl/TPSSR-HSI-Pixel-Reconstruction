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

## Reproducible protocol

The released TPSSR network trainers target the `swir_8x2` experiment. Train acquisitions are **SWIR1, SWIR2, SWIR3, SWIR4, SWIR6, SWIR7, SWIR8 and SWIR9**; independent held-out tests are **SWIR5 and SWIR10**. The spectral network fits its parameters using six training acquisitions and uses SWIR4/SWIR9 for validation. Both held-out acquisitions share one trained TPSSR model; `swir5` is only its checkpoint tag.

The test benchmark uses **200 masks in total across SWIR5 and SWIR10**, not 200 per acquisition. The global budget is allocated deterministically between test acquisitions. The same experiment mask selection is used by the benchmark launchers. No held-out reflectance is used for network training or choosing inference routes. Prior selection of the routing policy using these test acquisitions does, however, make performance on them *development-set*, not untouched holdout, evidence; document that distinction in publication materials.

The `experiment_config.py` module supports other baseline and data-preparation splits, but the included spectral TPSSR trainer is fixed to `swir_8x2`. Do not label other TPSSR splits as supported without adapting and validating the network trainers.

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

## Training and inference

```bash
python train.py --preset swir_8x2 --dry-run
python train.py --preset swir_8x2 --gpus 2
python preflight.py --preset swir_8x2
python test.py --preset swir_8x2 --gpus 2 --max-masks 200
python evaluate.py --preset swir_8x2 --visuals 3
```

Training stages: (1) spectral backbone; (2) band-wise confidence gate with a frozen backbone; (3) learned geometry guidance; (4) topology-based reconstruction with frozen guidance. The routing rules are deterministic and use only corrupted observations, mask geometry, and measured neighbors, with no additional trainable router. Existing accepted checkpoint files are not overwritten by the coordinator.

The spectral gate is pinned to the SHA256 digest of its source backbone. An incompatible backbone is rejected rather than silently loaded. No model weights or datasets are included in this code-only release.

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

The evaluator reports **RMSE, MAE, R², SAM and SSIM** where provided by each method. Missing benchmark SSIM values are not invented; the plotting script leaves them absent. TPSSR's combined results appear under `results/<experiment>/combined/tpssr/`. Figures show original / masked / reconstructed panels and should be interpreted alongside coverage and comparable sampled-mask identities.

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
```
