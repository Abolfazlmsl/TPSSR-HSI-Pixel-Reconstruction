# TPSSR: Hyperspectral Pixel Reconstruction

**Missing-pixel reconstruction for hyperspectral sediment-core images.**

TPSSR combines three reconstruction experts with a learned, pixel-level expert
router to recover missing or corrupted reflectance values while preserving
sedimentary structures.

The three experts are **Spectral Interpolation**, **Learned Neighbor Fusion**, and
**Geometry/Topology-guided Region Reconstruction**. A learned RankCV router
selects among them. The framework supports VNIR and SWIR imagery, including
missing detector lines and large spatial–spectral defects.

## Features

- Ten synthetic corruption scenarios at five severity levels.
- Split-aware training and evaluation with 200 test masks per experiment.
- SWIR and VNIR experiments, including cross-region transfer.
- Comparison with Spectral, Spatial, Spatio-Spectral, PCA, multiscale RF, DDS2M and PFGIN.
- Visual reconstruction of selected regions and real missing SWIR detector lines.
- RMSE, MAE, R² and spectral angle (SAM) evaluation.

## Project structure

```text
TPSSR-HSI-Pixel-Reconstruction/
├── tpssr.py                    # Main command-line entry point
├── train.py                    # TPSSR training stages
├── train_region_lineaware.py   # Geometry/topology expert training
├── test.py                     # Evaluation on fixed synthetic masks
├── prepare_dataset.py         # Reflectance calibration and mask generation
├── evaluate.py                 # Results aggregation and optional figures
├── benchmarks.py              # Five classical/reference benchmarks
├── external_methods.py        # PFGIN and DDS2M adapters
├── preflight.py                # Data and checkpoint checks
├── experiment_config.py       # Data splits
├── training/                    # Spectral, fusion and router training
├── core/                        # Reconstruction models and evaluation code
├── tools/                       # Figures, ROI and real-line reconstruction
├── tests/                       # CPU-only regression tests
├── docs/                        # Method and dependency notes
├── external/                    # Cloned upstream methods (not tracked)
├── input_raw/                   # Your source acquisitions (not tracked)
├── processed_data/              # Prepared arrays (not tracked)
├── trained_models/              # Checkpoints (not tracked)
└── results/                     # Evaluation outputs (not tracked)
```

## Installation

Designed for Linux, Python 3.10+ and CUDA-capable PyTorch.
Install the PyTorch build compatible with your GPU, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the regression tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

No data or trained TPSSR checkpoints are included in the source release.

## Prepare the data

Put the raw acquisitions under `input_raw/SWIR1`, ..., `input_raw/SWIR10`
(and/or `VNIR1`, ..., `VNIR10`). Each acquisition needs the original ENVI data
and its corresponding white and dark reference files. The loader discovers
matching `.hdr` files; consult `core/src/envi_io.py` for the supported layout.

Prepare the main SWIR experiment:

```bash
python tpssr.py prepare --preset swir_8x2
python tpssr.py preflight --preset swir_8x2 --data-only
```

The preparation step writes `calibrated_reflectance.npy`, `invalid_mask.npy`
and `wavelengths_nm.npy` under each `processed_data/<dataset>/`, and writes
reproducible benchmark masks under `simulated_saturation/`.

## Train TPSSR

For a new experiment, train its components in sequence:

```bash
python tpssr.py train --preset swir_8x2 --stage all
python tpssr.py train-region --preset swir_8x2
python tpssr.py train --preset swir_8x2 --stage fusion
python tpssr.py train --preset swir_8x2 --stage router3_lineaware
python tpssr.py train --preset swir_8x2 --stage router3_lineaware_rankcv
```

The RankCV stage uses **source-only calibration labels** generated during the
preceding router-training stage. Do not fit a router with held-out test data.
Accepted checkpoints are reused rather than silently overwritten. The main
training path uses **two GPUs**; training a model from scratch is substantial.

If you already have the correctly trained models, restore the corresponding
`trained_models/<experiment>/` folders and skip training.

## Test and evaluate

```bash
python tpssr.py preflight --preset swir_8x2
python tpssr.py test --preset swir_8x2
python tpssr.py evaluate --preset swir_8x2 --visuals 0
```

`test` uses the final learned RankCV router, robust neighbor matching,
hybrid-gap policy (10 percentage points), a 10% two-expert threshold, and
lazy expert caching. It evaluates **200 masks in total**, shared across the
held-out images. Use `--gpus 2` for the default two-GPU workflow.

The evaluation command reads the corresponding final result folder, including
its routing-policy suffix. It aggregates metrics without regenerating images by
default (`--visuals 0`); use the ROI tools for publication figures.

## Other experiments

| Preset | Training images | Held-out test images |
|---|---|---|
| `swir_8x2` | SWIR1–4, SWIR6–9 | SWIR5, SWIR10 |
| `swir_region_a_to_b` | SWIR1–5 | SWIR6–10 |
| `swir_region_b_to_a` | SWIR6–10 | SWIR1–5 |
| `vnir_8x2` | VNIR1–4, VNIR6–9 | VNIR5, VNIR10 |

Replace the preset in the commands above to run another experiment. The same
trained model is shared across the held-out acquisitions in a given split.

## Benchmark methods

Five non-deep-learning baselines are included. `RF` is the multiscale
spatio-spectral Random Forest, while `PCA` uses spectral PCA only.

```bash
python tpssr.py benchmarks --preset swir_8x2 --method all --stage both --max-masks 200
```

PFGIN and DDS2M are optional and computationally expensive. Install the
upstream repositories before running them:

```bash
python core/reconstruction_methods/pfgin/setup_official.py
python core/reconstruction_methods/dds2m/setup_official.py
python tpssr.py external --preset swir_8x2 --method all --stage both --max-masks 200
```

Read [Third-party methods](docs/THIRD_PARTY.md) for sources and licensing.

## Figures and real missing lines

To compare reconstructed ROIs in the benchmark:

```bash
python tpssr.py roi --preset swir_8x2 --dataset SWIR5 \
  --scenario whole_spectrum_pixels --severity 5 --methods all --gpus 2
```

For a candidate list of genuine missing SWIR lines:

```bash
python tpssr.py real-lines --list-only
```

To reconstruct two selected lines and save full-band before/after images:

```bash
python tpssr.py real-lines --datasets SWIR5 SWIR10 \
  --band-swir5 141 --line-x-swir5 141 \
  --band-swir10 35 --line-x-swir10 183 --gpus 2 --dpi 400
```

The line tool preserves observed pixels and uses the same display settings
for each before/after pair. It saves separate two-dimensional PNG/TIFF files.
Real missing pixels have no ground truth, so no RMSE is reported for them.

Other tools in `tools/` export hyperspectral cube figures, scenario boxplots
and consolidated results. For example:

```bash
python tpssr.py boxplots --excel TPSSR_FINAL_ALL_RESULTS.xlsx \
  --split SWIR_8x2 --outdir figures/scenario_boxplots --dpi 400
```

## Outputs and reproducibility

`results/<experiment>/<test-image>/` stores reconstruction metrics. Published
comparisons should distinguish per-method valid coverage from the shared-mask
subset. The project reports RMSE, MAE, R² and SAM; SSIM is not used in the
main numerical results.

The scripts use the original image split and fixed synthetic-mask protocol.
Default values are set in `core/config.py` and `experiment_config.py`.
Because model weights and raw acquisitions are not distributed here, numeric
paper results cannot be reproduced from this source archive alone.

## License and citation

Original TPSSR code is released under the [MIT License](LICENSE).
For PFGIN and DDS2M, follow their upstream licenses and citation requirements.

If you use this repository in research, please cite the associated TPSSR paper
when it becomes available. Publication metadata can be added here after release.
