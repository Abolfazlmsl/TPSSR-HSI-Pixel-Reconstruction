# TPSSR-Missing-HSI-Pixel-Reconstruction

A clean research implementation of **TPSSR**, a hybrid deep-learning framework for texture-preserving reconstruction of corrupted, saturated, and missing pixels in hyperspectral sediment-core images.

The final model combines a spectral expert, a Deep-v3.1 expert, and a structure-guided Region branch through a hierarchical routing strategy.

## Final TPSSR architecture

```text
Corrupted HSI + mask
        |
        v
Stage-1 Router
  Spectral <-> Deep-v3.1
        |
        v
Hard-case Gate
   | No                    | Yes
   v                       v
keep Stage-1 output   Frozen V27 Stage-A
                           |
                           v
                    V13 TopologyFirst V1
                           |
                           v
                     final reconstruction
```

**Important:** V27 provides structural guidance only. Its reconstructed pixels are not used as the final Region output. Final Region values are synthesized by **V13 TopologyFirst V1**.

---

## Dataset organization

The current experimental design uses **20 hyperspectral acquisitions**:

```text
SWIR1 ... SWIR10
VNIR1 ... VNIR10
```

For each sensor, the acquisitions belong to two different regions:

```text
Region A: 1-5
Region B: 6-10
```

Example:

```text
SWIR Region A: SWIR1, SWIR2, SWIR3, SWIR4, SWIR5
SWIR Region B: SWIR6, SWIR7, SWIR8, SWIR9, SWIR10

VNIR Region A: VNIR1, VNIR2, VNIR3, VNIR4, VNIR5
VNIR Region B: VNIR6, VNIR7, VNIR8, VNIR9, VNIR10
```

Raw data should be placed under:

```text
input_raw/
├── SWIR1/
├── SWIR2/
├── ...
├── SWIR10/
├── VNIR1/
├── VNIR2/
├── ...
└── VNIR10/
```

Each acquisition folder should contain the ENVI measurement and its matching white and dark references. The preparation code identifies the measurement, `whiteref`, and `darkref` headers automatically.

SWIR and VNIR are treated as **separate sensor experiments**. They are never mixed in the same training split.

---

## Final experimental plan

### Experiment 1 — Main SWIR method comparison

All methods are compared using an **8-train / 2-test SWIR split**, with one held-out acquisition from each region.

Default preset:

```text
Train:
SWIR1 SWIR2 SWIR3 SWIR4
SWIR6 SWIR7 SWIR8 SWIR9

Test:
SWIR5 SWIR10
```

This experiment is used for:

- Final TPSSR
- Spectral interpolation
- Spatial interpolation
- Spatio-spectral interpolation
- PCA
- Random Forest
- PFGIN
- DDS2M

The train/test split is identical for all methods.

### Experiment 2 — TPSSR cross-region generalization: Region A -> Region B

```text
Train: SWIR1 SWIR2 SWIR3 SWIR4 SWIR5
Test:  SWIR6 SWIR7 SWIR8 SWIR9 SWIR10
```

Only TPSSR is required for this experiment.

### Experiment 3 — TPSSR cross-region generalization: Region B -> Region A

```text
Train: SWIR6 SWIR7 SWIR8 SWIR9 SWIR10
Test:  SWIR1 SWIR2 SWIR3 SWIR4 SWIR5
```

Only TPSSR is required for this experiment.

### Experiment 4 — TPSSR VNIR evaluation

Default VNIR 8-train / 2-test split:

```text
Train:
VNIR1 VNIR2 VNIR3 VNIR4
VNIR6 VNIR7 VNIR8 VNIR9

Test:
VNIR5 VNIR10
```

Only TPSSR is required for this experiment.

---

## Included comparison methods

Five internal benchmark methods are included:

1. Spectral interpolation
2. Spatial interpolation
3. Spatio-spectral interpolation
4. PCA
5. Random Forest

Two external methods are included:

- PFGIN
- DDS2M

The external source trees are stored under:

```text
external/
├── PFGIN/
└── DDS2M/
```

---

## Project layout

```text
TPSSR_Final_Source/
├── prepare_dataset.py
├── train.py
├── test.py
├── evaluate.py
├── benchmarks.py
├── external_methods.py
├── run_all.py
├── experiment_config.py
├── requirements.txt
├── core/
│   ├── config.py
│   ├── data_preparation/
│   ├── reconstruction_methods/
│   └── src/
├── external/
│   ├── PFGIN/
│   └── DDS2M/
├── input_raw/
├── processed_data/
├── simulated_saturation/
├── trained_models/
└── results/
```

---

## Installation

Use the same CUDA/PyTorch environment in which TPSSR was developed, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

DDS2M also retains its original requirements file:

```text
external/DDS2M/requirements.txt
```

The final TPSSR training wrapper is configured for **two GPUs**.

---

# Running the experiments

## 1. Main SWIR 8-train / 2-test experiment

### Prepare the data and fixed benchmark masks

```bash
python prepare_dataset.py --preset swir_8x2
```

Default split:

```text
Train: SWIR1 SWIR2 SWIR3 SWIR4 SWIR6 SWIR7 SWIR8 SWIR9
Test:  SWIR5 SWIR10
```

For each held-out acquisition, the fixed benchmark contains:

- 10 corruption scenarios
- 5 severity levels
- 4 repeats
- 200 masks per test acquisition

Therefore the default 8/2 experiment evaluates each method on:

```text
SWIR5  -> 200 masks
SWIR10 -> 200 masks
Total  -> 400 masks
```

### Train Final TPSSR

```bash
python train.py \
  --preset swir_8x2 \
  --gpus 2
```

TPSSR is trained **once** on all eight training acquisitions. The same trained model is then used for both held-out acquisitions.

Training stages:

```text
1. Deep-v3 teacher
2. Deep-v3.1 expert
3. V27 Stage-A / correspondence model
4. V13 TopologyFirst V1
5. Final hierarchical router
```

Existing checkpoints are reused automatically. To intentionally rerun all stages:

```bash
python train.py \
  --preset swir_8x2 \
  --gpus 2 \
  --force
```

### Test Final TPSSR

```bash
python test.py \
  --preset swir_8x2 \
  --gpus 2 \
  --max-masks 200 \
  --deep-batch 16 \
  --region-batch 32
```

The test code reports global mask progress, for example:

```text
[SWIR5][GPU 0] Mask 37/200 (18.5%) | spatio_spectral_block S4 R1
```

After SWIR5 is complete, the same trained model is tested on SWIR10.

The test is resume-safe. Completed masks are reused unless `--force` is supplied.

### Evaluate Final TPSSR

```bash
python evaluate.py \
  --preset swir_8x2 \
  --visuals 3
```

Evaluation reports:

- RMSE
- MAE
- R²
- SAM
- SSIM
- Coverage
- scenario-wise results
- severity-wise results
- route fractions
- per-acquisition summaries
- combined summary across all configured test acquisitions

It also exports representative visual examples as:

```text
Original | Corrupted | Final TPSSR
```

By default, three representative examples are produced across the whole experiment. The number can be changed with `--visuals`.

---

## 2. Run the five benchmark methods on the same SWIR split

All five methods:

```bash
python benchmarks.py \
  --preset swir_8x2 \
  --method all \
  --stage both
```

Or run methods individually:

```bash
python benchmarks.py --preset swir_8x2 --method spectral --stage both
python benchmarks.py --preset swir_8x2 --method spatial --stage both
python benchmarks.py --preset swir_8x2 --method spatio_spectral --stage both
python benchmarks.py --preset swir_8x2 --method pca --stage both
python benchmarks.py --preset swir_8x2 --method random_forest --stage both
```

Notes:

- Spectral, Spatial, and Spatio-spectral are analytical and do not require global training.
- PCA and Random Forest are trained once on the configured eight training acquisitions and then evaluated on both held-out acquisitions.
- Benchmark testing also reports `Mask i/200` progress.

---

## 3. Run the two external methods

### PFGIN

```bash
python external_methods.py \
  --preset swir_8x2 \
  --method pfgin \
  --stage both
```

### DDS2M

```bash
python external_methods.py \
  --preset swir_8x2 \
  --method dds2m \
  --stage both
```

Run both:

```bash
python external_methods.py \
  --preset swir_8x2 \
  --method all \
  --stage both
```

PFGIN uses its global training workflow. DDS2M performs self-supervised optimization during reconstruction according to its original method design.

Both wrappers preserve their cache/resume behavior and report test-mask progress.

---

# TPSSR cross-region generalization

## 4. Train on SWIR Region A and test on Region B

### Prepare

```bash
python prepare_dataset.py --preset swir_region_a_to_b
```

### Train

```bash
python train.py \
  --preset swir_region_a_to_b \
  --gpus 2
```

### Test

```bash
python test.py \
  --preset swir_region_a_to_b \
  --gpus 2 \
  --max-masks 200
```

The same model is evaluated independently on:

```text
SWIR6
SWIR7
SWIR8
SWIR9
SWIR10
```

With 200 masks per acquisition, this produces up to **1000 held-out benchmark masks**.

### Evaluate

```bash
python evaluate.py \
  --preset swir_region_a_to_b \
  --visuals 3
```

---

## 5. Train on SWIR Region B and test on Region A

```bash
python prepare_dataset.py --preset swir_region_b_to_a

python train.py \
  --preset swir_region_b_to_a \
  --gpus 2

python test.py \
  --preset swir_region_b_to_a \
  --gpus 2 \
  --max-masks 200

python evaluate.py \
  --preset swir_region_b_to_a \
  --visuals 3
```

This performs:

```text
Train: SWIR6-SWIR10
Test:  SWIR1-SWIR5
```

---

# TPSSR VNIR experiment

## 6. VNIR 8-train / 2-test evaluation

```bash
python prepare_dataset.py --preset vnir_8x2

python train.py \
  --preset vnir_8x2 \
  --gpus 2

python test.py \
  --preset vnir_8x2 \
  --gpus 2 \
  --max-masks 200

python evaluate.py \
  --preset vnir_8x2 \
  --visuals 3
```

Default split:

```text
Train: VNIR1 VNIR2 VNIR3 VNIR4 VNIR6 VNIR7 VNIR8 VNIR9
Test:  VNIR5 VNIR10
```

---

# Custom train/test splits

The ready-made presets are only defaults. Any held-out acquisition combination can be specified explicitly.

For example, to test on `SWIR3` and `SWIR8`:

```bash
python prepare_dataset.py \
  --test-datasets SWIR3 SWIR8
```

If `--train-datasets` is omitted, the code automatically uses the remaining SWIR acquisitions as training data:

```text
Train: SWIR1 SWIR2 SWIR4 SWIR5 SWIR6 SWIR7 SWIR9 SWIR10
Test:  SWIR3 SWIR8
```

The same arguments can then be passed to training, testing, evaluation, benchmarks, and external methods:

```bash
python train.py --test-datasets SWIR3 SWIR8 --gpus 2
python test.py --test-datasets SWIR3 SWIR8 --gpus 2 --max-masks 200
python evaluate.py --test-datasets SWIR3 SWIR8 --visuals 3
```

Training datasets can also be specified explicitly:

```bash
python train.py \
  --train-datasets SWIR1 SWIR2 SWIR3 SWIR4 SWIR5 \
  --test-datasets SWIR6 SWIR7 SWIR8 SWIR9 SWIR10 \
  --gpus 2
```

Train and test sets are checked for overlap. A single experiment cannot mix SWIR and VNIR acquisitions.

---

# One-command workflows

## Main SWIR Final TPSSR workflow

```bash
python run_all.py \
  --preset swir_8x2 \
  --prepare \
  --final
```

## Final TPSSR + all five internal benchmarks

```bash
python run_all.py \
  --preset swir_8x2 \
  --prepare \
  --final \
  --benchmarks
```

## Everything, including external methods

```bash
python run_all.py \
  --preset swir_8x2 \
  --everything
```

The full workflow can take a long time because PFGIN and DDS2M are computationally expensive.

---

# Results organization

Each experiment receives its own model and result directory so that different train/test designs do not overwrite one another.

For the default SWIR 8/2 split, the experiment name is similar to:

```text
swir__train_1-2-3-4-6-7-8-9__test_5-10
```

Example result layout:

```text
results/
└── swir__train_1-2-3-4-6-7-8-9__test_5-10/
    ├── SWIR5/
    │   └── final_tpssr/
    ├── SWIR10/
    │   └── final_tpssr/
    └── combined/
        └── final_tpssr/
```

Evaluation outputs include files such as:

```text
SUMMARY.txt
summary.json
scenario_summary.csv
severity_summary.csv
visuals/
```

The combined folder summarizes all held-out acquisitions configured for that experiment.

---

# Evaluation protocol and data leakage

For every experiment:

- training uses only acquisitions explicitly assigned to the training set;
- held-out acquisitions are excluded from model training and validation;
- benchmark masks are generated separately for each held-out acquisition;
- all compared methods use the same configured split for the main SWIR comparison;
- SWIR and VNIR are evaluated independently;
- cross-region experiments train on one five-acquisition region and evaluate on the other region without retraining.

This structure supports both **method comparison** and **cross-region generalization analysis** while keeping the evaluation protocol explicit and reproducible.
